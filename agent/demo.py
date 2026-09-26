"""A simulated Proxmox homelab: the backend for demo mode (BACKEND=demo) and for evals.

It implements the same `Backend` protocol as `LiveBackend`, so the real graph, tools, prompt and
approval gate run unchanged; only the infrastructure is simulated. Nothing here touches a network.
"""

import time
from dataclasses import dataclass
from typing import Any

from agent.backends import VM_USAGE_UNKNOWN, summarize_trend

NODE = "pve-demo"
SANDBOX_VMID = 200
HOST = {"cores": 12, "memory_mib": 65536, "memory_used_pct": 58.0}
# VMs: disk size only (no guest agent), but Proxmox does report their CPU and memory.
# vmid: (name, disk GiB, cores, memory MiB)
VMS = {120: ("gitea", 400.0, 2, 4096), 130: ("ollama-server", 100.0, 8, 32768)}
SOURCE = "simulated homelab (demo mode)"


@dataclass
class Guest:
    vmid: int
    name: str
    disk_pct: float
    rate_pct_per_min: float = 0.0
    # True when the usage is held by a process, so a restart frees it (down to `baseline_pct`).
    leak: bool = False
    baseline_pct: float = 40.0
    status: str = "running"
    uptime_s: int = 86_400
    # Size and a 24h usage profile, in absolute units so a resize changes the percentages.
    disk_gib: float = 8.0
    cores: int = 1
    memory_mib: int = 512
    cpu_used_cores: tuple[float, float, float] = (0.05, 0.1, 0.2)  # avg, p95, peak
    mem_used_mib: tuple[float, float] = (200.0, 260.0)  # avg, peak


def homelab(*overrides: Guest) -> list[Guest]:
    """Four containers with distinct sizing stories, with per-scenario overrides:
    pihole about right, media memory-starved, home-assistant over-provisioned, sandbox fine."""
    base = {
        101: Guest(101, "pihole", 34.2, cpu_used_cores=(0.03, 0.08, 0.15), mem_used_mib=(200, 280)),
        105: Guest(
            105, "media", 62.8, cores=2, memory_mib=2048, cpu_used_cores=(0.9, 1.7, 2.0), mem_used_mib=(1800, 1990)
        ),
        110: Guest(
            110,
            "home-assistant",
            48.1,
            cores=4,
            memory_mib=4096,
            cpu_used_cores=(0.08, 0.15, 0.4),
            mem_used_mib=(820, 1150),
        ),
        SANDBOX_VMID: Guest(SANDBOX_VMID, "sandbox", 41.0, cpu_used_cores=(0.2, 0.35, 0.5), mem_used_mib=(180, 230)),
    }
    base.update({g.vmid: g for g in overrides})
    return list(base.values())


def _pct(used: float, total: float) -> float:
    return round(min(100.0, 100 * used / total), 1) if total else 0.0


class FakeBackend:
    def __init__(self, guests: list[Guest], grafana_error: str | None = None, live: bool = False):
        self.guests = {g.vmid: g for g in guests}
        self.grafana_error = grafana_error
        self.restarts: list[int] = []
        self.resizes: list[tuple[int, dict[str, Any]]] = []
        # live=True: disk keeps growing with wall-clock time, like a real incident. Evals leave it
        # off so every run sees the same numbers.
        self._t0 = time.monotonic() if live else None

    def _minutes(self) -> float:
        return 0.0 if self._t0 is None else (time.monotonic() - self._t0) / 60

    def _disk_pct(self, g: Guest) -> float:
        return round(min(99.9, g.disk_pct + g.rate_pct_per_min * self._minutes()), 1)

    def _disk_points(self, g: Guest, lookback_minutes: int) -> list[tuple[float, float]]:
        now, end, n = time.time(), self._disk_pct(g), 30
        points = []
        for i in range(n + 1):
            ago = lookback_minutes * (n - i) / n
            points.append((now - ago * 60, max(0.0, end - g.rate_pct_per_min * ago)))
        return points

    async def disk_usage(self, vmid: int | None, lookback_minutes: int) -> dict[str, Any]:
        if self.grafana_error:
            raise RuntimeError(self.grafana_error)
        guests = sorted(self.guests.values(), key=self._disk_pct, reverse=True)
        result: dict[str, Any] = {
            "source": SOURCE if self._t0 is not None else "grafana/prometheus (pve-exporter)",
            "guests": [{"vmid": g.vmid, "name": g.name, "type": "lxc", "disk_pct": self._disk_pct(g)} for g in guests]
            + [
                {"vmid": v, "name": n, "type": "vm", "disk_total_gib": size, "disk_pct": None, "note": VM_USAGE_UNKNOWN}
                for v, (n, size, _, _) in VMS.items()
            ],
        }
        if vmid is not None:
            g = self.guests.get(vmid)
            points = self._disk_points(g, lookback_minutes) if g else []
            result["trend"] = {"vmid": vmid, "lookback_minutes": lookback_minutes, **summarize_trend(points)}
        return result

    async def container_status(self, vmid: int) -> dict[str, Any]:
        if vmid in VMS:
            name, size, cores, memory = VMS[vmid]
            return {
                "vmid": vmid,
                "name": name,
                "type": "vm",
                "status": "running",
                "uptime_s": 604_800,
                "disk_used_gib": None,
                "disk_total_gib": size,
                "disk_pct": None,
                "mem_pct": 55.0,
                "cpu_pct": 4.0,
                "cores": cores,
                "memory_mib": memory,
                "note": VM_USAGE_UNKNOWN,
            }
        g = self.guests.get(vmid)
        if g is None:
            raise LookupError(f"No container or VM with ID {vmid} on this cluster")
        pct = self._disk_pct(g)
        return {
            "vmid": vmid,
            "name": g.name,
            "type": "lxc",
            "status": g.status,
            "uptime_s": g.uptime_s,
            "disk_used_gib": round(g.disk_gib * pct / 100, 2),
            "disk_total_gib": g.disk_gib,
            "disk_pct": pct,
            "mem_pct": _pct(g.mem_used_mib[0], g.memory_mib),
            "cpu_pct": _pct(g.cpu_used_cores[0], g.cores),
            "cores": g.cores,
            "memory_mib": g.memory_mib,
        }

    async def host_capacity(self) -> dict[str, Any]:
        return {
            **HOST,
            "allocated_cores": sum(g.cores for g in self.guests.values()) + sum(v[2] for v in VMS.values()),
            "allocated_memory_mib": sum(g.memory_mib for g in self.guests.values()) + sum(v[3] for v in VMS.values()),
        }

    async def resource_usage(self, vmid: int, lookback_hours: int) -> dict[str, Any]:
        if self.grafana_error:
            raise RuntimeError(self.grafana_error)
        host = await self.host_capacity()
        if vmid in VMS:
            _, _, cores, memory = VMS[vmid]
            cpu, mem, disk, kind = (0.3, 0.6, 1.2), (0.55 * memory, 0.7 * memory), {"note": VM_USAGE_UNKNOWN}, "vm"
        elif (g := self.guests.get(vmid)) is not None:
            cores, memory, kind = g.cores, g.memory_mib, "lxc"
            cpu, mem = g.cpu_used_cores, g.mem_used_mib
            disk = {"size_gib": g.disk_gib, **summarize_trend(self._disk_points(g, 60))}
        else:
            raise LookupError(f"No metrics for guest {vmid}")
        return {
            "vmid": vmid,
            "type": kind,
            "source": SOURCE if self._t0 is not None else "grafana/prometheus (pve-exporter)",
            "window_hours": lookback_hours,
            "cpu": {
                "cores": cores,
                "avg_pct": _pct(cpu[0], cores),
                "p95_pct": _pct(cpu[1], cores),
                "peak_pct": _pct(cpu[2], cores),
            },
            "memory": {"allocated_mib": memory, "avg_pct": _pct(mem[0], memory), "peak_pct": _pct(mem[1], memory)},
            "disk": disk,
            "host": host,
        }

    async def restart_container(self, vmid: int) -> dict[str, Any]:
        g = self.guests[vmid]
        self.restarts.append(vmid)
        operation = "start" if g.status == "stopped" else "reboot"
        if g.leak:
            g.disk_pct, g.rate_pct_per_min, g.leak = g.baseline_pct, 0.0, False
        g.status, g.uptime_s = "running", 12
        return {"operation": operation, "task": f"UPID:{NODE}:fake", "status_after": await self.container_status(vmid)}

    async def resize_container(
        self, vmid: int, cores: int | None, memory_mib: int | None, disk_gib: int | None
    ) -> dict[str, Any]:
        g = self.guests[vmid]
        self.resizes.append((vmid, {"cores": cores, "memory_mib": memory_mib, "disk_gib": disk_gib}))
        g.cores = cores or g.cores
        g.memory_mib = memory_mib or g.memory_mib
        if disk_gib:
            # Same data on a bigger disk; rebase so the current reading drops accordingly.
            now_pct = self._disk_pct(g) * g.disk_gib / disk_gib
            g.disk_gib = float(disk_gib)
            g.disk_pct = now_pct - g.rate_pct_per_min * self._minutes()
        return {"status_after": await self.container_status(vmid)}


def demo_backend() -> FakeBackend:
    """The demo-mode homelab: the sandbox has a runaway writer filling its disk at ~1.1%/min
    (a deleted-but-open file, so a restart frees it); media is short on memory; home-assistant is
    over-provisioned."""
    return FakeBackend(
        homelab(
            Guest(
                SANDBOX_VMID,
                "sandbox",
                91.0,
                rate_pct_per_min=1.1,
                leak=True,
                baseline_pct=41.0,
                cpu_used_cores=(0.2, 0.35, 0.5),
                mem_used_mib=(180, 230),
            )
        ),
        live=True,
    )
