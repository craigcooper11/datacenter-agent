"""Live infrastructure access: Grafana (via mcp-grafana) for metrics, Proxmox (via proxmoxer) for state and power.

Tools depend on the `Backend` protocol rather than this module directly, so the same agent
graph runs against real infra (CLI) or a scripted fixture backend (evals).
"""

import asyncio
import json
import time
from typing import Any, Protocol

from langchain_mcp_adapters.client import MultiServerMCPClient
from proxmoxer import ProxmoxAPI
from proxmoxer.tools import Tasks

from agent.config import Settings


class Backend(Protocol):
    async def disk_usage(self, vmid: int | None, lookback_minutes: int) -> dict[str, Any]: ...
    async def container_status(self, vmid: int) -> dict[str, Any]: ...
    async def find_vmid(self, name: str) -> int: ...
    async def resource_usage(self, vmid: int, lookback_hours: int) -> dict[str, Any]: ...
    async def host_capacity(self) -> dict[str, Any]: ...
    async def restart_container(self, vmid: int) -> dict[str, Any]: ...
    async def resize_container(
        self, vmid: int, cores: int | None, memory_mib: int | None, disk_gib: int | None
    ) -> dict[str, Any]: ...


def summarize_trend(points: list[tuple[float, float]]) -> dict[str, Any]:
    """Reduce a disk-% time series to the numbers an LLM can reason about."""
    if not points:
        return {"samples": 0}
    (t0, first), (t1, last) = points[0], points[-1]
    values = [v for _, v in points]
    minutes = max((t1 - t0) / 60, 1e-9)
    rate = (last - first) / minutes
    return {
        "samples": len(points),
        "window_minutes": round((t1 - t0) / 60, 1),
        "start_pct": round(first, 1),
        "end_pct": round(last, 1),
        "min_pct": round(min(values), 1),
        "max_pct": round(max(values), 1),
        "rate_pct_per_min": round(rate, 3),
        # Naive linear projection; None when flat or shrinking.
        "minutes_to_full": round((100 - last) / rate, 1) if rate > 0.01 else None,
    }


def summarize_usage(values: list[float]) -> dict[str, Any]:
    """Average / p95 / peak of a usage-% series: the numbers sizing decisions are made from."""
    if not values:
        return {"samples": 0}
    ordered = sorted(values)
    return {
        "samples": len(values),
        "avg_pct": round(sum(values) / len(values), 1),
        "p95_pct": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
        "peak_pct": round(ordered[-1], 1),
    }


def _vmid_from_id(guest_id: str) -> int | None:
    # pve-exporter ids look like "lxc/101" or "qemu/100"
    try:
        return int(guest_id.split("/", 1)[1])
    except (IndexError, ValueError):
        return None


VM_USAGE_UNKNOWN = "usage unknown: Proxmox can't see inside a VM without the QEMU guest agent"


def _format_status(vmid: int, raw: dict[str, Any], kind: str = "lxc") -> dict[str, Any]:
    gib = 1024**3
    maxdisk = raw.get("maxdisk") or 0
    maxmem = raw.get("maxmem") or 0
    status = {
        "vmid": vmid,
        "name": raw.get("name"),
        "type": kind,
        "status": raw.get("status"),
        "uptime_s": raw.get("uptime"),
        "disk_used_gib": round((raw.get("disk") or 0) / gib, 2),
        "disk_total_gib": round(maxdisk / gib, 2),
        "disk_pct": round(100 * (raw.get("disk") or 0) / maxdisk, 1) if maxdisk else None,
        "mem_pct": round(100 * (raw.get("mem") or 0) / maxmem, 1) if maxmem else None,
        "cpu_pct": round(100 * (raw.get("cpu") or 0), 1),
        "cores": raw.get("cpus"),
        "memory_mib": round(maxmem / 1024**2) if maxmem else None,
    }
    if kind == "vm":
        # Proxmox reports 0 used for VMs; don't let that read as an empty disk.
        status.update(disk_used_gib=None, disk_pct=None, note=VM_USAGE_UNKNOWN)
    return status


def _task_succeeded(task: dict[str, Any] | None) -> bool:
    # Proxmox reports "OK", or "WARNINGS: n" for a task that succeeded with warnings.
    status = (task or {}).get("exitstatus") or ""
    return status == "OK" or status.startswith("WARNINGS")


class LiveBackend:
    def __init__(self, settings: Settings):
        required = {
            "PROXMOX_HOST": settings.proxmox_host,
            "PROXMOX_USER": settings.proxmox_user,
            "PROXMOX_TOKEN_NAME": settings.proxmox_token_name,
            "PROXMOX_TOKEN_VALUE": settings.proxmox_token_value,
            "PROXMOX_NODE": settings.proxmox_node,
            "GRAFANA_MCP_URL": settings.grafana_mcp_url,
        }
        missing = [k for k, v in required.items() if v in (None, "")]
        if missing:
            raise ValueError(f"Missing required settings in .env: {', '.join(missing)}")

        self._s = settings
        connection = {"url": settings.grafana_mcp_url, "transport": "streamable_http"}
        if settings.grafana_mcp_token:
            connection["headers"] = {"Authorization": f"Bearer {settings.grafana_mcp_token}"}
        self._mcp = MultiServerMCPClient({"grafana": connection})
        self._prom_tool = None
        self._pve = ProxmoxAPI(
            settings.proxmox_host,
            user=settings.proxmox_user,
            token_name=settings.proxmox_token_name,
            token_value=settings.proxmox_token_value,
            verify_ssl=settings.proxmox_verify_ssl,
            timeout=15,
        )

    # --- Grafana / Prometheus via mcp-grafana -------------------------------------------

    async def _query(self, expr: str, **params: Any) -> list[dict[str, Any]]:
        if self._prom_tool is None:
            tools = await self._mcp.get_tools()
            self._prom_tool = next(t for t in tools if t.name == "query_prometheus")
        raw = await self._prom_tool.ainvoke(
            {"datasourceUid": self._s.grafana_datasource_uid, "expr": expr, "endTime": "now", **params}
        )
        # langchain-mcp-adapters returns a list of content blocks; the payload is JSON text.
        text = raw if isinstance(raw, str) else "".join(b.get("text", "") for b in raw if isinstance(b, dict))
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # mcp-grafana reports errors (bad datasource, bad PromQL) as plain text.
            raise RuntimeError(f"Grafana query failed: {text}") from None
        return payload.get("data") or []

    async def _disk_series(self, expr: str | None = None, **params: Any) -> list[dict[str, Any]]:
        expr = expr or self._s.disk_usage_promql
        if self._s.guest_info_promql:
            joined = f"({expr}) * on(id) group_left(name) ({self._s.guest_info_promql})"
            series = await self._query(joined, **params)
            if series:
                return series
        return await self._query(expr, **params)

    async def disk_usage(self, vmid: int | None, lookback_minutes: int) -> dict[str, Any]:
        current = await self._disk_series(queryType="instant")
        guests = sorted(
            (
                {
                    "vmid": _vmid_from_id(s["metric"].get("id", "")),
                    "name": s["metric"].get("name"),
                    "type": "lxc",
                    "disk_pct": round(float(s["value"][1]), 1),
                }
                for s in current
            ),
            key=lambda g: g["disk_pct"],
            reverse=True,
        )
        try:
            vms = await self._disk_series(self._s.vm_disk_size_promql, queryType="instant")
        except Exception:
            vms = []  # VMs are informational; never let them break the container view
        guests += sorted(
            (
                {
                    "vmid": _vmid_from_id(s["metric"].get("id", "")),
                    "name": s["metric"].get("name"),
                    "type": "vm",
                    "disk_total_gib": round(float(s["value"][1]) / 1024**3, 1),
                    "disk_pct": None,
                    "note": VM_USAGE_UNKNOWN,
                }
                for s in vms
            ),
            key=lambda g: g["vmid"] or 0,
        )
        result: dict[str, Any] = {"source": "grafana/prometheus (pve-exporter)", "guests": guests}

        if vmid is not None:
            step = max(15, lookback_minutes)  # ~60 points over the window
            history = await self._disk_series(queryType="range", startTime=f"now-{lookback_minutes}m", stepSeconds=step)
            match = next((s for s in history if _vmid_from_id(s["metric"].get("id", "")) == vmid), None)
            points = [(float(t), float(v)) for t, v in match["values"]] if match else []
            result["trend"] = {"vmid": vmid, "lookback_minutes": lookback_minutes, **summarize_trend(points)}
        return result

    async def host_capacity(self) -> dict[str, Any]:
        """Host totals, and how much of them is already allocated to guests (single-node cluster)."""
        node, guests = 'id=~"node/.*"', 'id=~"(lxc|qemu)/.*"'
        cores, mem, mem_used, alloc_cores, alloc_mem = await asyncio.gather(
            self._query(f"pve_cpu_usage_limit{{{node}}}", queryType="instant"),
            self._query(f"pve_memory_size_bytes{{{node}}}", queryType="instant"),
            self._query(f"pve_memory_usage_bytes{{{node}}}", queryType="instant"),
            self._query(f"sum(pve_cpu_usage_limit{{{guests}}})", queryType="instant"),
            self._query(f"sum(pve_memory_size_bytes{{{guests}}})", queryType="instant"),
        )

        def first(series: list[dict[str, Any]]) -> float:
            return float(series[0]["value"][1]) if series else 0.0

        mib = 1024**2
        return {
            "cores": int(first(cores)),
            "memory_mib": round(first(mem) / mib),
            "memory_used_pct": round(100 * first(mem_used) / first(mem), 1) if first(mem) else None,
            "allocated_cores": int(first(alloc_cores)),
            "allocated_memory_mib": round(first(alloc_mem) / mib),
        }

    async def resource_usage(self, vmid: int, lookback_hours: int) -> dict[str, Any]:
        gid = f'id=~"(lxc|qemu)/{vmid}"'
        rng = {"queryType": "range", "startTime": f"now-{lookback_hours}h", "stepSeconds": max(60, lookback_hours * 15)}
        cpu, mem_pct, cores, mem_size, disk, host = await asyncio.gather(
            self._query(f"100 * pve_cpu_usage_ratio{{{gid}}}", **rng),
            self._query(f"100 * pve_memory_usage_bytes{{{gid}}} / pve_memory_size_bytes{{{gid}}}", **rng),
            self._query(f"pve_cpu_usage_limit{{{gid}}}", queryType="instant"),
            self._query(f"pve_memory_size_bytes{{{gid}}}", queryType="instant"),
            self._query(f"100 * pve_disk_usage_bytes{{{gid}}} / pve_disk_size_bytes{{{gid}}}", **rng),
            self.host_capacity(),
        )
        if not cpu and not cores:
            raise LookupError(f"No metrics for guest {vmid}")

        def values(series: list[dict[str, Any]]) -> list[float]:
            return [float(v) for _, v in series[0]["values"]] if series else []

        is_vm = bool(cores) and cores[0]["metric"].get("id", "").startswith("qemu/")
        disk_points = [(float(t), float(v)) for t, v in disk[0]["values"]] if disk else []
        return {
            "vmid": vmid,
            "type": "vm" if is_vm else "lxc",
            "source": "grafana/prometheus (pve-exporter)",
            "window_hours": lookback_hours,
            # CPU % is of the guest's allocated cores (100% = all of them busy).
            "cpu": {"cores": int(float(cores[0]["value"][1])) if cores else None, **summarize_usage(values(cpu))},
            "memory": {
                "allocated_mib": round(float(mem_size[0]["value"][1]) / 1024**2) if mem_size else None,
                **summarize_usage(values(mem_pct)),
            },
            "disk": {"note": VM_USAGE_UNKNOWN} if is_vm else summarize_trend(disk_points),
            "host": host,
        }

    # --- Proxmox via proxmoxer --------------------------------------------------------------

    def _ct(self, vmid: int):
        return self._pve.nodes(self._s.proxmox_node).lxc(vmid)

    def _status_blocking(self, vmid: int) -> dict[str, Any]:
        node = self._pve.nodes(self._s.proxmox_node)
        kind = next(
            (
                "lxc" if g["type"] == "lxc" else "vm"
                for g in self._pve.cluster.resources.get(type="vm")
                if g["vmid"] == vmid
            ),
            None,
        )
        if kind is None:
            raise LookupError(f"No container or VM with ID {vmid} on this cluster")
        guest = node.lxc(vmid) if kind == "lxc" else node.qemu(vmid)
        return _format_status(vmid, guest.status.current.get(), kind)

    async def container_status(self, vmid: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._status_blocking, vmid)

    def _find_vmid_blocking(self, name: str) -> int:
        guests = self._pve.cluster.resources.get(type="vm")
        matches = [g["vmid"] for g in guests if (g.get("name") or "").lower() == name.strip().lower()]
        if len(matches) != 1:
            names = ", ".join(sorted(f"{g.get('name')} ({g['vmid']})" for g in guests))
            raise LookupError(f"{'No' if not matches else 'More than one'} guest named {name!r}. Guests: {names}")
        return matches[0]

    async def find_vmid(self, name: str) -> int:
        return await asyncio.to_thread(self._find_vmid_blocking, name)

    def _restart_blocking(self, vmid: int) -> dict[str, Any]:
        ct = self._ct(vmid)
        was = ct.status.current.get().get("status")
        operation = "start" if was == "stopped" else "reboot"
        upid = getattr(ct.status, operation).post()

        task = Tasks.blocking_status(self._pve, upid, timeout=120, polling_interval=2)
        if task is None:
            raise TimeoutError(f"Proxmox task {upid} did not finish within 120s")
        if not _task_succeeded(task):
            raise RuntimeError(f"Proxmox {operation} failed: {task.get('exitstatus')}")

        # The task finishing doesn't mean the guest is back; wait briefly for it to report running.
        for _ in range(15):
            after = ct.status.current.get()
            if after.get("status") == "running":
                break
            time.sleep(2)
        result = {"operation": operation, "task": upid, "status_after": _format_status(vmid, after)}
        if task.get("exitstatus") != "OK":
            result["task_warnings"] = f"{task['exitstatus']} (succeeded; see the task log in Proxmox)"
        return result

    async def restart_container(self, vmid: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._restart_blocking, vmid)

    def _resize_blocking(
        self, vmid: int, cores: int | None, memory_mib: int | None, disk_gib: int | None
    ) -> dict[str, Any]:
        ct = self._ct(vmid)
        config = {k: v for k, v in {"cores": cores, "memory": memory_mib}.items() if v is not None}
        if config:
            ct.config.put(**config)  # applies live to a running container
        if disk_gib is not None:
            # Grow-only; newer Proxmox runs this as a task.
            upid = ct.resize.put(disk="rootfs", size=f"{disk_gib}G")
            if isinstance(upid, str) and upid.startswith("UPID"):
                task = Tasks.blocking_status(self._pve, upid, timeout=120, polling_interval=2)
                if not _task_succeeded(task):
                    raise RuntimeError(f"Disk resize failed: {task and task.get('exitstatus')}")
        return {"status_after": self._status_blocking(vmid)}

    async def resize_container(
        self, vmid: int, cores: int | None, memory_mib: int | None, disk_gib: int | None
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._resize_blocking, vmid, cores, memory_mib, disk_gib)
