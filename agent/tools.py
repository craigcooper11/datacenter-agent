"""The agent's tools. Three are read-only; restart_container and resize_container pause on a human
approval gate."""

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from agent.backends import Backend

MIN_MEMORY_MIB = 256


def _json(payload: dict[str, Any]) -> str:
    # Compact: tool output is prompt tokens, and small local models have small context windows.
    return json.dumps(payload, default=str, separators=(",", ":"))


@dataclass(frozen=True)
class ActionScope:
    """Which containers the agent may change (restart or resize), always behind the approval gate."""

    vmids: frozenset[int] | None = frozenset()  # None = any container
    protected: frozenset[int] = frozenset()  # never, even if listed

    def allows(self, vmid: int) -> bool:
        return vmid not in self.protected and (self.vmids is None or vmid in self.vmids)

    @property
    def enabled(self) -> bool:
        return self.vmids is None or bool(self.vmids - self.protected)

    def describe(self) -> str:
        if not self.enabled:
            return "none (changes are disabled)"
        base = "any LXC container" if self.vmids is None else ", ".join(map(str, sorted(self.vmids - self.protected)))
        return base + (f", except protected {', '.join(map(str, sorted(self.protected)))}" if self.protected else "")


def check_resize(
    current: dict[str, Any],
    host: dict[str, Any],
    cores: int | None,
    memory_mib: int | None,
    disk_gib: int | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate a resize against the container and host. Returns (changes, problems)."""
    changes: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    if cores is not None and cores != current.get("cores"):
        if not 1 <= cores <= (host.get("cores") or cores):
            problems.append(f"cores must be between 1 and the host's {host.get('cores')}")
        changes["cores"] = {"from": current.get("cores"), "to": cores}
    if memory_mib is not None and memory_mib != current.get("memory_mib"):
        if memory_mib < MIN_MEMORY_MIB:
            problems.append(f"memory must be at least {MIN_MEMORY_MIB} MiB")
        if host.get("memory_mib") and memory_mib > host["memory_mib"]:
            problems.append(f"memory can't exceed the host's {host['memory_mib']} MiB")
        changes["memory_mib"] = {"from": current.get("memory_mib"), "to": memory_mib}
    if disk_gib is not None:
        size = current.get("disk_total_gib") or 0
        if disk_gib <= size:
            problems.append(f"disk can only grow (currently {size} GiB); Proxmox can't shrink a container disk")
        else:
            changes["disk_gib"] = {"from": size, "to": disk_gib}
    if not changes and not problems:
        problems.append("nothing to change: pass the values that differ from the current size")
    return changes, problems


UNITS = {"cores": "", "memory_mib": " MiB", "disk_gib": " GiB"}


def approval_summary(request: dict[str, Any]) -> tuple[str, list[str]]:
    """(title, detail lines) describing a pending action, shared by the Chainlit card and the CLI."""
    current = request.get("current", {})
    verb = {"restart_container": "restart", "resize_container": "resize"}.get(request.get("action"), "change")
    title = f"{verb} CT {request['vmid']} ({current.get('name', '?')})"
    lines = [f"Status: {current.get('status')}, up {current.get('uptime_s')}s"]
    if request.get("action") == "resize_container":
        for key, change in request.get("changes", {}).items():
            unit = UNITS.get(key, "")
            lines.append(
                f"{key.removesuffix('_mib').removesuffix('_gib')}: {change['from']}{unit} → {change['to']}{unit}"
            )
    else:
        lines.append(f"Disk: {current.get('disk_pct')}% of {current.get('disk_total_gib')} GiB")
    lines.append(f"Reason: {request.get('reason')}")
    return title, lines


def make_tools(backend: Backend, scope: ActionScope) -> list[BaseTool]:
    async def preflight(vmid: int, verb: str) -> tuple[dict[str, Any] | None, str | None]:
        """Scope and type checks shared by every action. Returns (current status, error JSON)."""
        if not scope.enabled:
            return None, _json(
                {
                    "executed": False,
                    "error": f"Changes are disabled in this deployment. Do not retry; recommend the {verb} "
                    "to the operator as a manual step instead.",
                }
            )
        if not scope.allows(vmid):
            return None, _json(
                {
                    "executed": False,
                    "error": f"Container {vmid} is outside this agent's permitted scope "
                    f"(may change: {scope.describe()}). Do not retry; report it to the operator.",
                }
            )
        try:
            current = await backend.container_status(vmid)
        except Exception as e:
            return None, _json({"executed": False, "error": f"Could not read container state: {e}"})
        if current.get("type") == "vm":
            return None, _json({"executed": False, "error": f"{vmid} is a VM; only LXC containers can be changed."})
        return current, None

    def gate(action: str, vmid: int, reason: str, current: dict[str, Any], **details: Any) -> bool:
        # Human-in-the-loop gate. The graph checkpoints here and resumes with the operator's decision.
        # On resume the calling tool re-runs from the top, so nothing before this may have side effects.
        decision = interrupt({"action": action, "vmid": vmid, "reason": reason, "current": current, **details})
        return decision.get("approved", False) if isinstance(decision, dict) else bool(decision)

    def denied(verb: str) -> str:
        return _json(
            {
                "executed": False,
                "approved": False,
                "message": f"Operator DENIED the {verb}. Do not retry it; summarize and suggest next steps.",
            }
        )

    @tool
    async def query_disk_usage(vmid: int | None = None, lookback_minutes: int = 30) -> str:
        """Query Grafana (Prometheus, fed by the Proxmox exporter) for guest disk usage.

        Always returns current disk usage % for every LXC container, highest first, then every VM
        with its disk size only (VM usage is unknown without the QEMU guest agent). When `vmid`
        is given, also returns that container's trend over `lookback_minutes`: start/end %,
        growth rate in %-points per minute, and a naive projection of minutes until full.
        """
        try:
            return _json(await backend.disk_usage(vmid, lookback_minutes))
        except Exception as e:
            return _json({"error": f"{type(e).__name__}: {e}"})

    @tool
    async def get_container_status(vmid: int | None = None, name: str | None = None) -> str:
        """Read a container's or VM's live state directly from the Proxmox API.

        Identify the guest by `vmid`, or by `name` when that's all you have (e.g. "kwx"); the result
        includes the vmid to use for any action. Returns run status, uptime, cores, memory, disk
        used/total and %, memory %, and CPU %. Use this to confirm what the metrics suggest before
        proposing any action.
        """
        try:
            if vmid is None:
                if not name:
                    return _json({"error": "Pass a vmid or a name."})
                vmid = await backend.find_vmid(name)
            status = await backend.container_status(vmid)
        except Exception as e:
            return _json({"error": f"{type(e).__name__}: {e}"})
        return _json({**status, "changes_permitted": scope.allows(vmid) and status.get("type") != "vm"})

    @tool
    async def get_resource_usage(vmid: int, lookback_hours: int = 24) -> str:
        """Measure a guest's CPU, memory and disk usage over `lookback_hours`, plus host capacity.

        CPU and memory come as avg/p95/peak % of what's allocated (CPU 100% = all allocated cores
        busy). Disk comes as a growth trend. `host` gives the host's cores and memory and how much is
        already allocated to guests. Use this before any sizing recommendation.
        """
        try:
            return _json(await backend.resource_usage(vmid, lookback_hours))
        except Exception as e:
            return _json({"error": f"{type(e).__name__}: {e}"})

    @tool
    async def restart_container(vmid: int, reason: str) -> str:
        """Restart (or start, if stopped) an LXC container. Requires human approval.

        Execution pauses until the operator approves or denies. `reason` is shown to the
        operator: state the evidence and why a restart should resolve it.
        """
        before, error = await preflight(vmid, "restart")
        if error:
            return error
        if not gate("restart_container", vmid, reason, before):
            return denied("restart")
        try:
            result = await backend.restart_container(vmid)
        except Exception as e:
            return _json({"executed": False, "approved": True, "error": f"Restart failed: {type(e).__name__}: {e}"})
        return _json({"executed": True, "approved": True, "before": before, **result})

    @tool
    async def resize_container(
        vmid: int,
        reason: str,
        cores: int | None = None,
        memory_mib: int | None = None,
        disk_gib: int | None = None,
    ) -> str:
        """Change an LXC container's CPU cores, memory (MiB) and/or root disk size (GiB). Requires
        human approval.

        Pass only the values that change. Disk can only grow. Call this only when the operator asks
        to apply a change; for sizing questions, recommend instead. `reason` is shown to the
        operator: cite the measured usage behind each new value.
        """
        before, error = await preflight(vmid, "resize")
        if error:
            return error
        try:
            host = await backend.host_capacity()
        except Exception as e:
            return _json({"executed": False, "error": f"Could not read host capacity: {e}"})
        changes, problems = check_resize(before, host, cores, memory_mib, disk_gib)
        if problems:
            return _json({"executed": False, "error": "; ".join(problems) + ". Do not retry the same values."})
        if not gate("resize_container", vmid, reason, before, changes=changes):
            return denied("resize")
        try:
            result = await backend.resize_container(vmid, cores, memory_mib, disk_gib)
        except Exception as e:
            return _json({"executed": False, "approved": True, "error": f"Resize failed: {type(e).__name__}: {e}"})
        return _json({"executed": True, "approved": True, "changes": changes, **result})

    return [query_disk_usage, get_container_status, get_resource_usage, restart_container, resize_container]
