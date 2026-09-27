"""Offline tests: the approval gate, restart scope, and regressions, run through the real graph."""

import asyncio
import json

from conftest import ScriptedLLM, call
from langchain_core.messages import AIMessage

from agent.backends import summarize_trend
from agent.config import Settings
from agent.demo import SANDBOX_VMID, FakeBackend, Guest, demo_backend, homelab
from agent.graph import build_graph, run_turn
from agent.tools import ActionScope, make_tools

RUNAWAY = Guest(SANDBOX_VMID, "sandbox", 96.4, rate_pct_per_min=1.1, leak=True, baseline_pct=41.0)
ALL = ActionScope(vmids=None)


def run(llm, backend, approve=True, scope=ALL, question="Disk alerts are firing. Fix it."):
    graph = build_graph(backend, scope, node="pve-test", llm=llm)
    asked = []

    async def decide(request):
        asked.append(request)
        return approve

    result = asyncio.run(run_turn(graph, question, "test-thread", decide))
    return result, asked


def restart_output(result):
    return next(c["output"] for c in result["tool_calls"] if c["name"] == "restart_container")


# --- The approval gate ------------------------------------------------------------------


def test_approved_restart_executes_and_recovers():
    backend = FakeBackend(homelab(RUNAWAY))
    llm = ScriptedLLM(
        responses=[
            call("query_disk_usage"),
            call("get_container_status", vmid=200),
            call("restart_container", vmid=200, reason="growing 1.1%/min"),
            call("get_container_status", vmid=200),
            AIMessage("Restarted CT 200; disk back to 41%."),
        ]
    )
    result, asked = run(llm, backend, approve=True)
    assert [a["vmid"] for a in asked] == [200]
    assert backend.restarts == [200]
    assert restart_output(result)["executed"] is True
    assert restart_output(result)["status_after"]["disk_pct"] == 41.0
    assert result["approvals"] == [{"request": asked[0], "approved": True}]


def test_denied_restart_does_not_execute():
    backend = FakeBackend(homelab(RUNAWAY))
    llm = ScriptedLLM(
        responses=[
            call("get_container_status", vmid=200),
            call("restart_container", vmid=200, reason="growing"),
            AIMessage("Restart denied; suggest clearing the writer manually."),
        ]
    )
    result, asked = run(llm, backend, approve=False)
    assert len(asked) == 1
    assert backend.restarts == []
    assert restart_output(result) == {
        "executed": False,
        "approved": False,
        "message": "Operator DENIED the restart. Do not retry it; summarize and suggest next steps.",
    }


def _refused_before_gate(scope, vmid):
    """Call restart_container outside a graph: reaching interrupt() there would raise."""
    tools = {t.name: t for t in make_tools(FakeBackend(homelab(RUNAWAY)), scope)}
    return json.loads(asyncio.run(tools["restart_container"].ainvoke({"vmid": vmid, "reason": "x"})))


def test_out_of_scope_is_refused_before_the_gate():
    out = _refused_before_gate(ActionScope(vmids=frozenset({200})), 105)
    assert out["executed"] is False and "outside this agent's permitted scope" in out["error"]


def test_protected_container_is_refused_even_with_all():
    out = _refused_before_gate(ActionScope(vmids=None, protected=frozenset({110})), 110)
    assert out["executed"] is False


def test_vms_are_never_restarted():
    out = _refused_before_gate(ALL, 120)  # 120 is a VM in the simulated homelab
    assert out["executed"] is False and "is a VM" in out["error"]


def test_restarts_disabled_when_scope_empty():
    out = _refused_before_gate(ActionScope(vmids=frozenset()), 200)
    assert out["executed"] is False and "disabled" in out["error"]


# --- Restart scope parsing ---------------------------------------------------------------


def test_action_scope_from_settings():
    def scope(restartable, protected=""):
        s = Settings(proxmox_managed_vmids=restartable, proxmox_protected_vmids=protected)
        return s.action_scope()

    assert scope("all").allows(104) and scope("all").allows(110)
    assert not scope("all", "110").allows(110)
    assert scope("104, 105").allows(105) and not scope("104,105").allows(200)
    assert not scope("").enabled
    assert scope("all", "110").describe() == "any LXC container, except protected 110"


# --- Regressions ---------------------------------------------------------------------------


def test_parallel_tool_calls_are_trimmed_to_one():
    """If a restart shared a step with other calls, they would all re-run on resume."""
    backend = FakeBackend(homelab(RUNAWAY))
    two_calls = AIMessage(
        "",
        tool_calls=[
            {"name": "restart_container", "args": {"vmid": 200, "reason": "x"}, "id": "a"},
            {"name": "restart_container", "args": {"vmid": 101, "reason": "y"}, "id": "b"},
        ],
    )
    result, asked = run(ScriptedLLM(responses=[two_calls, AIMessage("done")]), backend)
    assert [c["id"] for c in result["tool_calls"]] == ["a"]
    assert backend.restarts == [200]


def test_empty_final_answer_gets_one_nudge():
    llm = ScriptedLLM(responses=[call("query_disk_usage"), AIMessage(""), AIMessage("All healthy.")])
    result, _ = run(llm, FakeBackend(homelab()), question="Anything low on disk?")
    assert result["response"] == "All healthy."
    assert llm.calls == 3


# --- Backends -------------------------------------------------------------------------------


def test_summarize_trend_projects_time_to_full():
    points = [(0.0, 90.0), (600.0, 95.0)]  # +5 points over 10 minutes
    trend = summarize_trend(points)
    assert trend["rate_pct_per_min"] == 0.5
    assert trend["minutes_to_full"] == 10.0
    assert summarize_trend([(0.0, 50.0), (600.0, 50.0)])["minutes_to_full"] is None


def test_demo_backend_restart_frees_the_leak():
    backend = demo_backend()
    before = asyncio.run(backend.container_status(SANDBOX_VMID))["disk_pct"]
    after = asyncio.run(backend.restart_container(SANDBOX_VMID))["status_after"]["disk_pct"]
    assert before >= 91.0 and after == 41.0


def test_vms_are_listed_without_a_usage_percentage():
    guests = asyncio.run(FakeBackend(homelab()).disk_usage(None, 30))["guests"]
    vms = [g for g in guests if g["type"] == "vm"]
    assert vms and all(g["disk_pct"] is None for g in vms)


# --- Resizing ---------------------------------------------------------------------------------

HOST = {"cores": 12, "memory_mib": 65536}
MEDIA = {"cores": 2, "memory_mib": 2048, "disk_total_gib": 8.0}


def test_check_resize_rules():
    from agent.tools import check_resize

    changes, problems = check_resize(MEDIA, HOST, cores=None, memory_mib=3072, disk_gib=None)
    assert changes == {"memory_mib": {"from": 2048, "to": 3072}} and not problems
    assert check_resize(MEDIA, HOST, None, None, disk_gib=4)[1]  # disks only grow
    assert check_resize(MEDIA, HOST, None, None, disk_gib=8)[1]  # same size isn't a change
    assert check_resize(MEDIA, HOST, cores=16, memory_mib=None, disk_gib=None)[1]  # more than the host
    assert check_resize(MEDIA, HOST, None, memory_mib=128, disk_gib=None)[1]  # below the minimum
    assert check_resize(MEDIA, HOST, cores=2, memory_mib=2048, disk_gib=None)[1]  # nothing to change


def test_approved_resize_executes_through_the_gate():
    backend = FakeBackend(homelab())
    llm = ScriptedLLM(
        responses=[
            call("get_resource_usage", vmid=105),
            call("resize_container", vmid=105, memory_mib=3072, reason="memory peak 97%"),
            call("get_container_status", vmid=105),
            AIMessage("Resized media to 3 GiB."),
        ]
    )
    result, asked = run(llm, backend, approve=True, question="Give media more RAM.")
    assert asked[0]["action"] == "resize_container"
    assert asked[0]["changes"] == {"memory_mib": {"from": 2048, "to": 3072}}
    assert backend.resizes == [(105, {"cores": None, "memory_mib": 3072, "disk_gib": None})]
    assert backend.guests[105].memory_mib == 3072


def test_denied_resize_changes_nothing():
    backend = FakeBackend(homelab())
    llm = ScriptedLLM(responses=[call("resize_container", vmid=105, cores=3, reason="cpu"), AIMessage("Denied.")])
    result, asked = run(llm, backend, approve=False)
    assert len(asked) == 1 and backend.resizes == [] and backend.guests[105].cores == 2


def test_disk_shrink_is_refused_before_the_gate():
    tools = {t.name: t for t in make_tools(FakeBackend(homelab()), ALL)}
    out = json.loads(asyncio.run(tools["resize_container"].ainvoke({"vmid": 200, "disk_gib": 4, "reason": "x"})))
    assert out["executed"] is False and "only grow" in out["error"]


def test_resource_usage_reports_sizing_numbers():
    usage = asyncio.run(FakeBackend(homelab()).resource_usage(105, 24))
    assert usage["memory"]["peak_pct"] > 90  # media is memory-starved
    assert usage["cpu"]["cores"] == 2
    assert usage["host"]["allocated_cores"] > 0


def test_approval_summary_describes_resizes():
    from agent.tools import approval_summary

    title, lines = approval_summary(
        {
            "action": "resize_container",
            "vmid": 105,
            "reason": "peak 97%",
            "current": {"name": "media"},
            "changes": {"memory_mib": {"from": 2048, "to": 3072}},
        }
    )
    assert title == "resize CT 105 (media)" and "memory: 2048 MiB → 3072 MiB" in lines


def test_truncated_answer_is_flagged():
    cut = AIMessage("Recommendations: grow media to 3 GiB and", response_metadata={"finish_reason": "length"})
    result, _ = run(ScriptedLLM(responses=[cut]), FakeBackend(homelab()), question="Sizing?")
    assert result["response"].startswith("Recommendations") and "Answer cut off" in result["response"]


def test_proxmox_task_with_warnings_counts_as_success():
    from agent.backends import _task_succeeded

    assert _task_succeeded({"exitstatus": "OK"})
    assert _task_succeeded({"exitstatus": "WARNINGS: 1"})
    assert not _task_succeeded({"exitstatus": "command 'lxc-start' failed: exit code 1"})
    assert not _task_succeeded(None)


def test_narrated_action_gets_one_nudge():
    """Small models sometimes write 'Calling restart_container' instead of calling it."""
    backend = FakeBackend(homelab())
    llm = ScriptedLLM(
        responses=[
            call("get_container_status", vmid=200),
            AIMessage("Action taken: calling restart_container for vmid=200."),  # narrated, not called
            call("restart_container", vmid=200, reason="operator asked"),
            call("get_container_status", vmid=200),
            AIMessage("Restarted CT 200."),
        ]
    )
    result, asked = run(llm, backend, approve=True, question="Restart container 200.")
    assert [a["vmid"] for a in asked] == [200] and backend.restarts == [200]
    assert result["response"] == "Restarted CT 200."


def test_status_by_name_resolves_the_vmid():
    tools = {t.name: t for t in make_tools(FakeBackend(homelab()), ALL)}
    out = json.loads(asyncio.run(tools["get_container_status"].ainvoke({"name": "media"})))
    assert out["vmid"] == 105 and out["name"] == "media"
    missing = json.loads(asyncio.run(tools["get_container_status"].ainvoke({"name": "nope"})))
    assert "No guest named 'nope'" in missing["error"] and "media (105)" in missing["error"]


def test_accepted_restart_is_not_reported_as_not_executed(monkeypatch):
    """Proxmox accepts the reboot, then the connection drops (e.g. an IP conflict): the reboot still
    happened, so it must come back as executed with a verification error, never as 'not executed'."""
    from types import SimpleNamespace

    import agent.backends as backends

    class Container:
        status = SimpleNamespace(
            current=SimpleNamespace(get=lambda: {"status": "running"}),
            reboot=SimpleNamespace(post=lambda: "UPID:pve:reboot"),
        )

    def refused(*args, **kwargs):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr(backends.Tasks, "blocking_status", refused)
    live = object.__new__(backends.LiveBackend)
    live._pve = None
    live._ct = lambda vmid: Container()
    result = live._restart_blocking(200)
    assert result["task"] == "UPID:pve:reboot" and "Connection refused" in result["verification_error"]

    backend = FakeBackend(homelab())

    async def accepted_then_lost(vmid):
        backend.restarts.append(vmid)
        return result

    backend.restart_container = accepted_then_lost
    llm = ScriptedLLM(
        responses=[call("restart_container", vmid=200, reason="x"), AIMessage("Restart sent; outcome unconfirmed.")]
    )
    out, _ = run(llm, backend, approve=True)
    assert restart_output(out)["executed"] is True and "verification_error" in restart_output(out)
