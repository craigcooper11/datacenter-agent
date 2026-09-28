"""Chainlit chat UI: `docker compose up -d` (or locally, `uv run chainlit run agent/app.py`)."""

import json
from typing import Any

import chainlit as cl
from chainlit.config import config as chainlit_config

from agent.backends import LiveBackend
from agent.config import get_settings
from agent.demo import NODE as DEMO_NODE
from agent.demo import demo_backend
from agent.graph import build_graph, run_turn, setup_tracing
from agent.tools import ActionScope, approval_summary

AGENT_NAME = "Datacenter Agent"
chainlit_config.ui.name = AGENT_NAME  # page title and assistant name

setup_tracing()
_backend: LiveBackend | None = None


def _get_backend() -> LiveBackend:
    # One shared connection to Grafana/Proxmox per process; graphs (and their memory) are per chat.
    global _backend
    if _backend is None:
        _backend = LiveBackend(get_settings())
    return _backend


async def ask_operator(request: dict[str, Any]) -> bool:
    title, lines = approval_summary(request)
    res = await cl.AskActionMessage(
        content=f"### ⚠️ Approval required: {title}\n" + "\n".join(f"- {line}" for line in lines),
        actions=[
            cl.Action(name="approve", payload={"approved": True}, label=f"Approve {title.split()[0]}"),
            cl.Action(name="deny", payload={"approved": False}, label="Deny"),
        ],
        timeout=600,
    ).send()
    # No answer within the timeout counts as a denial.
    return bool(res and res.get("payload", {}).get("approved"))


STARTERS = [
    # Each shows a different capability, and all work in both live and demo mode.
    cl.Starter(
        label="Investigate disk alerts", message="Disk alerts are firing on the homelab. Investigate and fix it."
    ),
    cl.Starter(label="What fills up first?", message="Which container will run out of disk first, and when?"),
    cl.Starter(label="Full inventory", message="Show me every container and VM with its disk usage."),
    cl.Starter(label="Right-size containers", message="Review the sizing of my containers. What would you change?"),
    cl.Starter(label="Blind spots", message="What can't you see on this cluster, and why?"),
]


def _mode(s) -> tuple[bool, ActionScope, str]:
    """(demo?, restart scope, node name) for the configured backend."""
    if s.backend == "demo":
        return True, ActionScope(vmids=None), DEMO_NODE
    return False, s.action_scope(), s.proxmox_node


@cl.set_starters
async def set_starters(current_user: Any = None, language: str | None = None) -> list[cl.Starter]:
    return STARTERS


@cl.on_chat_start
async def on_chat_start() -> None:
    s = get_settings()
    demo, scope, node = _mode(s)
    if demo:
        # A fresh simulated homelab per chat, so every chat starts with the fault in progress.
        backend = demo_backend()
    else:
        try:
            backend = _get_backend()
        except ValueError as e:
            cl.user_session.set("config_error", str(e))
            await cl.Message(f"⚠️ The agent isn't configured: {e}").send()
            return
    cl.user_session.set("graph", build_graph(backend, scope, node=node))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    graph = cl.user_session.get("graph")
    if graph is None:
        error = cl.user_session.get("config_error") or "see the server logs"
        await cl.Message(
            f"⚠️ The agent isn't configured: {error}\n\nFill these in `.env`, then run `docker compose up -d` "
            "and start a new chat."
        ).send()
        return

    if welcome := cl.user_session.get("welcome"):
        cl.user_session.set("welcome", None)
        await cl.Message(welcome).send()

    steps: dict[str, cl.Step] = {}

    async def on_event(kind: str, call: dict[str, Any]) -> None:
        if kind == "tool_call":
            step = cl.Step(name=call["name"], type="tool", language="json")
            step.input = call["args"]
            await step.send()
            steps[call["id"]] = step
        elif kind == "tool_result" and (step := steps.get(call["id"])):
            output = call["output"]
            step.output = output if isinstance(output, str) else json.dumps(output, indent=2)
            await step.update()

    try:
        result = await run_turn(
            graph, message.content, cl.user_session.get("id"), decide=ask_operator, on_event=on_event
        )
    except Exception as e:
        # Most often a missing/invalid OPENAI_API_KEY or an unreachable model; say so plainly.
        await cl.Message(
            f"⚠️ The turn failed: `{type(e).__name__}: {str(e)[:300]}`\n\n"
            "Check `LLM_PROVIDER` and `OPENAI_API_KEY` in `.env`, then run `docker compose up -d` and start a new chat."
        ).send()
        return
    await cl.Message(result["response"]).send()
