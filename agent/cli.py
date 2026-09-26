"""Terminal REPL for the agent: `uv run python -m agent.cli`."""

import asyncio
import json
import uuid
from typing import Any

from agent.backends import LiveBackend
from agent.config import get_settings, model_label
from agent.demo import NODE as DEMO_NODE
from agent.demo import demo_backend
from agent.graph import build_graph, run_turn, setup_tracing
from agent.tools import ActionScope, approval_summary

DIM, BOLD, YELLOW, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[0m"


def _short(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[:limit] + "…"


def on_event(kind: str, call: dict[str, Any]) -> None:
    if kind == "tool_call":
        args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
        print(f"{DIM}→ {call['name']}({args}){RESET}")
    elif kind == "tool_result":
        print(f"{DIM}  ← {_short(call['output'])}{RESET}")


async def ask_operator(request: dict[str, Any]) -> bool:
    title, lines = approval_summary(request)
    print(f"\n{YELLOW}{BOLD}⚠  Approval required: {title}{RESET}")
    for line in lines:
        print(f"{YELLOW}   {line}{RESET}")
    answer = await asyncio.to_thread(input, f"{BOLD}   Approve? [y/N] {RESET}")
    return answer.strip().lower() in {"y", "yes"}


async def main() -> None:
    s = get_settings()
    setup_tracing()
    if s.backend == "demo":
        backend, scope, node = demo_backend(), ActionScope(vmids=None), DEMO_NODE
        print(f"{YELLOW}Demo mode: simulated homelab; CT 200 'sandbox' is filling up. Nothing real is touched.{RESET}")
    else:
        backend, scope, node = LiveBackend(s), s.action_scope(), s.proxmox_node
    graph = build_graph(backend, scope, node=node)
    thread_id = str(uuid.uuid4())

    print(
        f"{BOLD}Datacenter Agent{RESET} · node {node} · may restart: {scope.describe()}"
        f' · model {model_label(s)}\nAsk e.g. "Why is disk filling up?"  (Ctrl-D to quit)\n'
    )
    while True:
        try:
            question = (await asyncio.to_thread(input, f"{BOLD}you ›{RESET} ")).strip()
        except EOFError:
            print()
            return
        if not question:
            continue
        result = await run_turn(graph, question, thread_id, decide=ask_operator, on_event=on_event)
        print(f"\n{result['response']}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
