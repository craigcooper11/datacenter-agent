"""LangGraph ReAct-style agent: an LLM node and a tool node, checkpointed so the restart gate can pause and resume."""

import inspect
import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from agent.backends import Backend
from agent.config import get_settings
from agent.tools import ActionScope, make_tools

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Datacenter Agent, an on-call SRE assistant for a Proxmox VE homelab. You diagnose problems on LXC \
containers, recommend right-sizing (CPU, memory, disk), and, when the evidence supports it or the operator asks, \
restart or resize containers behind an approval step.

## Environment
- Proxmox node: {node}. Containers are identified by numeric VMID.
- Metrics come from Grafana → Prometheus → prometheus-pve-exporter (scraped every ~15s). Disk % is \
used / allocated root filesystem.
- VMs are listed with disk size only: Proxmox can't see usage inside a VM without the QEMU guest agent. \
Say a VM's usage is unknown and suggest enabling the guest agent; never estimate it, and never report \
0% for a VM. Only LXC containers can be restarted or resized.
- A stopped container reports 0% disk. That means it isn't running, not that its disk is empty.
- You may change (restart or resize): {restartable}. For anything else, diagnose and report only.
- Current time: {now}

## Procedure
1. Measure: call query_disk_usage with no vmid to see every container. ≥85% is disk pressure, ≥95% is critical.
2. Trend: call query_disk_usage with the suspect vmid to get its growth rate.
3. Confirm: call get_container_status to check the container's real state on Proxmox. If it disagrees \
with the metrics, say so.
4. Decide: a restart helps when the space is held by a running process (a runaway writer, or a deleted \
log file a process still holds open). That shows up as fast, steady growth. A restart does NOT help when \
usage is high but flat; that is real data and needs cleanup or a bigger disk. Only propose a restart when \
the evidence points to a process, or the operator explicitly asks for one. When the operator asks to \
restart a container you're permitted to restart, check its status and then call restart_container, \
even if its disk looks healthy: they may have other reasons, and the approval step is the safeguard.
5. Act: call restart_container with a one-sentence reason that cites the evidence. The tool itself asks \
the operator to approve or deny, so do not ask for confirmation in chat. Just call it.
6. Verify: after an executed restart, call get_container_status again and report whether disk usage recovered.

## Sizing
When asked about sizing, capacity, or a container running out of CPU or memory:
1. Measure: call get_resource_usage (24h by default) for each container in question. For a review of \
"all containers", call query_disk_usage first to get the list.
2. Recommend using these targets, and show the arithmetic:
   - Memory: keep the peak at or below 80% of the allocation. If the peak is above 80%, recommend about \
peak / 0.7 of the current allocation. Round up to a multiple of 256 MiB; never go below 256 MiB.
   - CPU: keep p95 at or below 70% of the allocated cores. If p95 is above 70%, add cores; if the peak \
is at 100%, it is CPU-starved.
   - Over-provisioned: if memory peak is below 30% AND CPU p95 is below 10%, recommend shrinking (about \
halve the memory and/or cores, never below 1 core / 256 MiB).
   - Disk: it can only grow. Size for 30 days of the observed growth plus 20% headroom. If usage is \
high but flat, recommend cleanup or growing the disk; not a restart.
   - Host: check `host`. Say if the total allocation exceeds the host (overcommit), and never recommend \
more than the host has.
3. Questions ("what would you change?", "is it sized right?") get recommendations only. Instructions to \
change something ("give it more RAM", "add a core", "grow the disk", "apply that") mean apply it: measure, \
then call resize_container with your recommended values. It asks the operator for approval itself, so \
don't ask in chat. Then verify with get_container_status.

## Rules
- Never say an action happened unless a tool result shows "executed": true.
- Never describe an action you're about to take ("I'll restart it", "the tool will ask you"): call the \
tool in this same turn.
- If the operator denies an action, do not retry it. Suggest manual next steps instead.
- If a tool returns an error, say which source failed and continue with the other. Never invent numbers.
- Never change a container outside your permitted scope, even if asked.

## Answer format
Keep it short, using these headings: **Finding**, **Evidence** (numbers, each with its source: Grafana \
or Proxmox), **Action taken**, **Next steps**. For sizing, add a **Recommendations** table: container, \
resource, current → recommended, and why (the measured numbers).
"""


class _MissingTracerHooks(logging.Filter):
    """MLflow's LangChain tracer predates LangGraph's on_interrupt/on_resume callbacks, and
    LangChain logs an error each time it can't call them. Everything is still traced."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("MlflowLangchainTracer" in msg and ("on_interrupt" in msg or "on_resume" in msg))


def setup_tracing(experiment: str | None = None) -> None:
    """Send LangGraph/LangChain traces to MLflow. Tracing problems must never break the agent."""
    s = get_settings()
    logging.getLogger("langchain_core.callbacks.manager").addFilter(_MissingTracerHooks())
    try:
        if s.mlflow_tracking_uri:
            mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        mlflow.set_experiment(experiment or s.mlflow_experiment)
        mlflow.langchain.autolog()
        # LangChain autolog already records every model call. OpenAI-level autolog would also add a
        # `traceparent` header, and when the model sits behind the MLflow AI Gateway the gateway then
        # writes into this trace and truncates it after the first LLM call.
        mlflow.openai.autolog(disable=True)
    except Exception as e:
        log.warning("MLflow tracing disabled: %s", e)


def make_llm() -> BaseChatModel:
    s = get_settings()
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        if not s.ollama_base_url:
            raise ValueError("LLM_PROVIDER=ollama needs OLLAMA_BASE_URL (your Ollama server) in .env")

        return ChatOllama(model=s.ollama_model, base_url=s.ollama_base_url, temperature=0, num_ctx=s.ollama_num_ctx)
    if s.llm_provider == "gateway":
        if not s.mlflow_tracking_uri:
            raise ValueError("LLM_PROVIDER=gateway needs MLFLOW_TRACKING_URI in .env")
        # The gateway speaks the OpenAI API; the endpoint name goes in `model`.
        return ChatOpenAI(
            model=s.mlflow_gateway_endpoint,
            reasoning_effort=s.llm_reasoning_effort,
            base_url=s.mlflow_tracking_uri.rstrip("/") + "/gateway/mlflow/v1",
            api_key="unused",  # auth to the model provider lives in the gateway
            temperature=0,
        )
    return ChatOpenAI(model=s.openai_model, reasoning_effort=s.llm_reasoning_effort)


def build_graph(
    backend: Backend,
    scope: ActionScope,
    node: str,
    llm: BaseChatModel | None = None,
):
    tools = make_tools(backend, scope)
    llm = llm or make_llm()
    # One tool call per step: if the restart interrupt shared a step with other calls, they
    # would all re-run on resume. OpenAI can be told directly; others are trimmed in `agent`.
    extra = {"parallel_tool_calls": False} if isinstance(llm, ChatOpenAI) else {}
    model = llm.bind_tools(tools, **extra)

    async def agent(state: MessagesState):
        prompt = SYSTEM_PROMPT.format(
            node=node,
            restartable=scope.describe(),
            now=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        )
        messages = [SystemMessage(prompt), *state["messages"]]
        response = await model.ainvoke(messages)
        if not response.tool_calls and not str(response.content).strip():
            # Small reasoning models (e.g. qwen via Ollama) sometimes end a turn with an empty answer
            # after their tool calls. Nudge once; the nudge isn't kept in the conversation.
            response = await model.ainvoke(
                [*messages, HumanMessage("Write your final answer now, using the answer format.")]
            )
        if not response.tool_calls and (narrated := _narrated_actions(str(response.content), state["messages"])):
            # Small models sometimes write "Calling restart_container…" as text instead of calling it.
            # Nudge once to either make the call or answer without claiming it.
            response = await model.ainvoke(
                [
                    *messages,
                    response,
                    HumanMessage(
                        f"You described calling {', '.join(sorted(narrated))} but didn't call it. Call the tool "
                        "now if you intend to, or rewrite your answer without claiming it."
                    ),
                ]
            )
        if response.response_metadata.get("finish_reason") == "length" and not response.tool_calls:
            # Out of room (often a small local model's context window). Say so instead of showing a
            # silently truncated answer.
            response.content = (
                f"{response.content}\n\n_(Answer cut off: the model ran out of context or output tokens.)_"
            )
        if len(response.tool_calls) > 1:
            response.tool_calls = response.tool_calls[:1]
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=MemorySaver())


ACTION_TOOLS = ("restart_container", "resize_container")


def _narrated_actions(text: str, messages: list) -> set[str]:
    """Action tools named in `text` that weren't actually called since the user's last message."""
    since_user = messages[max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0) :]
    called = {m.name for m in since_user if isinstance(m, ToolMessage)}
    return {t for t in ACTION_TOOLS if t in text and t not in called}


Decide = Callable[[dict[str, Any]], Awaitable[bool]]
# Sync (CLI printing) or async (Chainlit steps).
OnEvent = Callable[[str, dict[str, Any]], Awaitable[None] | None]


def _parse(content: Any) -> Any:
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return content


@mlflow.trace(name="agent_turn", span_type="AGENT")
async def run_turn(graph, message: str, thread_id: str, decide: Decide, on_event: OnEvent | None = None) -> dict:
    """Run one user turn to completion, pausing on each approval interrupt for `decide`.

    Used by both the CLI (decide = ask the human) and evals (decide = scripted policy), so the
    whole turn, including the pause and resume, lands in a single MLflow trace.
    """
    if mlflow.get_current_active_span():  # absent when MLflow has tracing switched off
        mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})
    config = {"configurable": {"thread_id": thread_id}}

    async def emit(kind: str, data: dict[str, Any]) -> None:
        if on_event and inspect.isawaitable(result := on_event(kind, data)):
            await result

    payload: Any = {"messages": [HumanMessage(message)]}
    tool_calls: dict[str, dict[str, Any]] = {}
    approvals: list[dict[str, Any]] = []

    while True:
        pending = None
        async for update in graph.astream(payload, config, stream_mode="updates"):
            for node, data in update.items():
                if node == "__interrupt__":
                    pending = data[0].value
                    continue
                for msg in (data or {}).get("messages", []):
                    if isinstance(msg, AIMessage):
                        for tc in msg.tool_calls:
                            tool_calls[tc["id"]] = {
                                "id": tc["id"],
                                "name": tc["name"],
                                "args": tc["args"],
                                "output": None,
                            }
                            await emit("tool_call", tool_calls[tc["id"]])
                    elif isinstance(msg, ToolMessage) and msg.tool_call_id in tool_calls:
                        tool_calls[msg.tool_call_id]["output"] = _parse(msg.content)
                        await emit("tool_result", tool_calls[msg.tool_call_id])
        if pending is None:
            break
        with mlflow.start_span(name="human_approval", span_type="TOOL") as span:
            span.set_inputs(pending)
            approved = await decide(pending)
            span.set_outputs({"approved": approved})
        approvals.append({"request": pending, "approved": approved})
        payload = Command(resume={"approved": approved})

    state = await graph.aget_state(config)
    return {
        "response": state.values["messages"][-1].content,
        "tool_calls": list(tool_calls.values()),
        "approvals": approvals,
    }
