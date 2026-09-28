"""Run the eval suite with MLflow: `uv run python -m evals.run_evals [--no-judges] [--scenario NAME ...]`."""

import argparse
import asyncio
import sys
import textwrap
import threading
import uuid

import mlflow
from mlflow.entities import Feedback

from agent.config import model_label
from agent.demo import NODE
from agent.graph import build_graph, make_llm, run_turn, setup_tracing
from agent.tools import ActionScope
from evals.scenarios import SCENARIOS, SCENARIOS_BY_NAME
from evals.scorers import CODE_SCORERS, judge_scorers

# One long-lived event loop for every prediction. asyncio.run() per call would close its loop, and
# langchain-openai reuses its HTTP client across calls, so the next call fails with "Event loop is closed".
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()


def predict(question: str, scenario: str) -> dict:
    """Run the real agent graph against the scenario's scripted backend."""
    sc = SCENARIOS_BY_NAME[scenario]
    backend = sc.backend()
    graph = build_graph(backend, ActionScope(sc.scope), node=NODE)

    async def decide(request: dict) -> bool:
        return sc.approve

    turn = run_turn(graph, question, f"eval-{scenario}-{uuid.uuid4().hex[:8]}", decide)
    result = asyncio.run_coroutine_threadsafe(turn, _loop).result()
    return {
        **result,
        "restarts_executed": backend.restarts,
        "resizes_executed": backend.resizes,
        "permitted_vmids": sorted(sc.scope),
    }


def preflight() -> None:
    """One cheap model call first, so an unreachable or misconfigured model is one clear error
    instead of every scorer failing on every scenario."""
    from agent.config import get_settings

    try:
        asyncio.run_coroutine_threadsafe(make_llm().ainvoke("Reply with OK."), _loop).result(timeout=120)
    except Exception as e:
        sys.exit(f"Model check failed for {model_label(get_settings())}: {type(e).__name__}: {str(e)[:400]}")


def _failed(feedback: Feedback) -> bool:
    value = feedback.value
    return bool(feedback.error) or value is False or (isinstance(value, str) and value.lower() == "no")


def failures(run_id: str) -> dict[str, list[tuple[str, str, str]]]:
    """{scenario: [(scorer, "code" | "judge", reason)]} for every failed check in a run."""
    experiment_id = mlflow.get_run(run_id).info.experiment_id
    found: dict[str, list[tuple[str, str, str]]] = {}
    for trace in mlflow.search_traces(locations=[experiment_id], run_id=run_id, return_type="list"):
        # Each eval turn's session is "eval-<scenario>-<8 hex>" (set in predict).
        session = trace.info.request_metadata.get("mlflow.trace.session", "")
        scenario = session.removeprefix("eval-").rsplit("-", 1)[0] or trace.info.trace_id
        for a in trace.info.assessments:
            if isinstance(a, Feedback) and _failed(a):
                kind = "judge" if "LLM" in str(getattr(a.source, "source_type", "")) else "code"
                reason = a.rationale or (a.error.error_message if a.error else "") or "(no reason recorded)"
                found.setdefault(scenario, []).append((a.name, kind, reason))
    return found


def print_failures(run_id: str) -> None:
    found = failures(run_id)
    total = sum(len(v) for v in found.values())
    if not total:
        print("\nNo failed checks.")
        return
    print(f"\nFailed checks ({total} across {len(found)} scenarios)")
    for scenario in sorted(found):
        print(f"\n  {scenario}")
        for name, kind, reason in sorted(found[scenario]):
            print(f"    ✗ {name} ({kind})")
            short = textwrap.shorten(" ".join(reason.split()), width=360, placeholder=" …")
            print(textwrap.indent(textwrap.fill(short, width=92), " " * 8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-judges", action="store_true", help="code scorers only (fast, no judge LLM calls)")
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS_BY_NAME), help="run only these")
    parser.add_argument("--failures", metavar="RUN_ID", help="only print the failed checks of an earlier run")
    args = parser.parse_args()

    if args.failures:
        setup_tracing(experiment="datacenter-agent-evals")
        print_failures(args.failures)
        return

    preflight()
    setup_tracing(experiment="datacenter-agent-evals")
    selected = [s for s in SCENARIOS if not args.scenario or s.name in args.scenario]
    data = [{"inputs": {"question": s.question, "scenario": s.name}, "expectations": s.expectations} for s in selected]
    scorers = CODE_SCORERS + ([] if args.no_judges else judge_scorers())

    result = mlflow.genai.evaluate(data=data, predict_fn=predict, scorers=scorers)

    print("\nScorer summary")
    for name, value in sorted(result.metrics.items()):
        print(f"  {name:<40} {value:.2f}")
    print_failures(result.run_id)
    print(f"\nRun {result.run_id}: open the MLflow UI's Evaluations tab for per-scenario traces.")


if __name__ == "__main__":
    main()
