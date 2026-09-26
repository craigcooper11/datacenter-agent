"""Offline test fixtures: a scripted chat model, so the real graph runs with no API key or network."""

import os
import uuid
from typing import Any

import pytest

# Keep tests hermetic: ignore any local .env and don't write traces anywhere.
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/datacenter-agent-tests-mlruns"

import mlflow  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

mlflow.tracing.disable()


def call(name: str, **args: Any) -> AIMessage:
    """An AIMessage that calls one tool."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": uuid.uuid4().hex}])


class ScriptedLLM(BaseChatModel):
    """Replays `responses` in order, one per model call."""

    responses: list[AIMessage]
    _i: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> int:
        return self._i

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self._i >= len(self.responses):
            raise AssertionError(f"ScriptedLLM ran out of responses after {self._i} calls")
        msg = self.responses[self._i]
        self._i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def scripted():
    return ScriptedLLM
