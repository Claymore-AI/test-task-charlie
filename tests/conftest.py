"""Shared fixtures for the Charlie AI test suite.

``MockLLMClient`` returns **predefined responses** so tests are
deterministic, fast, and require no API key.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from charlie_ai.agents import EvaluatorAgent, ResponderAgent, SafetyAgent
from charlie_ai.llm_client import LLMClient
from charlie_ai.models import (
    ActivityType,
    CharlieMessage,
    Emotion,
    EvalResult,
    EvalStatus,
    LessonState,
    ResponderOutput,
)


class MockLLMClient:
    """Drop-in replacement for ``LLMClient`` that returns canned responses.

    Accepts a queue of response objects.  Each call to ``generate()``
    pops the next response from the queue.  If the queue is empty it
    returns a sensible default based on ``response_model``.
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._queue: list[Any] = list(responses or [])
        self.call_log: list[dict] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        response_model: type,
        *,
        temperature: float = 0.7,
        max_tokens: int = 400,
        max_retries: int = 2,
    ) -> Any:
        self.call_log.append({
            "messages": messages,
            "model": response_model,
            "temperature": temperature,
        })
        if self._queue:
            return self._queue.pop(0)
        return self._default(response_model)

    @staticmethod
    def _default(model: type) -> Any:
        if model is CharlieMessage:
            return CharlieMessage(message="Hi there!", emotion=Emotion.EXCITED)
        if model is EvalResult:
            return EvalResult(
                status=EvalStatus.CORRECT,
                confidence=0.95,
                reasoning="Child said the word correctly.",
            )
        if model is ResponderOutput:
            return ResponderOutput(
                message="Yay! Great job!",
                emotion=Emotion.EXCITED,
                phonetic_hint=None,
            )
        raise ValueError(f"No default for {model}")


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Provide a fresh ``MockLLMClient`` with default responses."""
    return MockLLMClient()


@pytest.fixture
def lesson_state() -> LessonState:
    """Provide a fresh ``LessonState`` with 3 words."""
    state = LessonState(words=["cat", "dog", "bird"])
    state.init_word_progress()
    return state


@pytest.fixture
def agents(mock_llm: MockLLMClient) -> tuple[ResponderAgent, EvaluatorAgent, SafetyAgent]:
    """Provide agents wired to a ``MockLLMClient``."""
    # Type: ignore because MockLLMClient duck-types LLMClient.
    responder = ResponderAgent(mock_llm)  # type: ignore[arg-type]
    evaluator = EvaluatorAgent(mock_llm)  # type: ignore[arg-type]
    safety = SafetyAgent(mock_llm)  # type: ignore[arg-type]
    return responder, evaluator, safety
