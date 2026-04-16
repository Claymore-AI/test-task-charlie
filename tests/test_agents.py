"""Tests for the multi-agent layer."""

import pytest

from charlie_ai.agents import EvaluatorAgent, ResponderAgent, SafetyAgent
from charlie_ai.models import (
    ActivityType,
    CharlieMessage,
    Emotion,
    EvalResult,
    EvalStatus,
    LessonState,
    ResponderOutput,
    SafetyVerdict,
)
from tests.conftest import MockLLMClient


class TestSafetyAgent:
    def test_safe_input(self):
        llm = MockLLMClient()
        agent = SafetyAgent(llm)  # type: ignore[arg-type]
        result = agent.check_input("hello teacher")
        assert result.is_safe

    def test_empty_input_is_safe(self):
        llm = MockLLMClient()
        agent = SafetyAgent(llm)  # type: ignore[arg-type]
        assert agent.check_input("").is_safe
        assert agent.check_input("  ").is_safe

    def test_inappropriate_input(self):
        llm = MockLLMClient()
        agent = SafetyAgent(llm)  # type: ignore[arg-type]
        result = agent.check_input("I want to kill it")
        assert not result.is_safe

    def test_safe_output(self):
        llm = MockLLMClient()
        agent = SafetyAgent(llm)  # type: ignore[arg-type]
        result = agent.check_output("Yay! You said cat! Amazing!")
        assert result.is_safe

    def test_output_character_break(self):
        llm = MockLLMClient()
        agent = SafetyAgent(llm)  # type: ignore[arg-type]
        result = agent.check_output("I'm an AI language model, I can help.")
        assert not result.is_safe


class TestEvaluatorAgent:
    @pytest.mark.asyncio
    async def test_evaluate_returns_eval_result(self):
        expected = EvalResult(
            status=EvalStatus.CORRECT,
            confidence=0.95,
            reasoning="Child said 'cat' which matches target.",
        )
        llm = MockLLMClient([expected])
        agent = EvaluatorAgent(llm)  # type: ignore[arg-type]

        state = LessonState(words=["cat"])
        state.init_word_progress()

        result = await agent.evaluate(
            child_input="cat",
            target_word="cat",
            activity_type=ActivityType.REPEAT,
            state=state,
        )
        assert result.status == EvalStatus.CORRECT
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_evaluate_passes_messages(self):
        llm = MockLLMClient()
        agent = EvaluatorAgent(llm)  # type: ignore[arg-type]
        state = LessonState(words=["dog"])
        state.init_word_progress()

        await agent.evaluate("dog", "dog", ActivityType.REPEAT, state)

        assert len(llm.call_log) == 1
        assert llm.call_log[0]["temperature"] == 0.2  # low temp for eval


class TestResponderAgent:
    @pytest.mark.asyncio
    async def test_respond(self):
        expected = ResponderOutput(
            message="Woohoo! You said dog!",
            emotion=Emotion.EXCITED,
            phonetic_hint=None,
        )
        llm = MockLLMClient([expected])
        agent = ResponderAgent(llm)  # type: ignore[arg-type]

        state = LessonState(words=["dog"])
        state.init_word_progress()
        eval_result = EvalResult(
            status=EvalStatus.CORRECT,
            confidence=0.95,
            reasoning="Correct",
        )

        result = await agent.respond(
            eval_result, "dog", ActivityType.REPEAT, state, False
        )
        assert result.message == "Woohoo! You said dog!"
        assert result.emotion == Emotion.EXCITED

    @pytest.mark.asyncio
    async def test_greet(self):
        llm = MockLLMClient()
        agent = ResponderAgent(llm)  # type: ignore[arg-type]
        state = LessonState(words=["cat"])

        result = await agent.greet(state)
        assert isinstance(result, CharlieMessage)
        assert llm.call_log[0]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_farewell(self):
        llm = MockLLMClient()
        agent = ResponderAgent(llm)  # type: ignore[arg-type]
        state = LessonState(words=["cat", "dog"])
        state.init_word_progress()

        result = await agent.farewell(state)
        assert isinstance(result, CharlieMessage)
