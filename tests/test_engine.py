"""Tests for the lesson engine — full lesson flow scenarios."""

import pytest

from charlie_ai.engine import LessonEngine
from charlie_ai.models import (
    CharlieMessage,
    Emotion,
    EvalResult,
    EvalStatus,
    Phase,
    ResponderOutput,
    TurnResponse,
)
from tests.conftest import MockLLMClient


def _make_engine(responses: list | None = None) -> LessonEngine:
    """Create a LessonEngine with a MockLLMClient."""
    llm = MockLLMClient(responses or [])
    # Bypass the real LLMClient constructor (needs API key).
    engine = LessonEngine.__new__(LessonEngine)
    engine._llm = llm
    engine.state = __import__("charlie_ai.models", fromlist=["LessonState"]).LessonState(
        words=["cat", "dog"]
    )
    engine.state.init_word_progress()

    from charlie_ai.agents import EvaluatorAgent, ResponderAgent, SafetyAgent
    from charlie_ai.handlers import (
        FarewellHandler, GreetingHandler, ReviewHandler, VocabularyHandler,
    )
    from charlie_ai.models import Phase

    engine._safety = SafetyAgent(llm)  # type: ignore[arg-type]
    engine._evaluator = EvaluatorAgent(llm)  # type: ignore[arg-type]
    engine._responder = ResponderAgent(llm)  # type: ignore[arg-type]
    engine._handlers = {
        Phase.GREETING: GreetingHandler(engine._responder, engine._evaluator, engine._safety),
        Phase.VOCABULARY: VocabularyHandler(engine._responder, engine._evaluator, engine._safety),
        Phase.REVIEW: ReviewHandler(engine._responder, engine._evaluator, engine._safety),
        Phase.FAREWELL: FarewellHandler(engine._responder, engine._evaluator, engine._safety),
    }
    return engine


class TestLessonEngine:
    @pytest.mark.asyncio
    async def test_greeting_flow(self):
        engine = _make_engine()
        result = await engine.process("")
        assert isinstance(result, TurnResponse)
        assert result.message  # Charlie said something
        assert engine.state.greeting_sent is True

    @pytest.mark.asyncio
    async def test_process_returns_turn_response(self):
        engine = _make_engine()
        result = await engine.process("")
        assert isinstance(result, TurnResponse)
        assert result.emotion is not None

    @pytest.mark.asyncio
    async def test_is_finished(self):
        engine = _make_engine()
        assert not engine.is_finished
        engine.state.phase = Phase.ENDED
        assert engine.is_finished

    @pytest.mark.asyncio
    async def test_process_after_finished(self):
        engine = _make_engine()
        engine.state.phase = Phase.ENDED
        result = await engine.process("hello")
        assert "already over" in result.message.lower()

    @pytest.mark.asyncio
    async def test_get_progress(self):
        engine = _make_engine()
        progress = engine.get_progress()
        assert progress.total_words == 2
        assert progress.words_completed == 0

    @pytest.mark.asyncio
    async def test_happy_path_two_words(self):
        """Simulate a full lesson where the child answers correctly."""
        responses = [
            # Greeting
            CharlieMessage(message="Hi! I'm Charlie!", emotion=Emotion.EXCITED),
            # Greeting reply (introduces first word)
            CharlieMessage(message="Cool! Our first word is cat!", emotion=Emotion.PLAYFUL),
            # Eval for "cat" → correct
            EvalResult(status=EvalStatus.CORRECT, confidence=0.95, reasoning="Correct"),
            # Responder for "cat" correct
            ResponderOutput(message="Yay! Cat!", emotion=Emotion.EXCITED),
            # Introduce "dog" (auto-advance)
            ResponderOutput(message="Next word: dog!", emotion=Emotion.PLAYFUL),
            # Eval for "dog" → correct
            EvalResult(status=EvalStatus.CORRECT, confidence=0.95, reasoning="Correct"),
            # Responder for "dog" correct
            ResponderOutput(message="Woohoo! Dog!", emotion=Emotion.EXCITED),
            # Review (auto-advance)
            ResponderOutput(message="Quick quiz time!", emotion=Emotion.PLAYFUL),
            # Farewell (auto-advance)
            CharlieMessage(message="Bye! You did great!", emotion=Emotion.PROUD),
        ]
        engine = _make_engine(responses)

        # Greeting
        r1 = await engine.process("")
        assert "Charlie" in r1.message

        # Child responds to greeting
        r2 = await engine.process("hi, I'm Alex")
        assert r2.message  # Charlie acknowledges + introduces first word

        # Child says "cat"
        r3 = await engine.process("cat")
        assert engine.state.total_correct >= 1

        # Child says "dog" → triggers review + farewell auto-advance
        r4 = await engine.process("dog")
        assert engine.is_finished

    @pytest.mark.asyncio
    async def test_silent_child(self):
        """Child stays silent — engine should handle gracefully."""
        engine = _make_engine()

        # Greeting
        await engine.process("")
        # Child is silent for greeting reply
        await engine.process("")
        assert engine.state.phase == Phase.VOCABULARY

    @pytest.mark.asyncio
    async def test_get_state_is_serializable(self):
        engine = _make_engine()
        state = engine.get_state()
        data = state.model_dump()
        assert "words" in data
        assert "phase" in data
