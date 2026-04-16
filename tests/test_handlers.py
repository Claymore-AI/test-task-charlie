"""Tests for individual phase handlers."""

import pytest

from charlie_ai.handlers import (
    FarewellHandler,
    GreetingHandler,
    ReviewHandler,
    VocabularyHandler,
)
from charlie_ai.models import (
    ActivityType,
    CharlieMessage,
    Emotion,
    EvalResult,
    EvalStatus,
    LessonState,
    Phase,
    ResponderOutput,
    SubPhase,
    TurnResponse,
)
from tests.conftest import MockLLMClient


def _make_handler(cls, responses=None):
    llm = MockLLMClient(responses or [])
    from charlie_ai.agents import EvaluatorAgent, ResponderAgent, SafetyAgent
    responder = ResponderAgent(llm)  # type: ignore[arg-type]
    evaluator = EvaluatorAgent(llm)  # type: ignore[arg-type]
    safety = SafetyAgent(llm)  # type: ignore[arg-type]
    return cls(responder, evaluator, safety), llm


class TestGreetingHandler:
    @pytest.mark.asyncio
    async def test_first_turn_greeting(self):
        handler, _ = _make_handler(GreetingHandler)
        state = LessonState(words=["cat"])
        state.init_word_progress()

        turn, state = await handler.handle(state, "")
        assert isinstance(turn, TurnResponse)
        assert state.greeting_sent is True
        assert len(state.history) == 1

    @pytest.mark.asyncio
    async def test_second_turn_transitions(self):
        handler, _ = _make_handler(GreetingHandler)
        state = LessonState(words=["cat"], greeting_sent=True)
        state.init_word_progress()

        turn, state = await handler.handle(state, "I'm Alex")
        assert state.phase == Phase.VOCABULARY
        assert state.child_name == "Alex"

    @pytest.mark.asyncio
    async def test_name_extraction(self):
        handler, _ = _make_handler(GreetingHandler)
        state = LessonState(words=["cat"], greeting_sent=True)
        state.init_word_progress()

        await handler.handle(state, "my name is Sophie")
        assert state.child_name == "Sophie"

    @pytest.mark.asyncio
    async def test_silent_greeting_no_name(self):
        handler, _ = _make_handler(GreetingHandler)
        state = LessonState(words=["cat"], greeting_sent=True)
        state.init_word_progress()

        await handler.handle(state, "")
        assert state.child_name is None


class TestVocabularyHandler:
    @pytest.mark.asyncio
    async def test_introduce_phase(self):
        responses = [
            ResponderOutput(message="Cat is fluffy!", emotion=Emotion.PLAYFUL),
        ]
        handler, _ = _make_handler(VocabularyHandler, responses)
        state = LessonState(
            words=["cat", "dog"],
            phase=Phase.VOCABULARY,
            sub_phase=SubPhase.INTRODUCE,
        )
        state.init_word_progress()

        turn, state = await handler.handle(state, "")
        assert state.sub_phase == SubPhase.PRACTICE
        assert turn.highlight_word == "cat"
        assert turn.image_hint == "cat"

    @pytest.mark.asyncio
    async def test_correct_answer_advances(self):
        responses = [
            EvalResult(status=EvalStatus.CORRECT, confidence=0.95, reasoning="OK"),
            ResponderOutput(message="Yay!", emotion=Emotion.EXCITED),
        ]
        handler, _ = _make_handler(VocabularyHandler, responses)
        state = LessonState(
            words=["cat", "dog"],
            phase=Phase.VOCABULARY,
            sub_phase=SubPhase.PRACTICE,
        )
        state.init_word_progress()

        turn, state = await handler.handle(state, "cat")
        assert state.word_index == 1  # advanced to next word
        assert state.total_correct == 1
        assert state.streak == 1

    @pytest.mark.asyncio
    async def test_off_topic_tracked(self):
        responses = [
            EvalResult(status=EvalStatus.OFF_TOPIC, confidence=0.9, reasoning="Spiderman"),
            ResponderOutput(message="Cool! But can you say cat?", emotion=Emotion.PLAYFUL),
        ]
        handler, _ = _make_handler(VocabularyHandler, responses)
        state = LessonState(
            words=["cat", "dog"],
            phase=Phase.VOCABULARY,
            sub_phase=SubPhase.PRACTICE,
        )
        state.init_word_progress()

        turn, state = await handler.handle(state, "I like Spiderman")
        assert state.consecutive_off_topic == 1
        assert state.streak == 0

    @pytest.mark.asyncio
    async def test_silence_tracked(self):
        responses = [
            EvalResult(status=EvalStatus.SILENCE, confidence=1.0, reasoning="Silent"),
            ResponderOutput(message="That's okay!", emotion=Emotion.GENTLE),
        ]
        handler, _ = _make_handler(VocabularyHandler, responses)
        state = LessonState(
            words=["cat"],
            phase=Phase.VOCABULARY,
            sub_phase=SubPhase.PRACTICE,
        )
        state.init_word_progress()

        turn, state = await handler.handle(state, "")
        assert state.consecutive_silence == 1

    @pytest.mark.asyncio
    async def test_last_attempt_advances(self):
        responses = [
            EvalResult(status=EvalStatus.INCORRECT, confidence=0.9, reasoning="Wrong"),
            ResponderOutput(message="It's cat!", emotion=Emotion.GENTLE),
        ]
        handler, _ = _make_handler(VocabularyHandler, responses)
        state = LessonState(
            words=["cat", "dog"],
            phase=Phase.VOCABULARY,
            sub_phase=SubPhase.PRACTICE,
            attempt=2,  # will become 3 (max)
        )
        state.init_word_progress()

        turn, state = await handler.handle(state, "mouse")
        assert state.word_index == 1  # advanced despite wrong answer


class TestReviewHandler:
    @pytest.mark.asyncio
    async def test_review_transitions_to_farewell(self):
        responses = [
            ResponderOutput(message="Quiz time!", emotion=Emotion.PLAYFUL),
        ]
        handler, _ = _make_handler(ReviewHandler, responses)
        state = LessonState(words=["cat", "dog"], phase=Phase.REVIEW)
        state.init_word_progress()
        state.word_progress[0].is_mastered = True

        turn, state = await handler.handle(state, "")
        assert state.phase == Phase.FAREWELL
        assert turn.activity_type == ActivityType.RIDDLE


class TestFarewellHandler:
    @pytest.mark.asyncio
    async def test_farewell_ends_lesson(self):
        handler, _ = _make_handler(FarewellHandler)
        state = LessonState(words=["cat"], phase=Phase.FAREWELL)
        state.init_word_progress()

        turn, state = await handler.handle(state, "")
        assert state.phase == Phase.ENDED
        assert turn.emotion == Emotion.PROUD
