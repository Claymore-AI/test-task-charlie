"""Phase handlers — one class per lesson phase.

Each handler receives the current ``LessonState`` and the child's text input,
then returns ``(TurnResponse, updated_state)``.

Handlers are intentionally **stateless**: all mutable context lives in
``LessonState``.  This makes sessions trivially serialisable and testable.

The ``VocabularyHandler`` demonstrates **multi-agent orchestration**:
it calls the Evaluator agent (low-temperature analysis) and Responder
agent (high-temperature creative reply) in sequence, with the Safety
agent gating both input and output.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from .activities import ActivitySelector
from .agents import EvaluatorAgent, ResponderAgent, SafetyAgent
from .config import settings
from .models import (
    ActivityType,
    Emotion,
    EvalResult,
    EvalStatus,
    LessonProgress,
    LessonState,
    Message,
    Phase,
    SubPhase,
    TurnResponse,
)
from .prompts import get_phonetic_hint
from .safety import OutputGuardrail

logger = logging.getLogger(__name__)


def _build_progress(state: LessonState) -> LessonProgress:
    """Snapshot current lesson progress for the frontend progress bar."""
    return LessonProgress(
        words_completed=state.word_index,
        total_words=len(state.words),
        current_word=state.current_word,
        streak=state.streak,
        score=state.total_correct,
    )


class PhaseHandler(ABC):
    """Abstract base for lesson-phase handlers."""

    def __init__(
        self,
        responder: ResponderAgent,
        evaluator: EvaluatorAgent,
        safety: SafetyAgent,
    ) -> None:
        self.responder = responder
        self.evaluator = evaluator
        self.safety = safety

    @abstractmethod
    async def handle(
        self, state: LessonState, user_input: str
    ) -> tuple[TurnResponse, LessonState]:
        """Process *user_input* and return ``(TurnResponse, new_state)``."""


# Greeting


class GreetingHandler(PhaseHandler):
    """Manages the opening greeting exchange.

    Turn 1 (``greeting_sent=False``): Charlie introduces himself and asks
    the child's name.

    Turn 2 (``greeting_sent=True``): Charlie acknowledges the child's
    response, extracts their name if given, and introduces the first word.
    """

    async def handle(
        self, state: LessonState, user_input: str
    ) -> tuple[TurnResponse, LessonState]:
        if not state.greeting_sent:
            resp = await self.responder.greet(state)
            state.greeting_sent = True
            state.history.append(Message(role="charlie", text=resp.message))
            return TurnResponse(
                message=resp.message,
                emotion=resp.emotion,
                progress=_build_progress(state),
            ), state

        # Child responded (or was silent) — extract name if present.
        state.history.append(Message(role="child", text=user_input))
        self._try_extract_name(user_input, state)

        resp = await self.responder.greet_reply(user_input, state)
        state.phase = Phase.VOCABULARY
        state.sub_phase = SubPhase.PRACTICE  # word introduced in reply
        state.attempt = 0

        first_word = state.words[0] if state.words else None
        state.history.append(Message(role="charlie", text=resp.message))

        return TurnResponse(
            message=resp.message,
            emotion=resp.emotion,
            highlight_word=first_word,
            phonetic_hint=get_phonetic_hint(first_word) if first_word else None,
            image_hint=first_word,
            expected_response=first_word,
            activity_type=state.current_activity,
            progress=_build_progress(state),
        ), state

    @staticmethod
    def _try_extract_name(text: str, state: LessonState) -> None:
        """Best-effort name extraction from the child's greeting."""
        if not text or not text.strip():
            return
        text_clean = text.strip()

        patterns = [
            r"(?:my name is|i'?m|i am|they call me|call me)\s+(\w+)",
            r"^(\w+)$",  # single word → likely a name
        ]
        for pattern in patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                name = match.group(1).capitalize()
                # Avoid capturing lesson words or greetings as names.
                skip = {"hello", "hi", "hey", "yes", "no", "cat", "dog",
                        "bird", "fish", "sun", "ok", "okay"}
                if name.lower() not in skip:
                    state.child_name = name
                    return


# Vocabulary


class VocabularyHandler(PhaseHandler):
    """Orchestrates word introduction and practice cycles.

    **Multi-agent pipeline per practice turn**::

        SafetyAgent.check_input(child_text)
          → EvaluatorAgent.evaluate(child_text, target_word)
            → ResponderAgent.respond(eval_result, context)
              → SafetyAgent.check_output(charlie_text)

    Sub-phases:
      ``INTRODUCE`` — Charlie presents a new word (no user input needed).
      ``PRACTICE``  — Child attempts the word; agents evaluate + respond.
    """

    async def handle(
        self, state: LessonState, user_input: str
    ) -> tuple[TurnResponse, LessonState]:
        if state.sub_phase == SubPhase.INTRODUCE:
            return await self._introduce(state)
        return await self._practice(state, user_input)

    async def _introduce(
        self, state: LessonState
    ) -> tuple[TurnResponse, LessonState]:
        word = state.words[state.word_index]

        # Select activity for this word.
        activity = ActivitySelector.select(state)
        state.current_activity = activity

        resp = await self.responder.introduce_word(
            word=word,
            word_num=state.word_index + 1,
            total=len(state.words),
            activity_type=activity,
            state=state,
        )

        state.sub_phase = SubPhase.PRACTICE
        state.attempt = 0
        state.history.append(Message(role="charlie", text=resp.message))

        return TurnResponse(
            message=resp.message,
            emotion=resp.emotion,
            highlight_word=word,
            phonetic_hint=resp.phonetic_hint or get_phonetic_hint(word),
            image_hint=word,
            expected_response=word,
            activity_type=activity,
            progress=_build_progress(state),
        ), state

    async def _practice(
        self, state: LessonState, user_input: str
    ) -> tuple[TurnResponse, LessonState]:
        state.history.append(Message(role="child", text=user_input))
        word = state.words[state.word_index]
        state.attempt += 1
        state.total_attempts += 1

        # Update word progress.
        wp = state.current_word_progress
        if wp:
            wp.attempts += 1

        # Safety check on input
        input_verdict = self.safety.check_input(user_input)
        if not input_verdict.is_safe:
            logger.warning("Input flagged: %s", input_verdict.reason)
            # Treat unsafe input as off-topic — redirect gently.
            eval_result = EvalResult(
                status=EvalStatus.OFF_TOPIC,
                confidence=1.0,
                reasoning="Input flagged by safety filter.",
            )
        else:
            # Evaluator agent
            eval_result = await self.evaluator.evaluate(
                child_input=user_input,
                target_word=word,
                activity_type=state.current_activity,
                state=state,
            )
        logger.debug(
            "Eval: word=%s status=%s conf=%.2f reason=%s",
            word, eval_result.status, eval_result.confidence,
            eval_result.reasoning,
        )

        if wp:
            wp.last_eval = eval_result.status

        # Update engagement signals
        self._update_engagement(state, eval_result.status)

        is_last = state.attempt >= settings.max_attempts_per_word

        # Responder agent
        resp = await self.responder.respond(
            eval_result=eval_result,
            target_word=word,
            activity_type=state.current_activity,
            state=state,
            is_last_attempt=is_last,
        )

        # Safety check on output
        output_ok, output_reason = OutputGuardrail.check(resp.message)
        if not output_ok:
            logger.warning("Output guardrail: %s", output_reason)
            resp.message = OutputGuardrail.safe_fallback()

        state.history.append(Message(role="charlie", text=resp.message))

        # Advance logic
        if eval_result.status == EvalStatus.CORRECT:
            if wp:
                wp.is_mastered = True
                wp.activities_completed.append(state.current_activity)
            self._advance_word(state)
        elif is_last:
            if wp:
                wp.activities_completed.append(state.current_activity)
            self._advance_word(state)
        elif eval_result.status in (EvalStatus.INCORRECT, EvalStatus.PARTIAL):
            # Scaffold down to an easier activity on failure.
            state.current_activity = ActivitySelector.scaffold_down(
                state.current_activity
            )

        return TurnResponse(
            message=resp.message,
            emotion=resp.emotion,
            highlight_word=word,
            phonetic_hint=resp.phonetic_hint or get_phonetic_hint(word),
            image_hint=word,
            expected_response=word if state.phase == Phase.VOCABULARY else None,
            activity_type=state.current_activity,
            progress=_build_progress(state),
        ), state

    @staticmethod
    def _update_engagement(state: LessonState, status: EvalStatus) -> None:
        """Update streak and engagement counters based on evaluation."""
        if status == EvalStatus.CORRECT:
            state.streak += 1
            state.total_correct += 1
            state.consecutive_silence = 0
            state.consecutive_off_topic = 0
        elif status == EvalStatus.SILENCE:
            state.consecutive_silence += 1
            state.streak = 0
        elif status == EvalStatus.OFF_TOPIC:
            state.consecutive_off_topic += 1
            state.streak = 0
        else:
            state.consecutive_silence = 0
            state.consecutive_off_topic = 0
            state.streak = 0

    @staticmethod
    def _advance_word(state: LessonState) -> None:
        """Move to the next word, or to REVIEW if all words are done."""
        state.word_index += 1
        state.attempt = 0
        if state.word_index >= len(state.words):
            state.phase = Phase.REVIEW
        else:
            state.sub_phase = SubPhase.INTRODUCE


# Review

class ReviewHandler(PhaseHandler):
    """Quick recap game — Charlie asks riddle-style questions about
    learned words to reinforce memory."""

    async def handle(
        self, state: LessonState, user_input: str
    ) -> tuple[TurnResponse, LessonState]:
        # Pick mastered words for review (or all words if none mastered).
        mastered = [wp.word for wp in state.word_progress if wp.is_mastered]
        review_words = mastered[:3] if mastered else state.words[:3]

        resp = await self.responder.review(review_words, state)

        state.phase = Phase.FAREWELL
        state.history.append(Message(role="charlie", text=resp.message))

        return TurnResponse(
            message=resp.message,
            emotion=resp.emotion,
            activity_type=ActivityType.RIDDLE,
            progress=_build_progress(state),
        ), state


# Farewell

class FarewellHandler(PhaseHandler):
    """Wraps up the lesson with a warm, personalized goodbye."""

    async def handle(
        self, state: LessonState, user_input: str
    ) -> tuple[TurnResponse, LessonState]:
        resp = await self.responder.farewell(state)
        state.phase = Phase.ENDED
        state.history.append(Message(role="charlie", text=resp.message))

        return TurnResponse(
            message=resp.message,
            emotion=Emotion.PROUD,
            progress=_build_progress(state),
        ), state
