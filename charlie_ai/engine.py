"""Lesson engine — the single entry-point for processing conversation turns.

The engine orchestrates a **multi-agent pipeline**::

    Child text
      → InputSanitizer
        → Safety check (input)
          → Phase handler (Evaluator + Responder agents)
            → Safety check (output)
              → TurnResponse  ──→  Frontend / TTS

It auto-advances through phases that need no user input (word introductions,
review, farewell) so that every ``process()`` call returns a complete,
displayable ``TurnResponse``.

``TurnResponse`` is a rich structured payload designed for consumption by
an Expo (React Native) mobile app — it carries Charlie's spoken text,
emotion tag for TTS prosody, pronunciation hints, image hints, exercise
type, and a progress snapshot for the UI progress bar.

The engine is fully stateless between calls: all session context lives in
``LessonState`` which can be serialised to Redis / PostgreSQL for a
production API deployment.
"""

from __future__ import annotations

import logging

from .agents import EvaluatorAgent, ResponderAgent, SafetyAgent
from .handlers import (
    FarewellHandler,
    GreetingHandler,
    PhaseHandler,
    ReviewHandler,
    VocabularyHandler,
)
from .llm_client import LLMClient
from .models import (
    Emotion,
    LessonProgress,
    LessonState,
    Phase,
    SubPhase,
    TurnResponse,
)
from .safety import InputSanitizer

logger = logging.getLogger(__name__)


class LessonEngine:
    """Orchestrates a single lesson session.

    Usage::

        engine = LessonEngine(words=["cat", "dog", "bird"])
        greeting = await engine.process("")      # Charlie greets
        reply    = await engine.process("hi!")    # child responds → lesson starts
        # reply is a TurnResponse with message, emotion, hints, progress...
    """

    def __init__(
        self,
        words: list[str] | None = None,
        *,
        llm: LLMClient | None = None,
    ) -> None:
        from .config import settings

        lesson_words = words if words is not None else list(settings.default_words)

        if not lesson_words:
            raise ValueError("At least one word is required for a lesson.")

        self._llm = llm or LLMClient()

        # Initialise agents.
        self._safety = SafetyAgent(self._llm)
        self._evaluator = EvaluatorAgent(self._llm)
        self._responder = ResponderAgent(self._llm)

        # Initialise state.
        self.state = LessonState(words=lesson_words)
        self.state.init_word_progress()

        # Phase → handler mapping (all share the same agent instances).
        self._handlers: dict[Phase, PhaseHandler] = {
            Phase.GREETING: GreetingHandler(self._responder, self._evaluator, self._safety),
            Phase.VOCABULARY: VocabularyHandler(self._responder, self._evaluator, self._safety),
            Phase.REVIEW: ReviewHandler(self._responder, self._evaluator, self._safety),
            Phase.FAREWELL: FarewellHandler(self._responder, self._evaluator, self._safety),
        }

    @property
    def is_finished(self) -> bool:
        return self.state.phase == Phase.ENDED

    async def process(self, user_input: str = "") -> TurnResponse:
        """Accept one user turn and return Charlie's complete ``TurnResponse``.

        Automatically chains handler calls for phases that require no
        user input.  For example, after a correct answer the response
        includes both the celebration **and** the next word introduction.
        """
        if self.is_finished:
            return TurnResponse(
                message="The lesson is already over. Start a new one to play again!",
                emotion=Emotion.GENTLE,
                progress=self.get_progress(),
            )

        # ── Sanitize input ───────────────────────────────────────────
        clean_input = InputSanitizer.sanitize(user_input)

        try:
            handler = self._handlers[self.state.phase]
            turn, self.state = await handler.handle(self.state, clean_input)

            # Auto-advance through non-interactive states.
            while self._needs_auto_advance():
                handler = self._handlers[self.state.phase]
                extra, self.state = await handler.handle(self.state, "")
                turn = TurnResponse(
                    message=f"{turn.message}\n\n{extra.message}",
                    emotion=extra.emotion,
                    highlight_word=extra.highlight_word or turn.highlight_word,
                    phonetic_hint=extra.phonetic_hint or turn.phonetic_hint,
                    image_hint=extra.image_hint or turn.image_hint,
                    expected_response=extra.expected_response,
                    activity_type=extra.activity_type or turn.activity_type,
                    progress=extra.progress or turn.progress,
                )

            return turn

        except Exception:
            logger.exception("Error during lesson processing")
            return TurnResponse(
                message="Oops! Charlie got a bit confused. Can you say that again?",
                emotion=Emotion.GENTLE,
                progress=self.get_progress(),
            )

    def get_progress(self) -> LessonProgress:
        """Return current lesson progress for the frontend."""
        return LessonProgress(
            words_completed=self.state.word_index,
            total_words=len(self.state.words),
            current_word=self.state.current_word,
            streak=self.state.streak,
            score=self.state.total_correct,
        )

    def get_state(self) -> LessonState:
        """Return the full serialisable lesson state.

        In a production API this would be persisted to Redis / DB between
        requests so the service stays stateless.
        """
        return self.state

    def _needs_auto_advance(self) -> bool:
        """Return True when the current state can proceed without input."""
        if (
            self.state.phase == Phase.VOCABULARY
            and self.state.sub_phase == SubPhase.INTRODUCE
        ):
            return True
        if self.state.phase == Phase.REVIEW:
            return True
        if self.state.phase == Phase.FAREWELL:
            return True
        return False
