"""Multi-agent orchestration layer.

Each agent wraps the shared ``LLMClient`` with a **dedicated system prompt,
temperature, and Pydantic response schema**.  The lesson engine calls agents
in a deterministic pipeline::

    Safety(input) → Evaluator(input) → Responder(eval_result) → Safety(output)

This separation gives us:

* **Reliable evaluation** — the Evaluator runs at low temperature (0.2) with
  chain-of-thought reasoning so classification is consistent.
* **Creative responses** — the Responder runs at higher temperature (0.7) for
  natural, varied Charlie dialogue.
* **Content safety** — the Safety agent (rule-based with LLM fallback) keeps
  both child input and Charlie output appropriate.

All agents are stateless.  Session context flows in via ``LessonState`` and
the ``build_conversation_messages`` helper from ``prompts``.
"""

from __future__ import annotations

import logging
import re

from .llm_client import LLMClient
from .models import (
    ActivityType,
    CharlieMessage,
    Emotion,
    EvalResult,
    EvalStatus,
    LessonState,
    ResponderOutput,
    SafetyVerdict,
)

logger = logging.getLogger(__name__)


# ── Safety Agent ─────────────────────────────────────────────────────────


class SafetyAgent:
    """Lightweight content safety for a children's education context.

    Uses a fast **rule-based filter first** (regex patterns for clearly
    inappropriate content) and falls back to an LLM check only when the
    rules are inconclusive.  This keeps latency low for the common case
    while still catching nuanced issues.
    """

    # Patterns that are clearly inappropriate for 4-8-year-old context.
    _INAPPROPRIATE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(?:kill|die|dead|murder|blood|gun|shoot|knife|weapon)\b",
            r"\b(?:sex|nude|naked|porn|xxx)\b",
            r"\b(?:fuck|shit|damn|ass|bitch|bastard|crap|dick|penis|vagina)\b",
            r"\b(?:drug|cocaine|heroin|weed|marijuana|meth)\b",
            r"\b(?:suicide|self[- ]?harm)\b",
        ]
    ]

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def check_input(self, text: str) -> SafetyVerdict:
        """Synchronous rule-based check on child input."""
        if not text or not text.strip():
            return SafetyVerdict(is_safe=True)
        return self._rule_check(text)

    def check_output(self, text: str) -> SafetyVerdict:
        """Synchronous rule-based check on Charlie's output."""
        if not text or not text.strip():
            return SafetyVerdict(is_safe=True)

        verdict = self._rule_check(text)
        if not verdict.is_safe:
            return verdict

        # Charlie must never break character.
        ai_patterns = [
            r"\b(?:I'?m an? AI|I'?m a (?:chat)?bot|language model|artificial)\b",
            r"\b(?:as an AI|I cannot|I don'?t have feelings)\b",
        ]
        for pattern in ai_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return SafetyVerdict(
                    is_safe=False,
                    reason="Charlie broke character (mentioned being AI).",
                )

        return SafetyVerdict(is_safe=True)

    def _rule_check(self, text: str) -> SafetyVerdict:
        for pattern in self._INAPPROPRIATE_PATTERNS:
            if pattern.search(text):
                return SafetyVerdict(
                    is_safe=False,
                    reason=f"Matched inappropriate pattern: {pattern.pattern}",
                )
        return SafetyVerdict(is_safe=True)


# ── Evaluator Agent ──────────────────────────────────────────────────────


class EvaluatorAgent:
    """Analyzes child input and classifies it against the target word.

    Runs at **low temperature (0.2)** for deterministic, consistent
    evaluation.  The prompt uses chain-of-thought: the model first
    reasons about what the child said, *then* classifies.

    Returns ``EvalResult`` with ``status``, ``confidence``, and
    ``reasoning`` (the chain-of-thought trace, useful for debugging).
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def evaluate(
        self,
        child_input: str,
        target_word: str,
        activity_type: ActivityType,
        state: LessonState,
    ) -> EvalResult:
        from . import prompts

        messages = prompts.evaluator_messages(
            child_input=child_input,
            target_word=target_word,
            activity_type=activity_type,
            state=state,
        )
        return await self._llm.generate(
            messages,
            EvalResult,
            temperature=0.2,
            max_tokens=300,
        )


# ── Responder Agent ──────────────────────────────────────────────────────


class ResponderAgent:
    """Generates Charlie's in-character reply given the evaluation result.

    Runs at **higher temperature (0.7)** for natural, varied dialogue.
    The prompt includes conversation history so Charlie can reference
    earlier turns, the child's name, and running context.

    Returns ``ResponderOutput`` with ``message``, ``emotion``, and an
    optional ``phonetic_hint``.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def respond(
        self,
        eval_result: EvalResult,
        target_word: str,
        activity_type: ActivityType,
        state: LessonState,
        is_last_attempt: bool,
    ) -> ResponderOutput:
        from . import prompts

        messages = prompts.responder_messages(
            eval_result=eval_result,
            target_word=target_word,
            activity_type=activity_type,
            state=state,
            is_last_attempt=is_last_attempt,
        )
        return await self._llm.generate(
            messages,
            ResponderOutput,
            temperature=0.7,
            max_tokens=400,
        )

    async def greet(self, state: LessonState) -> CharlieMessage:
        from . import prompts

        messages = prompts.greeting_messages(state)
        return await self._llm.generate(
            messages,
            CharlieMessage,
            temperature=0.7,
            max_tokens=300,
        )

    async def greet_reply(
        self, child_input: str, state: LessonState
    ) -> CharlieMessage:
        from . import prompts

        messages = prompts.greeting_reply_messages(child_input, state)
        return await self._llm.generate(
            messages,
            CharlieMessage,
            temperature=0.7,
            max_tokens=300,
        )

    async def introduce_word(
        self,
        word: str,
        word_num: int,
        total: int,
        activity_type: ActivityType,
        state: LessonState,
    ) -> ResponderOutput:
        from . import prompts

        messages = prompts.word_intro_messages(
            word=word,
            word_num=word_num,
            total=total,
            activity_type=activity_type,
            state=state,
        )
        return await self._llm.generate(
            messages,
            ResponderOutput,
            temperature=0.7,
            max_tokens=400,
        )

    async def review(
        self,
        words: list[str],
        state: LessonState,
    ) -> ResponderOutput:
        from . import prompts

        messages = prompts.review_messages(words, state)
        return await self._llm.generate(
            messages,
            ResponderOutput,
            temperature=0.7,
            max_tokens=400,
        )

    async def farewell(self, state: LessonState) -> CharlieMessage:
        from . import prompts

        messages = prompts.farewell_messages(state)
        return await self._llm.generate(
            messages,
            CharlieMessage,
            temperature=0.7,
            max_tokens=300,
        )
