"""Content safety pipeline for a children's education context.

Three stages, applied in order::

    sanitize(input) → filter(input) → [lesson logic] → guard(output)

``InputSanitizer``  — normalises and cleans raw text from STT / keyboard.
``ContentFilter``   — rejects clearly inappropriate child input.
``OutputGuardrail`` — verifies Charlie's response is safe and in-character.

Design: rule-based for speed (children's apps must feel instant).  An LLM
fallback can be added later for nuanced cases, but the common path should
never wait on an extra LLM round-trip.
"""

from __future__ import annotations

import re
import unicodedata


class InputSanitizer:
    """Normalise raw text input before any processing.

    - Strips control characters (STT artefacts, copy-paste junk).
    - Collapses whitespace.
    - Truncates to a safe maximum length.
    """

    MAX_LENGTH = 200

    @classmethod
    def sanitize(cls, text: str) -> str:
        if not text:
            return ""
        # Remove control chars except newline / tab.
        cleaned = "".join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")
        )
        # Collapse whitespace.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Truncate.
        return cleaned[: cls.MAX_LENGTH]


class ContentFilter:
    """Rejects clearly inappropriate input for a 4-8 year-old context.

    Returns ``(is_safe, reason)`` — callers decide what to do with
    flagged content (typically: ignore the input and respond with a
    gentle redirect).
    """

    _PATTERNS: list[re.Pattern[str]] = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(?:kill|die|dead|murder|blood|gun|shoot|knife|weapon)\b",
            r"\b(?:sex|nude|naked|porn|xxx)\b",
            r"\b(?:fuck|shit|damn|ass|bitch|bastard|crap)\b",
            r"\b(?:drug|cocaine|heroin|weed|marijuana|meth)\b",
            r"\b(?:suicide|self[- ]?harm)\b",
        ]
    ]

    @classmethod
    def check(cls, text: str) -> tuple[bool, str | None]:
        """Return ``(True, None)`` if safe, ``(False, reason)`` if not."""
        if not text or not text.strip():
            return True, None
        for pattern in cls._PATTERNS:
            if pattern.search(text):
                return False, f"Matched inappropriate pattern"
        return True, None


class OutputGuardrail:
    """Verify Charlie's LLM-generated output before returning it.

    Checks:
    - Response length is reasonable (not empty, not a wall of text).
    - No inappropriate content leaked through.
    - Charlie didn't break character (no AI self-references).
    """

    MAX_RESPONSE_LENGTH = 500
    MIN_RESPONSE_LENGTH = 2

    _CHARACTER_BREAK_PATTERNS: list[re.Pattern[str]] = [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"\b(?:I'?m an? AI|I'?m a (?:chat)?bot|language model)\b",
            r"\b(?:as an AI|artificial intelligence)\b",
            r"\b(?:I don'?t have feelings|I'?m not real)\b",
        ]
    ]

    @classmethod
    def check(cls, text: str) -> tuple[bool, str | None]:
        """Return ``(True, None)`` if safe, ``(False, reason)`` if not."""
        if not text or len(text.strip()) < cls.MIN_RESPONSE_LENGTH:
            return False, "Response too short or empty"

        if len(text) > cls.MAX_RESPONSE_LENGTH:
            return False, "Response too long"

        # Check for inappropriate content in output.
        is_safe, reason = ContentFilter.check(text)
        if not is_safe:
            return False, f"Output content filter: {reason}"

        # Check for character breaks.
        for pattern in cls._CHARACTER_BREAK_PATTERNS:
            if pattern.search(text):
                return False, "Charlie broke character"

        return True, None

    @classmethod
    def safe_fallback(cls, emotion_context: str = "encouraging") -> str:
        """Return a safe generic Charlie response when the LLM output
        fails guardrail checks."""
        fallbacks = {
            "encouraging": "You're doing great! Let's keep going!",
            "gentle": "That's okay! Let's try together!",
            "excited": "Woohoo! Let's learn more words!",
        }
        return fallbacks.get(emotion_context, fallbacks["encouraging"])
