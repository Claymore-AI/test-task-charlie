"""Tests for content safety — input sanitization, content filtering, output guardrails."""

from charlie_ai.safety import ContentFilter, InputSanitizer, OutputGuardrail


class TestInputSanitizer:
    def test_empty(self):
        assert InputSanitizer.sanitize("") == ""

    def test_strips_control_chars(self):
        result = InputSanitizer.sanitize("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result

    def test_collapses_whitespace(self):
        result = InputSanitizer.sanitize("hello    world   ")
        assert result == "hello world"

    def test_truncates_long_input(self):
        long_text = "a" * 500
        result = InputSanitizer.sanitize(long_text)
        assert len(result) <= InputSanitizer.MAX_LENGTH

    def test_preserves_normal_input(self):
        assert InputSanitizer.sanitize("cat") == "cat"
        assert InputSanitizer.sanitize("I like dogs!") == "I like dogs!"


class TestContentFilter:
    def test_safe_input(self):
        is_safe, reason = ContentFilter.check("hello teacher")
        assert is_safe
        assert reason is None

    def test_empty_is_safe(self):
        assert ContentFilter.check("")[0]
        assert ContentFilter.check("  ")[0]

    def test_violence_flagged(self):
        is_safe, reason = ContentFilter.check("I want to kill the cat")
        assert not is_safe
        assert reason is not None

    def test_profanity_flagged(self):
        is_safe, reason = ContentFilter.check("this is shit")
        assert not is_safe

    def test_normal_words_not_flagged(self):
        # "sun" should not match any pattern
        assert ContentFilter.check("the sun is bright")[0]
        assert ContentFilter.check("I like my cat")[0]
        assert ContentFilter.check("Spiderman is cool")[0]


class TestOutputGuardrail:
    def test_good_output(self):
        is_safe, _ = OutputGuardrail.check("Yay! You said cat! Amazing!")
        assert is_safe

    def test_empty_output(self):
        is_safe, reason = OutputGuardrail.check("")
        assert not is_safe
        assert "short" in reason.lower() or "empty" in reason.lower()

    def test_too_long_output(self):
        long_text = "word " * 200
        is_safe, reason = OutputGuardrail.check(long_text)
        assert not is_safe
        assert "long" in reason.lower()

    def test_character_break(self):
        is_safe, _ = OutputGuardrail.check("I'm an AI and I help children learn.")
        assert not is_safe

    def test_safe_fallback(self):
        fallback = OutputGuardrail.safe_fallback("encouraging")
        assert len(fallback) > 0
        assert "great" in fallback.lower() or "doing" in fallback.lower()

    def test_inappropriate_content_in_output(self):
        is_safe, _ = OutputGuardrail.check("Let's shoot the bad guys!")
        assert not is_safe
