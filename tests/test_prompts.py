"""Tests for prompt construction — history inclusion, schemas, engagement modifiers."""

from charlie_ai.models import (
    ActivityType,
    Emotion,
    EvalResult,
    EvalStatus,
    LessonState,
    Message,
)
from charlie_ai.prompts import (
    CHARLIE_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    build_conversation_messages,
    evaluator_messages,
    farewell_messages,
    get_phonetic_hint,
    greeting_messages,
    greeting_reply_messages,
    responder_messages,
    review_messages,
    word_intro_messages,
)


class TestBuildConversationMessages:
    def test_empty_history(self):
        assert build_conversation_messages([]) == []

    def test_role_mapping(self):
        history = [
            Message(role="charlie", text="Hi!"),
            Message(role="child", text="Hello!"),
        ]
        msgs = build_conversation_messages(history)
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "user"

    def test_window_limit(self):
        history = [Message(role="child", text=f"msg {i}") for i in range(20)]
        msgs = build_conversation_messages(history, max_turns=5)
        assert len(msgs) == 5

    def test_skips_empty_messages(self):
        history = [
            Message(role="charlie", text="Hi!"),
            Message(role="child", text="  "),
            Message(role="charlie", text="You there?"),
        ]
        msgs = build_conversation_messages(history)
        assert len(msgs) == 2  # empty child message skipped


class TestGreetingMessages:
    def test_has_system_and_user_roles(self):
        state = LessonState(words=["cat", "dog"])
        msgs = greeting_messages(state)
        assert msgs[0]["role"] == "system"
        assert CHARLIE_SYSTEM_PROMPT in msgs[0]["content"]
        assert any(m["role"] == "user" for m in msgs)

    def test_includes_words(self):
        state = LessonState(words=["cat", "dog"])
        msgs = greeting_messages(state)
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "cat" in user_msg["content"]
        assert "dog" in user_msg["content"]


class TestGreetingReplyMessages:
    def test_includes_history(self):
        state = LessonState(words=["cat"])
        state.history = [Message(role="charlie", text="Hi there!")]
        msgs = greeting_reply_messages("Hello!", state)
        # Should have system + history (assistant) + user
        assert len(msgs) >= 3

    def test_handles_silence(self):
        state = LessonState(words=["cat"])
        msgs = greeting_reply_messages("", state)
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "silence" in user_msg["content"].lower()


class TestWordIntroMessages:
    def test_includes_activity_instructions(self):
        state = LessonState(words=["cat"])
        msgs = word_intro_messages(
            "cat", 1, 3, ActivityType.RIDDLE, state
        )
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "RIDDLE" in user_msg["content"]

    def test_includes_phonetic_hint(self):
        state = LessonState(words=["cat"])
        msgs = word_intro_messages(
            "cat", 1, 3, ActivityType.REPEAT, state
        )
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "rhymes" in user_msg["content"].lower() or "kæt" in user_msg["content"]


class TestEvaluatorMessages:
    def test_has_evaluator_system_prompt(self):
        state = LessonState(words=["cat"])
        state.init_word_progress()
        msgs = evaluator_messages("cat", "cat", ActivityType.REPEAT, state)
        assert EVALUATOR_SYSTEM_PROMPT in msgs[0]["content"]

    def test_includes_target_word(self):
        state = LessonState(words=["dog"])
        state.init_word_progress()
        msgs = evaluator_messages("doggy", "dog", ActivityType.REPEAT, state)
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "dog" in user_msg["content"]


class TestResponderMessages:
    def test_engagement_modifier_silence(self):
        state = LessonState(words=["cat"], consecutive_silence=3)
        state.init_word_progress()
        eval_result = EvalResult(
            status=EvalStatus.SILENCE, confidence=1.0, reasoning="Silent"
        )
        msgs = responder_messages(
            eval_result, "cat", ActivityType.REPEAT, state, False
        )
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "silent" in user_msg["content"].lower() or "gentle" in user_msg["content"].lower()

    def test_last_attempt_note(self):
        state = LessonState(words=["cat"])
        state.init_word_progress()
        eval_result = EvalResult(
            status=EvalStatus.INCORRECT, confidence=0.9, reasoning="Wrong"
        )
        msgs = responder_messages(
            eval_result, "cat", ActivityType.REPEAT, state, is_last_attempt=True
        )
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "LAST attempt" in user_msg["content"]


class TestPhoneticHints:
    def test_known_word(self):
        assert get_phonetic_hint("cat") is not None
        assert "kæt" in get_phonetic_hint("cat")  # type: ignore

    def test_unknown_word(self):
        assert get_phonetic_hint("xylophone") is None

    def test_case_insensitive(self):
        assert get_phonetic_hint("CAT") == get_phonetic_hint("cat")


class TestReviewMessages:
    def test_includes_words(self):
        state = LessonState(words=["cat", "dog"])
        state.init_word_progress()
        msgs = review_messages(["cat", "dog"], state)
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "cat" in user_msg["content"]


class TestFarewellMessages:
    def test_includes_progress(self):
        state = LessonState(words=["cat", "dog"], total_correct=2, total_attempts=3)
        state.init_word_progress()
        msgs = farewell_messages(state)
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert "2" in user_msg["content"]  # score
