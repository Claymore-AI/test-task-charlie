"""Tests for data models — state, enums, TurnResponse, serialisation."""

import json

from charlie_ai.models import (
    ACTIVITY_DIFFICULTY,
    ActivityType,
    Emotion,
    EvalStatus,
    LessonProgress,
    LessonState,
    Phase,
    SubPhase,
    TurnResponse,
    WordProgress,
)


class TestLessonState:
    def test_init_word_progress(self):
        state = LessonState(words=["cat", "dog"])
        state.init_word_progress()
        assert len(state.word_progress) == 2
        assert state.word_progress[0].word == "cat"
        assert state.word_progress[1].word == "dog"

    def test_init_word_progress_idempotent(self):
        state = LessonState(words=["cat"])
        state.init_word_progress()
        state.word_progress[0].attempts = 5
        state.init_word_progress()  # should not overwrite
        assert state.word_progress[0].attempts == 5

    def test_current_word(self):
        state = LessonState(words=["cat", "dog", "bird"])
        assert state.current_word == "cat"
        state.word_index = 1
        assert state.current_word == "dog"
        state.word_index = 10
        assert state.current_word is None

    def test_current_word_progress(self):
        state = LessonState(words=["cat", "dog"])
        state.init_word_progress()
        assert state.current_word_progress is not None
        assert state.current_word_progress.word == "cat"

    def test_serialization_roundtrip(self):
        state = LessonState(
            words=["cat", "dog"],
            phase=Phase.VOCABULARY,
            child_name="Alex",
            streak=3,
        )
        state.init_word_progress()
        data = state.model_dump()
        json_str = json.dumps(data)
        restored = LessonState.model_validate(json.loads(json_str))
        assert restored.words == ["cat", "dog"]
        assert restored.child_name == "Alex"
        assert restored.streak == 3
        assert len(restored.word_progress) == 2

    def test_default_values(self):
        state = LessonState()
        assert state.phase == Phase.GREETING
        assert state.streak == 0
        assert state.consecutive_silence == 0
        assert state.child_name is None


class TestWordProgress:
    def test_default(self):
        wp = WordProgress(word="cat")
        assert wp.attempts == 0
        assert wp.is_mastered is False
        assert wp.activities_completed == []

    def test_tracking(self):
        wp = WordProgress(word="cat")
        wp.attempts += 1
        wp.activities_completed.append(ActivityType.REPEAT)
        wp.is_mastered = True
        assert wp.attempts == 1
        assert wp.is_mastered
        assert ActivityType.REPEAT in wp.activities_completed


class TestTurnResponse:
    def test_minimal(self):
        tr = TurnResponse(message="Hello!")
        assert tr.message == "Hello!"
        assert tr.emotion == Emotion.NEUTRAL
        assert tr.highlight_word is None

    def test_full(self):
        tr = TurnResponse(
            message="Yay! Cat!",
            emotion=Emotion.EXCITED,
            highlight_word="cat",
            phonetic_hint="/kæt/",
            image_hint="cat",
            expected_response="cat",
            activity_type=ActivityType.REPEAT,
            progress=LessonProgress(
                words_completed=1,
                total_words=3,
                current_word="dog",
                streak=2,
                score=1,
            ),
        )
        assert tr.emotion == Emotion.EXCITED
        assert tr.progress is not None
        assert tr.progress.words_completed == 1

    def test_json_serialization(self):
        tr = TurnResponse(
            message="Hi!",
            emotion=Emotion.PLAYFUL,
            activity_type=ActivityType.RIDDLE,
        )
        data = tr.model_dump()
        assert data["emotion"] == "playful"
        assert data["activity_type"] == "riddle"


class TestEnums:
    def test_activity_difficulty_ordering(self):
        assert ACTIVITY_DIFFICULTY[ActivityType.REPEAT] < ACTIVITY_DIFFICULTY[ActivityType.RIDDLE]
        assert ACTIVITY_DIFFICULTY[ActivityType.RIDDLE] < ACTIVITY_DIFFICULTY[ActivityType.USE_IN_SENTENCE]

    def test_all_phases(self):
        phases = list(Phase)
        assert Phase.GREETING in phases
        assert Phase.REVIEW in phases
        assert Phase.ENDED in phases

    def test_all_emotions(self):
        emotions = list(Emotion)
        assert len(emotions) == 6
        assert Emotion.EXCITED in emotions
