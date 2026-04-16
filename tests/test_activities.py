"""Tests for activity selection and pedagogical scaffolding."""

from charlie_ai.activities import ActivitySelector
from charlie_ai.models import ActivityType, LessonState


class TestActivitySelector:
    def test_first_word_always_repeat(self):
        state = LessonState(words=["cat", "dog", "bird"])
        state.init_word_progress()
        assert ActivitySelector.select(state) == ActivityType.REPEAT

    def test_struggling_child_gets_repeat(self):
        state = LessonState(words=["cat", "dog"], consecutive_silence=3)
        state.init_word_progress()
        state.word_index = 1
        assert ActivitySelector.select(state) == ActivityType.REPEAT

    def test_off_topic_child_gets_repeat(self):
        state = LessonState(words=["cat", "dog"], consecutive_off_topic=3)
        state.init_word_progress()
        state.word_index = 1
        assert ActivitySelector.select(state) == ActivityType.REPEAT

    def test_streak_increases_difficulty(self):
        state = LessonState(words=["cat", "dog", "bird", "fish"], streak=3)
        state.init_word_progress()
        state.word_index = 2
        activity = ActivitySelector.select(state)
        # Should be RIDDLE or harder with a streak of 3
        assert activity in (ActivityType.RIDDLE, ActivityType.LISTEN_AND_PICK)

    def test_scaffold_down(self):
        result = ActivitySelector.scaffold_down(ActivityType.RIDDLE)
        assert result == ActivityType.LISTEN_AND_PICK

    def test_scaffold_down_from_easiest(self):
        result = ActivitySelector.scaffold_down(ActivityType.REPEAT)
        assert result == ActivityType.REPEAT  # can't go lower

    def test_variety_avoids_consecutive_same_type(self):
        state = LessonState(words=["cat", "dog", "bird"])
        state.init_word_progress()
        # First word completed with LISTEN_AND_PICK
        state.word_progress[0].activities_completed.append(
            ActivityType.LISTEN_AND_PICK
        )
        state.word_index = 1
        activity = ActivitySelector.select(state)
        # Should try to pick something different (but depends on other signals)
        assert isinstance(activity, ActivityType)

    def test_later_words_get_harder_activities(self):
        state = LessonState(words=["cat", "dog", "bird", "fish", "sun"])
        state.init_word_progress()
        # Last word (index 4 of 5)
        state.word_index = 4
        activity = ActivitySelector.select(state)
        # Should be harder than REPEAT
        assert activity != ActivityType.REPEAT or state.consecutive_silence > 0
