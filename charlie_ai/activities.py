"""Activity selection and pedagogical scaffolding.

The ``ActivitySelector`` chooses which exercise type to use for each
vocabulary word, based on:

1. **Word position** — the first word always starts with REPEAT (warm-up).
2. **Child performance** — struggling children get easier activities;
   children on a streak get harder ones.
3. **Variety** — consecutive words don't get the same activity type.
4. **Scaffolding** — if a child fails an activity, the next attempt falls
   back to an easier type for the same word.

Difficulty ordering::

    REPEAT (easiest) → LISTEN_AND_PICK → RIDDLE → USE_IN_SENTENCE (hardest)
"""

from __future__ import annotations

from .models import (
    ACTIVITY_DIFFICULTY,
    ActivityType,
    LessonState,
)

# Ordered from easiest to hardest.
_DIFFICULTY_ORDER: list[ActivityType] = sorted(
    ActivityType, key=lambda a: ACTIVITY_DIFFICULTY[a]
)


class ActivitySelector:
    """Picks the next activity type for a vocabulary word."""

    @staticmethod
    def select(state: LessonState) -> ActivityType:
        """Choose an activity for the current word based on context."""
        word_idx = state.word_index
        wp = state.current_word_progress

        # First word → always start with REPEAT for warm-up.
        if word_idx == 0 and (wp is None or wp.attempts == 0):
            return ActivityType.REPEAT

        # Determine base difficulty from child performance.
        base = ActivitySelector._base_difficulty(state)

        # Avoid repeating the same activity type as the previous word.
        if word_idx > 0 and state.word_progress[word_idx - 1].activities_completed:
            last_activity = state.word_progress[word_idx - 1].activities_completed[-1]
            if base == last_activity and base != ActivityType.REPEAT:
                # One step easier to create variety.
                idx = _DIFFICULTY_ORDER.index(base)
                base = _DIFFICULTY_ORDER[max(0, idx - 1)]

        return base

    @staticmethod
    def scaffold_down(current: ActivityType) -> ActivityType:
        """Fall back to an easier activity after a failed attempt.

        If already at the easiest level (REPEAT), stays there.
        """
        idx = _DIFFICULTY_ORDER.index(current)
        return _DIFFICULTY_ORDER[max(0, idx - 1)]

    @staticmethod
    def _base_difficulty(state: LessonState) -> ActivityType:
        """Determine baseline difficulty from engagement signals."""
        # Struggling child → keep it simple.
        if state.consecutive_silence >= 2 or state.consecutive_off_topic >= 2:
            return ActivityType.REPEAT

        # On a streak → ramp up difficulty.
        if state.streak >= 3:
            return ActivityType.RIDDLE
        if state.streak >= 2:
            return ActivityType.LISTEN_AND_PICK

        # Default progression by word index.
        word_idx = state.word_index
        total = len(state.words)

        if total <= 1:
            return ActivityType.REPEAT

        # Map word position to difficulty: early words easier, later harder.
        progress_ratio = word_idx / max(total - 1, 1)
        if progress_ratio < 0.33:
            return ActivityType.REPEAT
        elif progress_ratio < 0.66:
            return ActivityType.LISTEN_AND_PICK
        else:
            return ActivityType.RIDDLE
