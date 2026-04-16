"""Data models — lesson state, LLM response schemas, and structured turn responses.

All mutable session data lives in ``LessonState``, which is fully Pydantic-
serialisable for stateless API deployment (store in Redis / PostgreSQL between
HTTP requests from the mobile app).

LLM response contracts are modelled as separate Pydantic schemas — one per
agent — so ``LLMClient`` can validate every response before business logic
touches it.

``TurnResponse`` is the rich structured payload returned to the frontend
(Expo / React Native).  It carries everything the mobile app needs to render
a turn: Charlie's spoken text, emotion tag for TTS prosody, pronunciation
hints, image/word highlights, the active exercise type, and a progress
snapshot for the UI progress bar.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# Enums

class Phase(str, Enum):
    """Top-level lesson phase."""

    GREETING = "greeting"
    VOCABULARY = "vocabulary"
    REVIEW = "review"
    FAREWELL = "farewell"
    ENDED = "ended"


class SubPhase(str, Enum):
    """Sub-phase within the VOCABULARY phase."""

    INTRODUCE = "introduce"
    PRACTICE = "practice"


class EvalStatus(str, Enum):
    """Result of the Evaluator agent's analysis of a child's attempt."""

    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    OFF_TOPIC = "off_topic"
    SILENCE = "silence"


class Emotion(str, Enum):
    """Emotion tag attached to every Charlie utterance.

    The TTS engine maps each tag to prosody parameters (pitch, rate,
    energy) so Charlie's voice *sounds* the way he feels.
    """

    EXCITED = "excited"  # "Yay! You got it!"
    ENCOURAGING = "encouraging"  # "You can do it!"
    GENTLE = "gentle"  # handling silence / struggle
    PLAYFUL = "playful"  # riddles, jokes, word games
    PROUD = "proud"  # streak celebrations
    NEUTRAL = "neutral"  # default / transitional


class ActivityType(str, Enum):
    """Exercise format for a vocabulary turn.

    Ordered by pedagogical difficulty so the ``ActivitySelector`` can
    scaffold appropriately.
    """

    REPEAT = "repeat"  # "Can you say 'cat'?"
    LISTEN_AND_PICK = "listen_and_pick"  # "Which word means a fluffy pet that goes meow?"
    RIDDLE = "riddle"  # "I have whiskers and I purr. What am I?"
    USE_IN_SENTENCE = "use_in_sentence"  # "Can you say: I see a cat?"


# Activity difficulty ordering — used by ActivitySelector for scaffolding.
ACTIVITY_DIFFICULTY: dict[ActivityType, int] = {
    ActivityType.REPEAT: 0,
    ActivityType.LISTEN_AND_PICK: 1,
    ActivityType.RIDDLE: 2,
    ActivityType.USE_IN_SENTENCE: 3,
}


# Conversation history


class Message(BaseModel):
    """Single dialogue turn persisted in ``LessonState.history``."""

    role: str  # "charlie" | "child"
    text: str


# Per-word progress


class WordProgress(BaseModel):
    """Tracks the child's progress on a single vocabulary word."""

    word: str
    attempts: int = 0
    is_mastered: bool = False
    activities_completed: list[ActivityType] = Field(default_factory=list)
    last_eval: EvalStatus | None = None


# Session state


class LessonState(BaseModel):
    """Complete snapshot of a lesson session.

    Designed to be serialisable (e.g. to Redis / PostgreSQL) so the
    service can be fully stateless between HTTP requests from the Expo
    mobile app.
    """

    # Phase control
    phase: Phase = Phase.GREETING
    sub_phase: SubPhase = SubPhase.INTRODUCE
    greeting_sent: bool = False

    # Vocabulary progress
    words: list[str] = Field(default_factory=list)
    word_index: int = 0
    attempt: int = 0
    current_activity: ActivityType = ActivityType.REPEAT
    word_progress: list[WordProgress] = Field(default_factory=list)

    # Engagement tracking
    child_name: str | None = None
    streak: int = 0
    consecutive_silence: int = 0
    consecutive_off_topic: int = 0
    total_correct: int = 0
    total_attempts: int = 0

    # Conversation history
    history: list[Message] = Field(default_factory=list)

    def init_word_progress(self) -> None:
        """Populate ``word_progress`` from ``words`` if not already set."""
        if not self.word_progress and self.words:
            self.word_progress = [WordProgress(word=w) for w in self.words]

    @property
    def current_word(self) -> str | None:
        if 0 <= self.word_index < len(self.words):
            return self.words[self.word_index]
        return None

    @property
    def current_word_progress(self) -> WordProgress | None:
        if 0 <= self.word_index < len(self.word_progress):
            return self.word_progress[self.word_index]
        return None


# Lesson progress (for UI)


class LessonProgress(BaseModel):
    """Lightweight progress snapshot for the frontend progress bar."""

    words_completed: int
    total_words: int
    current_word: str | None
    streak: int
    score: int  # total_correct


# LLM response schemas (one per agent)


class CharlieMessage(BaseModel):
    """Response schema for the Responder agent."""

    message: str
    emotion: Emotion = Emotion.NEUTRAL


class EvalResult(BaseModel):
    """Response schema for the Evaluator agent.

    ``reasoning`` captures the chain-of-thought so we can debug
    unexpected evaluations without replaying the full prompt.
    """

    status: EvalStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ResponderOutput(BaseModel):
    """Full Responder agent output — Charlie's reply plus metadata."""

    message: str
    emotion: Emotion = Emotion.NEUTRAL
    phonetic_hint: str | None = None


class SafetyVerdict(BaseModel):
    """Response schema for the Safety agent."""

    is_safe: bool
    reason: str | None = None


# Structured turn response for frontend


class TurnResponse(BaseModel):
    """Rich, structured payload returned to the mobile app for every turn.

    The frontend uses these fields to:
    - Play Charlie's voice with correct emotion (``emotion`` → TTS prosody)
    - Highlight the target word on screen (``highlight_word``)
    - Show a pronunciation guide (``phonetic_hint``)
    - Display a matching illustration (``image_hint``)
    - Pre-populate STT expected text (``expected_response``)
    - Render a progress bar (``progress``)
    - Show the right exercise UI (``activity_type``)
    """

    message: str
    emotion: Emotion = Emotion.NEUTRAL
    highlight_word: str | None = None
    phonetic_hint: str | None = None
    image_hint: str | None = None
    expected_response: str | None = None
    activity_type: ActivityType | None = None
    progress: LessonProgress | None = None
