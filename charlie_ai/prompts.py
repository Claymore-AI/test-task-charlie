"""Prompt engineering — system prompts, few-shot examples, and message builders.

This module is the **single source of truth** for every LLM interaction.
Each public function returns a ``list[dict]`` of OpenAI-format messages
(system / user / assistant) ready to be passed to ``LLMClient.generate()``.

Design principles
~~~~~~~~~~~~~~~~~
1. **Charlie persona lives in ``system`` role** — this is the strongest
   behavioral anchor for chat models.
2. **Conversation history is always included** — Charlie must remember
   what happened earlier in the lesson.  A sliding window (last N turns)
   keeps us within context limits.
3. **Few-shot examples** — show the model exactly what good responses
   look like for each scenario.
4. **Separate prompts per agent** — the Evaluator gets a clinical,
   analytical prompt; the Responder gets a warm, creative one.
5. **Engagement-aware modifiers** — prompts adapt based on the child's
   engagement signals (silence streaks, off-topic streaks, correct streaks).
"""

from __future__ import annotations

from .config import settings
from .models import (
    ActivityType,
    Emotion,
    EvalResult,
    EvalStatus,
    LessonState,
    Message,
)

# History window

MAX_HISTORY_TURNS = 10  # last N messages included in prompts


def build_conversation_messages(
    history: list[Message],
    max_turns: int = MAX_HISTORY_TURNS,
) -> list[dict[str, str]]:
    """Convert ``LessonState.history`` into OpenAI-format messages.

    Maps ``role="charlie"`` → ``assistant`` and ``role="child"`` → ``user``.
    Keeps the most recent *max_turns* messages to stay within context limits.
    """
    recent = history[-max_turns:] if len(history) > max_turns else history
    role_map = {"charlie": "assistant", "child": "user"}
    return [
        {"role": role_map.get(m.role, "user"), "content": m.text}
        for m in recent
        if m.text.strip()
    ]


# Charlie persona (system prompt)

CHARLIE_SYSTEM_PROMPT = """\
You are Charlie — an 8-year-old fox cub from London who teaches English to \
young children (ages 4-8) through playful conversation.

## Personality
- Playful, warm, and endlessly encouraging. You LOVE learning new words!
- You speak in very short, simple sentences (max 10 words each).
- You use fun exclamations: "Yay!", "Woohoo!", "Oh wow!", "Cool beans!"
- You NEVER criticise, scold, or say "no, that's wrong."
- If a child struggles, you become EXTRA warm and supportive.
- You celebrate every tiny success like it's the best thing ever.

## Voice & Emotion
Every reply must feel like a *specific* emotion from this list:
  excited, encouraging, gentle, playful, proud, neutral.
Choose the emotion that best fits this moment. A TTS engine will use it
to control Charlie's voice prosody (pitch, speed, energy).

## Pronunciation Teaching
When introducing a word:
- Break it into syllables if helpful (e.g. "el-e-phant").
- Offer a rhyming cue or fun sound (e.g. "'cat' rhymes with 'hat'!").
- Describe how to shape the mouth (e.g. "open wide and say 'aaa'!").

## Hard Rules
- You are a FOX. Never say you are an AI, robot, or computer.
- Never use words a 4-year-old wouldn't know.
- Never discuss violence, politics, or adult topics.
- Keep replies to 1-3 short sentences max.
- If the child says something off-topic, briefly acknowledge it with \
curiosity, then gently guide back to the lesson.

## JSON Output
Always respond ONLY with valid JSON. No markdown, no extra text.\
"""


# Evaluator system prompt

EVALUATOR_SYSTEM_PROMPT = """\
You are an evaluation module for a children's English lesson. Your job is to \
analyze what a child said and classify it against a target word.

## Chain of Thought
1. First, write your reasoning: what did the child actually say? Is it \
related to the target word?
2. Then classify using EXACTLY ONE of these statuses:
   - "correct"   — the child said the target word (allow minor typos, \
wrong case, extra spaces, slight mispronunciation like "catt" for "cat").
   - "partial"   — the child said something semantically related \
(e.g. "kitty" for "cat", "doggy" for "dog").
   - "incorrect" — the child tried but said a completely wrong word.
   - "off_topic" — the child said something unrelated to the lesson \
(e.g. talking about Spiderman, asking random questions).
   - "silence"   — the input is empty, just whitespace, or "[silence]".

## For Different Activity Types
- REPEAT: child should say the exact target word.
- LISTEN_AND_PICK: child picks one word from options; check if it matches.
- RIDDLE: child guesses the word from a description.
- USE_IN_SENTENCE: child should use the target word in any sentence.

## Be GENEROUS
Children aged 4-8 make typos and mispronounce words. "Catt", "kat", "CAT" \
are all "correct" for target "cat". "Doggie" for "dog" is "partial".

## Examples
Target: "cat", Child: "cat" → correct, confidence 1.0
Target: "cat", Child: "catt" → correct, confidence 0.9
Target: "cat", Child: "kitty" → partial, confidence 0.7
Target: "cat", Child: "I like spiderman" → off_topic, confidence 0.95
Target: "cat", Child: "" → silence, confidence 1.0
Target: "cat", Child: "I see a cat" → correct, confidence 0.95 (for USE_IN_SENTENCE)

## JSON Output
Respond ONLY with valid JSON:
{"status": "<status>", "confidence": <0.0-1.0>, "reasoning": "<your analysis>"}\
"""


# Engagement-aware prompt modifiers


def _engagement_context(state: LessonState) -> str:
    """Return extra prompt guidance based on engagement signals."""
    parts: list[str] = []

    if state.child_name:
        parts.append(f"The child's name is {state.child_name}. "
                      "Use it occasionally to make things personal.")

    if state.consecutive_silence >= 2:
        parts.append(
            "IMPORTANT: The child has been silent for multiple turns. "
            "Be EXTRA gentle and warm. Give them the answer if needed. "
            "Maybe they're shy — that's totally okay!"
        )
    elif state.consecutive_silence == 1:
        parts.append(
            "The child was silent last turn. Encourage them gently. "
            "Give a small hint to help."
        )

    if state.consecutive_off_topic >= 2:
        parts.append(
            "The child keeps talking about other things. "
            "Briefly acknowledge what interests them, then simplify "
            "the task (make it easier to answer)."
        )

    if state.streak >= 3:
        parts.append(
            f"The child has a streak of {state.streak} correct answers! "
            "Be EXTRA excited and celebrate! They're on fire!"
        )

    return "\n".join(parts) if parts else ""


# Pronunciation helpers


# Simple phonetic hints for common lesson words.  In production this
# would come from a pronunciation dictionary / API.
_PHONETIC_HINTS: dict[str, str] = {
    "cat": '/kæt/ — rhymes with "hat"!',
    "dog": '/dɒɡ/ — rhymes with "log"!',
    "bird": '/bɜːrd/ — it has an "rrr" sound inside!',
    "fish": '/fɪʃ/ — ends with a "shh" like a secret!',
    "sun": '/sʌn/ — rhymes with "fun"!',
    "tree": '/triː/ — stretch the "ee" sound!',
    "house": '/haʊs/ — say "how" then add an "s"!',
    "apple": '/ˈæp.əl/ — say "ap" then "pull"!',
    "book": '/bʊk/ — rhymes with "look"!',
    "star": '/stɑːr/ — stretch the "ar" sound!',
}


def get_phonetic_hint(word: str) -> str | None:
    return _PHONETIC_HINTS.get(word.lower())


# Activity-specific instructions 


_ACTIVITY_INSTRUCTIONS: dict[ActivityType, str] = {
    ActivityType.REPEAT: (
        "This is a REPEAT exercise. Introduce the word, give a fun "
        "one-sentence description, offer a pronunciation tip, and ask "
        "the child to say the word."
    ),
    ActivityType.LISTEN_AND_PICK: (
        "This is a LISTEN AND PICK exercise. Describe the word without "
        "saying it (use clues a 4-year-old would understand) and ask "
        "the child to guess which word you're thinking of."
    ),
    ActivityType.RIDDLE: (
        "This is a RIDDLE exercise. Give a fun, simple riddle about the "
        "word (2-3 clues) and ask the child to guess the answer."
    ),
    ActivityType.USE_IN_SENTENCE: (
        "This is a USE IN SENTENCE exercise. Say a simple sentence using "
        "the word and ask the child to repeat the whole sentence."
    ),
}


# Message builders (one per agent interaction)


def greeting_messages(state: LessonState) -> list[dict[str, str]]:
    """Initial greeting — Charlie introduces himself."""
    return [
        {"role": "system", "content": CHARLIE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Start a new English lesson. Today's words are: "
                f"{', '.join(state.words)}.\n\n"
                "Greet the child warmly. Introduce yourself as Charlie "
                "the fox. Say you'll learn some fun words today. "
                "Ask the child their name.\n\n"
                "Keep it to 2-3 short sentences.\n\n"
                'Respond with JSON: {"message": "...", '
                '"emotion": "<one of: excited, encouraging, gentle, '
                'playful, proud, neutral>"}'
            ),
        },
    ]


def greeting_reply_messages(
    child_input: str, state: LessonState
) -> list[dict[str, str]]:
    """Charlie responds to the child's greeting and moves to first word."""
    safe_input = child_input.strip() if child_input.strip() else "[silence]"
    first_word = state.words[0] if state.words else "cat"
    phonetic = get_phonetic_hint(first_word)

    history = build_conversation_messages(state.history)

    msgs: list[dict[str, str]] = [
        {"role": "system", "content": CHARLIE_SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": (
                f'The child said: "{safe_input}"\n\n'
                "If they shared their name, use it! If they were silent, "
                "be warm and give them a fun nickname.\n\n"
                "1. Briefly acknowledge what they said (one short sentence).\n"
                f'2. Introduce the first word: "{first_word}".\n'
                "   Give a fun, simple description a 4-year-old would get.\n"
                f"{'3. Pronunciation tip: ' + phonetic + chr(10) if phonetic else ''}"
                "3. Ask the child to say the word.\n\n"
                "Keep it to 2-3 short sentences.\n\n"
                'Respond with JSON: {"message": "...", '
                '"emotion": "<emotion>"}'
            ),
        },
    ]
    return msgs


def word_intro_messages(
    word: str,
    word_num: int,
    total: int,
    activity_type: ActivityType,
    state: LessonState,
) -> list[dict[str, str]]:
    """Charlie introduces a new vocabulary word with an activity."""
    phonetic = get_phonetic_hint(word)
    engagement = _engagement_context(state)
    activity_instructions = _ACTIVITY_INSTRUCTIONS[activity_type]
    history = build_conversation_messages(state.history)

    return [
        {"role": "system", "content": CHARLIE_SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": (
                f"Introduce vocabulary word {word_num} of {total}: "
                f'"{word}".\n\n'
                f"Activity type: {activity_instructions}\n\n"
                f"{'Pronunciation tip: ' + phonetic + chr(10) if phonetic else ''}"
                f"{engagement + chr(10) if engagement else ''}\n"
                "Keep it to 2-3 short sentences.\n\n"
                'Respond with JSON: {"message": "...", '
                '"emotion": "<emotion>", '
                '"phonetic_hint": "<optional pronunciation tip or null>"}'
            ),
        },
    ]


def evaluator_messages(
    child_input: str,
    target_word: str,
    activity_type: ActivityType,
    state: LessonState,
) -> list[dict[str, str]]:
    """Build message list for the Evaluator agent."""
    safe_input = (
        child_input.strip()
        if child_input.strip()
        else "[silence — the child said nothing]"
    )
    history = build_conversation_messages(state.history, max_turns=6)

    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": (
                f'TARGET WORD: "{target_word}"\n'
                f"ACTIVITY TYPE: {activity_type.value}\n"
                f"ATTEMPT: {state.attempt + 1} of {settings.max_attempts_per_word}\n"
                f'CHILD SAID: "{safe_input}"\n\n'
                "Analyze the child's response. Use chain-of-thought.\n\n"
                'Respond with JSON: {"status": "<status>", '
                '"confidence": <0.0-1.0>, '
                '"reasoning": "<your analysis>"}'
            ),
        },
    ]


def responder_messages(
    eval_result: EvalResult,
    target_word: str,
    activity_type: ActivityType,
    state: LessonState,
    is_last_attempt: bool,
) -> list[dict[str, str]]:
    """Build message list for the Responder agent."""
    engagement = _engagement_context(state)
    phonetic = get_phonetic_hint(target_word)
    history = build_conversation_messages(state.history)

    status_guidance = {
        EvalStatus.CORRECT: (
            "The child got it RIGHT! Celebrate enthusiastically! "
            "Make them feel amazing."
        ),
        EvalStatus.PARTIAL: (
            "The child said something close but not exact. "
            "Praise what they got right, give a tiny hint, ask again."
        ),
        EvalStatus.INCORRECT: (
            "The child tried but said the wrong word. "
            "Say it's okay! Give a clear hint. Ask to try again."
        ),
        EvalStatus.OFF_TOPIC: (
            "The child said something unrelated to the lesson. "
            "Briefly show curiosity about what they said, "
            "then gently guide back to the word."
        ),
        EvalStatus.SILENCE: (
            "The child was silent. Be extra warm and gentle. "
            "Encourage them to try. Give a helpful hint."
        ),
    }

    last_attempt_note = ""
    if is_last_attempt and eval_result.status != EvalStatus.CORRECT:
        last_attempt_note = (
            "\nIMPORTANT: This was the LAST attempt for this word. "
            "Be extra encouraging. Say the word for them so they can "
            "learn it: 'The word is [word]! Great trying!'"
        )

    return [
        {"role": "system", "content": CHARLIE_SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": (
                f'Target word: "{target_word}" '
                f"(activity: {activity_type.value})\n"
                f"Evaluation: {eval_result.status.value} "
                f"(confidence: {eval_result.confidence})\n"
                f"Evaluator reasoning: {eval_result.reasoning}\n\n"
                f"{status_guidance.get(eval_result.status, '')}\n"
                f"{last_attempt_note}\n"
                f"{'Pronunciation tip: ' + phonetic + chr(10) if phonetic else ''}"
                f"{engagement + chr(10) if engagement else ''}\n"
                "Keep it to 1-2 short sentences. Do NOT say goodbye "
                "or wrap up the lesson.\n\n"
                'Respond with JSON: {"message": "...", '
                '"emotion": "<emotion>", '
                '"phonetic_hint": "<optional or null>"}'
            ),
        },
    ]


def review_messages(
    words: list[str], state: LessonState
) -> list[dict[str, str]]:
    """Build review / recap prompt for 2-3 words from the lesson."""
    engagement = _engagement_context(state)
    words_str = ", ".join(words[:3])
    history = build_conversation_messages(state.history)

    return [
        {"role": "system", "content": CHARLIE_SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": (
                "Time for a quick review game! Pick one of these words: "
                f"{words_str}.\n\n"
                "Give a fun riddle or description clue (NOT the word "
                "itself) and ask the child to guess.\n\n"
                f"{engagement + chr(10) if engagement else ''}"
                "Keep it to 2-3 short sentences.\n\n"
                'Respond with JSON: {"message": "...", '
                '"emotion": "playful", '
                '"phonetic_hint": null}'
            ),
        },
    ]


def farewell_messages(state: LessonState) -> list[dict[str, str]]:
    """Build farewell message using lesson context."""
    engagement = _engagement_context(state)
    history = build_conversation_messages(state.history)

    mastered = [
        wp.word for wp in state.word_progress if wp.is_mastered
    ]
    mastered_str = ", ".join(mastered) if mastered else ", ".join(state.words)

    return [
        {"role": "system", "content": CHARLIE_SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": (
                "The lesson is over! The child practised these words: "
                f"{', '.join(state.words)}.\n"
                f"Words they mastered: {mastered_str}.\n"
                f"Streak: {state.streak}, "
                f"Score: {state.total_correct}/{state.total_attempts}.\n\n"
                f"{engagement + chr(10) if engagement else ''}"
                "Say a warm, personalized goodbye:\n"
                "- Tell them they did an amazing job.\n"
                "- Mention 1-2 specific words they learned.\n"
                "- Say you hope to see them again soon.\n"
                f"{'- Use their name: ' + state.child_name + chr(10) if state.child_name else ''}"
                "\nKeep it to 2-3 short sentences.\n\n"
                '{"message": "...", "emotion": "proud"}'
            ),
        },
    ]
