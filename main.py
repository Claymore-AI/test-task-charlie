#!/usr/bin/env python3
"""Interactive CLI for Charlie AI — run a vocabulary lesson in the terminal.

Shows the rich TurnResponse data (emotion, activity, progress, pronunciation
hints) that a mobile app would use to render each turn.
"""

from __future__ import annotations

import asyncio
import sys

from charlie_ai.config import settings
from charlie_ai.engine import LessonEngine

BANNER = """
╔══════════════════════════════════════════════════╗
║            Charlie AI — English Lesson           ║
╠══════════════════════════════════════════════════╣
║  Type what the child says and press Enter.       ║
║  Empty input = the child stayed silent.          ║
║  Type "quit" to exit at any time.                ║
╚══════════════════════════════════════════════════╝
"""


def _format_turn(turn) -> str:
    """Format a TurnResponse for the terminal."""
    parts = [f" 🦊 Charlie [{turn.emotion.value}]: {turn.message}"]

    meta: list[str] = []
    if turn.highlight_word:
        meta.append(f"word: {turn.highlight_word}")
    if turn.activity_type:
        meta.append(f"activity: {turn.activity_type.value}")
    if turn.phonetic_hint:
        meta.append(f"pronunciation: {turn.phonetic_hint}")
    if turn.image_hint:
        meta.append(f"image: 🖼  {turn.image_hint}")
    if turn.expected_response:
        meta.append(f"expected: \"{turn.expected_response}\"")

    if meta:
        parts.append(f"    ⌊ {' │ '.join(meta)}")

    if turn.progress:
        p = turn.progress
        bar_len = 20
        filled = int(bar_len * p.words_completed / max(p.total_words, 1))
        bar = "█" * filled + "░" * (bar_len - filled)
        parts.append(
            f"    ⌊ progress: [{bar}] "
            f"{p.words_completed}/{p.total_words} "
            f"│ streak: {p.streak} │ score: {p.score}"
        )

    return "\n".join(parts)


async def main(words: list[str] | None = None) -> None:
    lesson_words = words or settings.default_words
    print(BANNER)
    print(f"  Words for today: {', '.join(lesson_words)}\n")

    engine = LessonEngine(words=lesson_words)

    # Charlie speaks first — send empty input to trigger the greeting.
    greeting = await engine.process("")
    print(_format_turn(greeting))
    print()

    while not engine.is_finished:
        try:
            user_input = input(" 👧 Child:   ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if user_input.strip().lower() == "quit":
            print("Lesson ended early. Bye!")
            return

        reply = await engine.process(user_input)
        print()
        print(_format_turn(reply))
        print()

    print(" ✅ Lesson complete!\n")


if __name__ == "__main__":
    # Optional: pass custom words as CLI arguments.
    #   python main.py apple house tree
    custom_words = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(main(custom_words))
