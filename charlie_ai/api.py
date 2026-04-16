"""FastAPI REST layer for mobile app (Expo / React Native) integration.

Endpoints:
    POST /lesson/start      — Begin a new lesson, returns session_id + first TurnResponse.
    POST /lesson/{id}/turn  — Send child's text, returns TurnResponse.
    GET  /lesson/{id}/progress — Query current lesson progress.

In production, the ``_sessions`` dict should be replaced with Redis or a
database backend so the service is horizontally scalable and stateless.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import LessonEngine
from .models import LessonProgress, TurnResponse

app = FastAPI(
    title="Charlie AI — Vocabulary Lesson Engine",
    version="1.0.0",
    description="Multi-agent lesson engine for children's English vocabulary.",
)

# ── In-memory session store (swap for Redis in production) ────────────
_sessions: dict[str, LessonEngine] = {}


# ── Request / Response schemas ────────────────────────────────────────

class StartRequest(BaseModel):
    words: list[str] = Field(..., min_length=1)


class StartResponse(BaseModel):
    session_id: str
    turn: TurnResponse


class TurnRequest(BaseModel):
    text: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/lesson/start", response_model=StartResponse)
async def start_lesson(req: StartRequest) -> StartResponse:
    """Create a new lesson session and return the greeting turn."""
    session_id = uuid.uuid4().hex
    engine = LessonEngine(words=req.words)
    greeting = await engine.process("")
    _sessions[session_id] = engine
    return StartResponse(session_id=session_id, turn=greeting)


@app.post("/lesson/{session_id}/turn", response_model=TurnResponse)
async def submit_turn(session_id: str, req: TurnRequest) -> TurnResponse:
    """Process one child turn and return Charlie's response."""
    engine = _sessions.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await engine.process(req.text)


@app.get("/lesson/{session_id}/progress", response_model=LessonProgress)
async def get_progress(session_id: str) -> LessonProgress:
    """Return the current lesson progress snapshot."""
    engine = _sessions.get(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return engine.get_progress()
