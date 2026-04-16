"""Async LLM client — thin structured-output wrapper around the Groq SDK.

Responsibilities
~~~~~~~~~~~~~~~~
- Accept **multi-turn message lists** (system / user / assistant roles) so
  that agents can pass conversation history for context-aware generation.
- Enforce JSON-mode output via ``response_format``.
- Parse raw JSON into a caller-supplied Pydantic model.
- Retry on transient parse / validation failures.

The client is intentionally **stateless** — it holds only the API key and
model name.  All session context is managed externally by the engine /
agents.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from groq import AsyncGroq
from pydantic import BaseModel, ValidationError

from .config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Async LLM client that guarantees structured (JSON) responses."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        key = api_key or settings.groq_api_key
        if not key:
            raise ValueError(
                "GROQ_API_KEY is required. "
                "Set it in .env or pass it explicitly."
            )
        self._client = AsyncGroq(api_key=key)
        self._model = model or settings.groq_model

    async def generate(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float = 0.7,
        max_tokens: int = 400,
        max_retries: int = 2,
    ) -> T:
        """Send a **multi-turn** message list to Groq and return a validated
        *response_model* instance.

        Parameters
        messages:
            OpenAI-format list of ``{"role": ..., "content": ...}`` dicts.
            The caller is responsible for structuring system / user /
            assistant turns — this keeps prompt engineering in the prompts
            module, not here.
        response_model:
            Pydantic model used to parse and validate the JSON response.
        temperature:
            Sampling temperature.  Evaluator uses ~0.2 for consistency;
            Responder uses ~0.7 for creativity.
        max_tokens:
            Maximum tokens in the completion.
        max_retries:
            Extra retries on parse / validation failure.

        Three layers enforce structured output:
        1. ``response_format=json_object`` constrains token generation.
        2. The prompt itself contains the exact JSON schema.
        3. Pydantic validates the parsed dict against *response_model*.
        """
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 2):
            try:
                completion = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                )
                raw = completion.choices[0].message.content or ""
                logger.debug("LLM raw (attempt %d): %s", attempt, raw)

                data = json.loads(raw)
                return response_model.model_validate(data)

            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning(
                    "Attempt %d — parse/validation error: %s", attempt, exc
                )
                last_error = exc
            except Exception as exc:
                logger.error("Attempt %d — LLM error: %s", attempt, exc)
                last_error = exc

        raise RuntimeError(
            f"LLM call failed after {max_retries + 1} attempts"
        ) from last_error
