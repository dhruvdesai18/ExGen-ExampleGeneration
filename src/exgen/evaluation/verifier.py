from __future__ import annotations

import os
import random
import time

from dotenv import load_dotenv

from exgen.evaluation.base import Verifier


class GeminiVerifier(Verifier):
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        load_dotenv()

        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.cooldown_seconds = _read_float_env("EXGEN_GEMINI_COOLDOWN_SECONDS", default=4.0)
        self.max_retries = _read_int_env("EXGEN_GEMINI_MAX_RETRIES", default=4)
        self.retry_backoff_seconds = _read_float_env(
            "EXGEN_GEMINI_RETRY_BACKOFF_SECONDS",
            default=10.0,
        )
        self.retry_jitter_seconds = _read_float_env(
            "EXGEN_GEMINI_RETRY_JITTER_SECONDS",
            default=2.0,
        )
        self.thinking_level = (
            os.environ.get("EXGEN_GEMINI_THINKING_LEVEL", "low").strip().lower() or "low"
        )
        self._thinking_supported = True
        self._last_request_finished_at = 0.0

        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY or pass api_key explicitly."
            )

        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def verify(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict | None = None,
    ) -> str:
        from google.genai import types
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._respect_cooldown()
            try:
                config = self._build_config(
                    types=types,
                    system_prompt=system_prompt,
                    response_schema=response_schema,
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                )
                self._last_request_finished_at = time.monotonic()
                raw_text = (response.text or "").strip()

                if not raw_text:
                    raise ValueError(f"Gemini response contained no text payload: {response}")

                return raw_text
            except Exception as exc:
                self._last_request_finished_at = time.monotonic()
                if self._thinking_supported and _looks_like_unsupported_thinking_error(exc):
                    self._thinking_supported = False
                    continue
                last_error = exc
                if attempt >= self.max_retries or not _looks_like_rate_limit_error(exc):
                    raise
                delay = self.retry_backoff_seconds * (attempt + 1) + random.uniform(
                    0.0,
                    self.retry_jitter_seconds,
                )
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini verification failed without an exception.")

    def _build_config(self, *, types, system_prompt: str, response_schema: dict | None):
        config_kwargs = {
            "system_instruction": system_prompt,
            "temperature": 0.0,
            "response_mime_type": "application/json",
        }
        if response_schema is not None:
            config_kwargs["response_json_schema"] = response_schema
        if self._thinking_supported and self.thinking_level != "off":
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self.thinking_level,
            )
        return types.GenerateContentConfig(**config_kwargs)

    def _respect_cooldown(self) -> None:
        if self.cooldown_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_finished_at
        remaining = self.cooldown_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _read_float_env(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "429",
            "rate limit",
            "quota",
            "resource exhausted",
            "too many requests",
        )
    )


def _looks_like_unsupported_thinking_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "thinking level is not supported for this model" in message
