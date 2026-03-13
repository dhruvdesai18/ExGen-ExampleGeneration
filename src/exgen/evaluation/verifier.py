from __future__ import annotations

import os

import requests

from src.exgen.evaluation.base import Verifier


class GeminiVerifier(Verifier):
    def __init__(
            self,
            api_key: str | None = None,
            model_name: str = "gemini-3.0-flash",
            timeout_seconds: int = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY or pass api_key explicitly."
            )

    def verify(self, system_prompt: str, user_prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent"
        )
        response = requests.post(
            url,
            params={"key": self.api_key},
            json={
                "systemInstruction": {
                    "parts": [{"text": system_prompt}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                },
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError(f"Gemini returned no candidates: {data}")

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if "text" in part]
        raw_text = "".join(text_parts).strip()

        if not raw_text:
            raise ValueError(f"Gemini response contained no text payload: {data}")

        return raw_text
