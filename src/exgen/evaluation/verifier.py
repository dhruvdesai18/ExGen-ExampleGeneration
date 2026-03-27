from __future__ import annotations

import os

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

        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY or pass api_key explicitly."
            )

        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def verify(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        raw_text = (response.text or "").strip()

        if not raw_text:
            raise ValueError(f"Gemini response contained no text payload: {response}")

        return raw_text
