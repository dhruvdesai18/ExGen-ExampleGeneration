from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterator

from dotenv import load_dotenv

from .base import Model


@dataclass
class GPT(Model):
    model_id: str
    timeout_seconds: int | None = None
    _client: Any = field(default=None, init=False, repr=False)

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY before using GPT.")

        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.timeout_seconds is not None:
            client_kwargs["timeout"] = self.timeout_seconds
        self._client = OpenAI(**client_kwargs)
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]],
        num_candidates: int,
    ) -> Iterator[str]:
        client = self._ensure_client()
        input_items = [
            {
                "role": message["role"],
                "content": [
                    {
                        "type": "input_text",
                        "text": message["content"],
                    }
                ],
            }
            for message in messages
        ]

        for _ in range(num_candidates):
            response = client.responses.create(
                model=self.model_id,
                input=input_items,
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=self.max_tokens,
            )
            text = (response.output_text or "").strip()
            if not text:
                raise ValueError(f"OpenAI response contained no text payload: {response}")
            yield self._extract_example(text)

    @staticmethod
    def _extract_example(text: str) -> str:
        start_tag = "<EXAMPLE>"
        end_tag = "</EXAMPLE>"
        start = text.find(start_tag)
        if start == -1:
            return text
        end = text.find(end_tag, start)
        if end == -1:
            return text
        return text[start + len(start_tag) : end].strip()
