from __future__ import annotations

import json
import re
from typing import Any


def parse_llm_json(raw_response: str) -> dict[str, Any]:
    """Parse near-JSON model output after removing common formatting artifacts."""
    candidate = raw_response.strip()
    candidate = _strip_code_fences(candidate)
    candidate = _extract_json_object(candidate)
    candidate = _remove_trailing_commas(candidate)
    return json.loads(candidate)


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1)
        text = re.sub(r"\s*```$", "", text, count=1)
    return text.strip()


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return text[start : end + 1]
    return text


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)
