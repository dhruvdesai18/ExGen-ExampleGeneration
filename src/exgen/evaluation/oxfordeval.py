from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

from exgen.config import load_oxford_evaluation_config
from exgen.evaluation.base import Evaluator
from exgen.evaluation.verifier import GeminiVerifier
from exgen.types import GeneratedExample, EvaluationRecord

PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"


class OxfordEvalEvaluator(Evaluator):
    metric_name = "oxfordeval"

    def __init__(self) -> None:
        self.verifier = GeminiVerifier()
        self.config = load_oxford_evaluation_config()
        self.rng = random.Random()

    def evaluate(self, example: GeneratedExample) -> EvaluationRecord:
        lexical_item = example.lexical_item

        generated_first = (
            self.rng.choice([True, False])
            if self.config.randomize_order
            else True
        )

        if generated_first:
            sentence_a = example.example_text
            sentence_b = "Example sentence to change"
            generated_label = "A"
        else:
            sentence_a = "Example sentence to change"
            sentence_b = example.example_text
            generated_label = "B"

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            word=lexical_item.word,
            pos=lexical_item.pos,
            definition=lexical_item.definition,
            sentence_a=sentence_a,
            sentence_b=sentence_b,
        )

        raw_response = self.verifier.verify(system_prompt, user_prompt)
        parsed = self._parse_response(raw_response)

        generated_wins = parsed["preferred"] == generated_label

        return EvaluationRecord(
            id=str(uuid.uuid4()),
            metric_name=self.metric_name,
            lexical_item_id=lexical_item.id,
            metadata={
                "score": 1.0 if generated_wins else 0.0,
                "preferred": parsed["preferred"],
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
                "generated_first": generated_first,
                "generated_label": generated_label,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
            },
        )

    def _build_system_prompt(self) -> str:
        return (self._prompt_dir() / "system.txt").read_text(encoding="utf-8")

    def _build_user_prompt(
            self,
            *,
            word: str,
            pos: str,
            definition: str,
            sentence_a: str,
            sentence_b: str,
    ) -> str:
        template = (self._prompt_dir() / "user.txt").read_text(encoding="utf-8")
        replacements = {
            "word": word,
            "pos": pos,
            "definition": definition,
            "sentence_a": sentence_a,
            "sentence_b": sentence_b,
        }

        for key, value in replacements.items():
            template = template.replace(f"{{{{{key}}}}}", value)

        return template

    def _prompt_dir(self) -> Path:
        return PROMPTS_ROOT / self.config.prompt_version

    def _parse_response(self, raw_response: str) -> dict[str, Any]:
        parsed = json.loads(raw_response)

        required_keys = {"preferred", "confidence", "reason"}
        missing = required_keys - parsed.keys()
        if missing:
            raise ValueError(f"Verifier response missing keys: {sorted(missing)}")

        if parsed["preferred"] not in {"A", "B"}:
            raise ValueError("`preferred` must be 'A' or 'B'.")

        if parsed["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("Invalid confidence value.")

        return parsed
