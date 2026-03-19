from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from exgen.config import OxfordEvaluationConfig
from exgen.evaluation.base import Evaluator
from exgen.evaluation.verifier import GeminiVerifier
from exgen.types import EvaluationOutcome, GeneratedExample

PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"


class OxfordEvalEvaluator(Evaluator):
    metric_name = "oxfordeval"

    def __init__(self, config: OxfordEvaluationConfig) -> None:
        self.config = config
        self.verifier = GeminiVerifier(model_name=config.judge_model_name)
        self.rng = random.Random()

    def evaluate(self, example: GeneratedExample) -> EvaluationOutcome:
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

        return EvaluationOutcome(
            score=1.0 if generated_wins else 0.0,
            result={
                "preferred": parsed["preferred"],
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
            },
            metadata={
                "generated_first": generated_first,
                "generated_label": generated_label,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
            },
            provenance={
                "metric_name": self.metric_name,
                "metric_config_name": self.config.name,
                "metric_prompt_version": self.config.prompt_version,
                "judge_model": self.verifier.model_name,
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
