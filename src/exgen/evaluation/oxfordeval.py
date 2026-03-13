from __future__ import annotations

import json
import random
import uuid

from src.exgen.evaluation.base import Evaluator, Verifier
from src.exgen.types import GeneratedExample, EvaluationRecord


class MockExampleBuilder:
    def build(self, example: GeneratedExample) -> str:
        lexical_item = example.lexical_item
        return (
            f"The word {lexical_item.word} appears in an everyday sentence "
            f"about {lexical_item.definition.lower()}."
        )


class OxfordEvalEvaluator(Evaluator):
    metric_name = "oxfordeval"

    def __init__(
        self,
        verifier: Verifier,
        mock_builder: MockExampleBuilder | None = None,
        seed: int = 42,
    ) -> None:
        self.verifier = verifier
        self.mock_builder = mock_builder or MockExampleBuilder()
        self.rng = random.Random(seed)

    def evaluate(self, example: GeneratedExample) -> EvaluationRecord:
        lexical_item = example.lexical_item
        mock_example = self.mock_builder.build(example)

        generated_first = self.rng.choice([True, False])

        if generated_first:
            sentence_a = example.example_text
            sentence_b = mock_example
            generated_label = "A"
        else:
            sentence_a = mock_example
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
        return (
            "You are an expert lexicographer evaluating example sentences for a "
            "language learning dictionary. Decide which sentence is the better "
            "dictionary example for the given word sense. Return only JSON."
        )

    def _build_user_prompt(
        self,
        *,
        word: str,
        pos: str,
        definition: str,
        sentence_a: str,
        sentence_b: str,
    ) -> str:
        return (
            f"Headword: {word}\n"
            f"Part of speech: {pos}\n"
            f"Definition: {definition}\n\n"
            f"Sentence A: {sentence_a}\n"
            f"Sentence B: {sentence_b}\n\n"
            "Return JSON in this exact shape:\n"
            '{'
            '"preferred": "A" or "B", '
            '"confidence": "high" or "medium" or "low", '
            '"reason": "short explanation"'
            '}'
        )

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
