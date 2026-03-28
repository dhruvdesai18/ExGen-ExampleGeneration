from __future__ import annotations

from pathlib import Path
from typing import Any

from exgen.config import GdexEvaluationConfig
from exgen.evaluation.base import Evaluator
from exgen.evaluation.json_utils import parse_llm_json
from exgen.evaluation.verifier import GeminiVerifier
from exgen.types import EvaluationOutcome, GeneratedExample

PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"
CRITERION_SCORES = {
    "yes": 1.0,
    "somewhat": 0.5,
    "no": 0.0,
}


class GdexEvaluator(Evaluator):
    metric_name = "gdex"

    def __init__(self, config: GdexEvaluationConfig) -> None:
        self.config = config
        self.verifier = GeminiVerifier(model_name=config.judge_model_name)

    def evaluate(self, example: GeneratedExample) -> EvaluationOutcome:
        """Rate one generated sentence against the GDEX rubric."""
        lexical_item = example.lexical_item
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            word=lexical_item.word,
            pos=lexical_item.pos,
            definition=lexical_item.definition,
            sentence=example.example_text,
        )

        raw_response = self.verifier.verify(
            system_prompt,
            user_prompt,
            response_schema=self._response_schema(),
        )
        parsed = self._parse_response(raw_response)
        criterion_scores = {
            criterion: CRITERION_SCORES[parsed[criterion]["label"]]
            for criterion in self.config.criteria
        }
        mean_score = sum(criterion_scores.values()) / len(criterion_scores)

        return EvaluationOutcome(
            score=mean_score,
            result={
                "criteria": parsed,
                "criterion_scores": criterion_scores,
                "full_yes": all(
                    parsed[criterion]["label"] == "yes"
                    for criterion in self.config.criteria
                ),
            },
            metadata={
                "criteria_order": list(self.config.criteria),
            },
            provenance={
                "metric_name": self.metric_name,
                "metric_config_name": self.config.name,
                "metric_prompt_version": self.config.prompt_version,
                "judge_model": self.verifier.model_name,
            },
        )

    def _build_system_prompt(self) -> str:
        """Load the system prompt for the configured GDEX prompt version."""
        return (self._prompt_dir() / "system.txt").read_text(encoding="utf-8")

    def _build_user_prompt(
        self,
        *,
        word: str,
        pos: str,
        definition: str,
        sentence: str,
    ) -> str:
        """Populate the GDEX user prompt template with lexical item fields."""
        template = (self._prompt_dir() / "user.txt").read_text(encoding="utf-8")
        replacements = {
            "word": word,
            "pos": pos,
            "definition": definition,
            "sentence": sentence,
            "criteria_list": ", ".join(self.config.criteria),
        }

        for key, value in replacements.items():
            template = template.replace(f"{{{{{key}}}}}", value)

        return template

    def _prompt_dir(self) -> Path:
        """Resolve the prompt directory for the configured GDEX prompt version."""
        return PROMPTS_ROOT / str(self.config.prompt_version)

    def _response_schema(self) -> dict[str, Any]:
        criterion_object = {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["yes", "somewhat", "no"],
                },
                "reason": {"type": "string"},
            },
            "required": ["label", "reason"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                criterion: criterion_object
                for criterion in self.config.criteria
            },
            "required": list(self.config.criteria),
            "additionalProperties": False,
        }

    def _parse_response(self, raw_response: str) -> dict[str, Any]:
        """Validate the structured GDEX rubric response returned by the judge model."""
        parsed = parse_llm_json(raw_response)

        missing = set(self.config.criteria) - parsed.keys()
        if missing:
            raise ValueError(f"GDEX response missing criteria: {sorted(missing)}")

        for criterion in self.config.criteria:
            payload = parsed[criterion]
            if not isinstance(payload, dict):
                raise ValueError(f"GDEX criterion '{criterion}' must be an object.")

            label = payload.get("label")
            if label not in CRITERION_SCORES:
                raise ValueError(
                    f"Invalid label for '{criterion}': {label!r}. "
                    "Expected one of 'yes', 'somewhat', or 'no'."
                )

            reason = payload.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"GDEX criterion '{criterion}' must include a non-empty reason.")

        return parsed
