from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from exgen.mlm.mlm_metric import MaskedLMRanker
from exgen.types import GeneratedExample, LexicalItem

DEFAULT_MLM_MODELS = {
    "en": "roberta-large",
    "hi": "google/muril-base-cased",
    "uk": "xlm-roberta-large",
}


@dataclass
class MLMCandidateRanker:
    model_names: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MLM_MODELS))
    ranker_factory: Callable[[str], MaskedLMRanker] = MaskedLMRanker
    _rankers: dict[str, MaskedLMRanker] = field(default_factory=dict, init=False, repr=False)

    def annotate(
        self,
        lexical_item: LexicalItem,
        examples: list[GeneratedExample],
    ) -> list[GeneratedExample]:
        if not examples:
            return examples

        model_name = self.model_names.get(lexical_item.language)
        if model_name is None:
            return self._apply_fallback(examples, model_name="unsupported-language")

        ranker = self._rankers.get(lexical_item.language)
        if ranker is None:
            ranker = self.ranker_factory(model_name)
            self._rankers[lexical_item.language] = ranker

        best_index = 0
        best_log_prob = float("-inf")
        found_success = False

        for index, example in enumerate(examples):
            result = ranker.get_exemplification_score(example.example_text, lexical_item.word)
            status = str(result.get("status", "failure"))
            metadata = {
                "candidate_index": index,
                "mlm_model_name": model_name,
                "mlm_status": status,
                "mlm_score": result.get("score"),
                "mlm_log_prob": result.get("log_prob"),
            }
            example.metadata.update(metadata)

            if status.upper() == "SUCCESS":
                found_success = True
                log_prob = float(result["log_prob"])
                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    best_index = index

        if not found_success:
            return self._apply_fallback(examples, model_name=model_name)

        for index, example in enumerate(examples):
            example.metadata["selected"] = index == best_index
            example.metadata["mlm_fallback"] = False
        return examples

    def _apply_fallback(
        self,
        examples: list[GeneratedExample],
        *,
        model_name: str,
    ) -> list[GeneratedExample]:
        for index, example in enumerate(examples):
            example.metadata.setdefault("candidate_index", index)
            example.metadata.setdefault("mlm_model_name", model_name)
            example.metadata.setdefault("mlm_status", "fallback")
            example.metadata.setdefault("mlm_score", None)
            example.metadata.setdefault("mlm_log_prob", None)
            example.metadata["selected"] = index == 0
            example.metadata["mlm_fallback"] = True
        return examples
