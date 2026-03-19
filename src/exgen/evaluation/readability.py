from __future__ import annotations

import re

from exgen.config import ReadabilityEvaluationConfig
from exgen.evaluation.base import Evaluator
from exgen.types import EvaluationOutcome, GeneratedExample

WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*", re.UNICODE)
VOWEL_GROUP_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")


class ReadabilityEvaluator(Evaluator):
    metric_name = "readability"
    FKGL_LANGUAGES = {"english"}

    def __init__(self, config: ReadabilityEvaluationConfig) -> None:
        self.config = config

    def evaluate(self, example: GeneratedExample) -> EvaluationOutcome:
        """Compute deterministic readability metrics for one generated sentence."""
        text = example.example_text.strip()
        language = example.lexical_item.language.strip().lower()
        words = self._tokenize_words(text)
        sentence_count = self._count_sentences(text)
        word_count = len(words)
        syllable_count = sum(self._count_word_syllables(word) for word in words)
        fkgl = None
        if self._supports_fkgl(language) and "fkgl" in self.config.measures:
            fkgl = self._compute_fkgl(
                word_count=word_count,
                sentence_count=sentence_count,
                syllable_count=syllable_count,
            )

        result = {
            "language": language,
            "sentence_count": sentence_count,
            "word_count": word_count,
            "syllable_count": syllable_count,
        }
        if fkgl is not None:
            result["fkgl"] = fkgl

        score = fkgl if "fkgl" in self.config.measures else None

        return EvaluationOutcome(
            score=score,
            result=result,
            metadata={
                "fkgl_supported": self._supports_fkgl(language),
                "requested_measures": list(self.config.measures),
            },
            provenance={
                "metric_name": self.metric_name,
                "metric_config_name": self.config.name,
                "metric_prompt_version": self.config.prompt_version,
                "implementation": "deterministic-local",
            },
        )

    def _supports_fkgl(self, language: str) -> bool:
        """Return whether the configured text language is suitable for FKGL."""
        return language in self.FKGL_LANGUAGES

    def _compute_fkgl(
        self,
        *,
        word_count: int,
        sentence_count: int,
        syllable_count: int,
    ) -> float | None:
        """Calculate FKGL from aggregate counts, or return None for empty text."""
        if word_count == 0 or sentence_count == 0:
            return None
        return round(
            0.39 * (word_count / sentence_count)
            + 11.8 * (syllable_count / word_count)
            - 15.59,
            4,
        )

    def _count_sentences(self, text: str) -> int:
        """Estimate the number of sentences from terminal punctuation."""
        if not text:
            return 0
        segments = [segment for segment in SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]
        return len(segments) if segments else 1

    def _tokenize_words(self, text: str) -> list[str]:
        """Extract alphabetic word tokens for readability statistics."""
        return WORD_PATTERN.findall(text)

    def _count_word_syllables(self, word: str) -> int:
        """Estimate syllables with a simple English-oriented vowel-group heuristic."""
        normalized = word.lower()
        if not normalized:
            return 0

        groups = VOWEL_GROUP_PATTERN.findall(normalized)
        syllables = len(groups)

        if normalized.endswith("e") and not normalized.endswith(("le", "ye")) and syllables > 1:
            syllables -= 1

        return max(syllables, 1)
