from dataclasses import dataclass
from uuid import uuid4

from src.exgen.models.base import Model
from src.exgen.types import GeneratedExample, LexicalItem
from src.exgen.retrieval.fetcher import Fetcher


@dataclass
class AugmentedGenerator:
    model: Model
    fetcher: Fetcher

    num_candidates: int
    temperature: float
    max_tokens: int
    prompt_version: str

    def generate(self, lexical_item: LexicalItem) -> list[GeneratedExample]:
        messages = self._build_prompt(lexical_item)

        outputs = self.model.generate(
            messages,
            num_candidates=self.num_candidates,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return [
            GeneratedExample(
                id=str(uuid4()),
                lexical_item_id=lexical_item.item_id,
                example_text=text,
                metadata={
                    "model_name": self.model.name,
                    "generator_type": "baseline",
                },
            )
            for text in outputs
        ]

    def _build_prompt(self, lexical_item: LexicalItem) -> list[dict[str, str]]:
        # TODO: use prompt_version to extract the needed prompt
        # TODO: use example Fetcher

        item = lexical_item.lexical_item
        return [
            {
                "role": "system",
                "content": "Generate a clear dictionary example sentence.",
            },
            {
                "role": "user",
                "content": (
                    f"Language: {item.language}\n"
                    f"Word: {item.word}\n"
                    f"Part of speech: {item.pos}\n"
                    f"Definition: {item.definition}\n"
                    "Generate one natural sentence."
                ),
            },
        ]
