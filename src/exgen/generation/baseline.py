from dataclasses import dataclass
from uuid import uuid4

from src.exgen.models.base import Model
from src.exgen.types import GeneratedExample, LexicalItem


@dataclass
class BaselineGenerator:
    model: Model
    num_candidates: int
    prompt_version: str

    def generate(self, lexical_item: LexicalItem) -> list[GeneratedExample]:
        messages = self._build_prompt(lexical_item)

        outputs = self.model.generate(
            messages,
            num_candidates=self.num_candidates,
        )

        return [
            GeneratedExample(
                id=str(uuid4()),
                lexical_item=lexical_item,
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
