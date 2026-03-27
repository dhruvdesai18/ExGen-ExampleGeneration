from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from exgen.config import resolve_project_path
from exgen.models.base import Model
from exgen.retrieval.fetcher import Fetcher
from exgen.types import GeneratedExample, LexicalItem


@dataclass
class AugmentedGenerator:
    model: Model
    fetcher: Fetcher
    num_candidates: int
    prompt_version: str
    few_shot_examples: int

    def generate(self, lexical_item: LexicalItem) -> list[GeneratedExample]:
        messages = self._build_prompt(lexical_item)
        outputs = self.model.generate(messages, num_candidates=self.num_candidates)
        return [
            GeneratedExample(
                id=str(uuid4()),
                lexical_item=lexical_item,
                example_text=text,
                metadata={
                    "model_name": self.model.name,
                    "generator_type": "augmented",
                    "prompt_version": self.prompt_version,
                },
            )
            for text in outputs
        ]

    def _build_prompt(self, lexical_item: LexicalItem) -> list[dict[str, str]]:
        few_shot_rows = self.fetcher.fetch(lexical_item, limit=self.few_shot_examples)
        if few_shot_rows:
            formatted_examples = "\n".join(
                f"{index + 1}. {example.example_text}"
                for index, example in enumerate(few_shot_rows)
            )
        else:
            formatted_examples = "No reference examples available."

        system_prompt, user_template = self._load_prompt_pair()
        user_prompt = user_template.format(
            language=lexical_item.language,
            word=lexical_item.word,
            pos=lexical_item.pos,
            definition=lexical_item.definition,
            formatted_examples=formatted_examples,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _load_prompt_pair(self) -> tuple[str, str]:
        system_path, user_path = self._resolve_prompt_paths()
        return (
            Path(system_path).read_text(encoding="utf-8"),
            Path(user_path).read_text(encoding="utf-8"),
        )

    def _resolve_prompt_paths(self) -> tuple[Path, Path]:
        if self.prompt_version.startswith("augmented_"):
            suffix = self.prompt_version.removeprefix("augmented_")
            system_path = resolve_project_path(f"configs/prompts/augmented_system_{suffix}.txt")
            user_path = resolve_project_path(f"configs/prompts/augmented_user_{suffix}.txt")
            if system_path.exists() and user_path.exists():
                return system_path, user_path

        system_path = resolve_project_path(f"configs/prompts/{self.prompt_version}_system.txt")
        user_path = resolve_project_path(f"configs/prompts/{self.prompt_version}_user.txt")
        if system_path.exists() and user_path.exists():
            return system_path, user_path

        raise FileNotFoundError(
            f"Unable to resolve prompts for version '{self.prompt_version}'."
        )
