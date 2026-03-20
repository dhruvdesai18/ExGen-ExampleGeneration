import os
import random
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
    prompt_version: str
    few_shot_examples: str

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
                    # FIX: Changed this from "baseline" to "augmented" so tracking works!
                    "generator_type": "augmented", 
                    "prompt_version": self.prompt_version,
                },
            )
            for text in outputs
        ]

    def _build_prompt(self, lexical_item: LexicalItem) -> list[dict[str, str]]:
        item = lexical_item.lexical_item

        # -----------------------------------------------------------------
        # TODO 1: use example Fetcher
        # -----------------------------------------------------------------

        fetched_data = self.fetcher.fetch(item.word)
        
        # Pick up to 3 random examples for variety
        if fetched_data:
            shots = random.sample(fetched_data, min(3, len(fetched_data)))
            formatted_examples = "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(shots)])
        else:
            formatted_examples = "No dictionary examples found for this word."

        # -----------------------------------------------------------------
        # TODO 2: use prompt_version to extract the needed prompt
        # -----------------------------------------------------------------
        system_path = f"configs/prompts/augmented_system_{self.prompt_version}.txt"
        user_path = f"configs/prompts/augmented_user_{self.prompt_version}.txt"

        # Safely load the System Prompt
        try:
            with open(system_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt missing! Please create {system_path}")

        # Safely load the User Prompt template
        try:
            with open(user_path, "r", encoding="utf-8") as f:
                user_template = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"User prompt missing! Please create {user_path}")

        # Inject the actual word data into your text file template
        user_prompt = user_template.format(
            language=item.language,
            word=item.word,
            pos=item.pos,
            definition=item.definition,
            formatted_examples=formatted_examples
        )

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]