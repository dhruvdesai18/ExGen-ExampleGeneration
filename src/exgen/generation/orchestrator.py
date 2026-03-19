from dataclasses import dataclass

from exgen.config import BaseGenerationConfig, BaselineGenerationConfig, AugmentedGenerationConfig
from exgen.generation.augmented import AugmentedGenerator
from exgen.generation.baseline import BaselineGenerator
from exgen.models.base import Model
from exgen.retrieval.fetcher import Fetcher
from exgen.types import GeneratedExample


@dataclass
class Orchestrator:
    model: Model
    fetcher: Fetcher

    def generate(self, config: BaseGenerationConfig, lexical_item) -> list[GeneratedExample]:
        if isinstance(config, BaselineGenerationConfig):
            generator = BaselineGenerator(
                model=self.model,
                num_candidates=config.num_candidates,
                prompt_version=config.prompt_version,
            )
        elif isinstance(config, AugmentedGenerationConfig):
            generator = AugmentedGenerator(
                model=self.model,
                fetcher=self.fetcher,
                num_candidates=config.num_candidates,
                prompt_version=config.prompt_version,
                few_shot_examples=str(config.few_shot_count),
            )
        else:
            raise ValueError(f"Unsupported generation config: {type(config).__name__}")
        return generator.generate(lexical_item)
