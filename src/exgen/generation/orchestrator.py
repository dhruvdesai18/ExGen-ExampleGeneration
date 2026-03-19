from dataclasses import dataclass

from exgen.config import BaseGenerationConfig, BaselineGenerationConfig, AugmentedGenerationConfig
from exgen.generation.augmented import AugmentedGenerator
from exgen.generation.baseline import BaselineGenerator
from exgen.models.base import Model
from exgen.retrieval.fetcher import Fetcher
from exgen.types import GeneratedExample


@dataclass
class Orchestrator:
    run_id: str

    model: Model
    fetcher: Fetcher

    def generate(self, config: BaseGenerationConfig) -> GeneratedExample:
        if isinstance(config.generation, BaselineGenerationConfig):
            generator = BaselineGenerator(
                model=self.model,
                num_candidates=config.generation.num_candidates,
                prompt_version=config.generation.prompt_version,
            )
        elif isinstance(config.generation, AugmentedGenerationConfig):
            generator = AugmentedGenerator(
                model=self.model,
                fetcher=self.fetcher,
                num_candidates=config.generation.num_candidates,
                prompt_version=config.generation.prompt_version,
                few_shot_examples=config.generation.few_shot_examples,
            )

        ...
