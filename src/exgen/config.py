from typing import Literal

from pydantic import BaseModel
from pathlib import Path

# class RunSection(BaseModel):
#     run_id: str
#     seed: int = 42


class BaseGenerationConfig(BaseModel):
    num_candidates: int
    prompt_version: str

class BaselineGenerationConfig(BaseGenerationConfig):
    generator_type: Literal["baseline"]


class AugmentedGenerationConfig(BaseGenerationConfig):
    generator_type: Literal["augmented"]
    few_shot_count: int


# class EvaluationSection(BaseModel):
#     metrics: list[str]


# class OutputSection(BaseModel):
#     generation_path: Path | None = None
#     evaluation_dir: Path | None = None
#
#
# class RunConfig(BaseModel):
#     run: RunSection
#     dataset: DatasetSection
#     generation: GenerationSection
#     evaluation: EvaluationSection | None = None
#     output: OutputSection