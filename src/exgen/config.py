from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
T = TypeVar("T", bound=BaseModel)


# class RunSection(BaseModel):
#     run_id: str
#     seed: int = 42


class BaseGenerationConfig(BaseModel):
    num_candidates: int
    prompt_version: str


class BaselineGenerationConfig(BaseGenerationConfig):
    generator_type: Literal["baseline"] = Field(validation_alias="type")


class AugmentedGenerationConfig(BaseGenerationConfig):
    generator_type: Literal["augmented"] = Field(validation_alias="type")
    few_shot_count: int


class BaseEvaluationConfig(BaseModel):
    name: str
    prompt_version: str


class OxfordEvaluationConfig(BaseEvaluationConfig):
    randomize_order: bool


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

def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


@lru_cache(maxsize=None)
def load_yaml_config(path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {resolved}, got {type(data).__name__}")
    return data


def load_config(path: str | Path, model_type: type[T]) -> T:
    return model_type.model_validate(load_yaml_config(path))


def load_oxford_evaluation_config(
        path: str | Path = "configs/evaluation/oxfordeval.yaml",
) -> OxfordEvaluationConfig:
    return load_config(path, OxfordEvaluationConfig)
