from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from exgen.types import ArtifactPaths, RunContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


class ModelConfig(BaseModel):
    """Model-level settings that are resolved once per run."""
    name: str
    model_id: str
    max_tokens: int
    temperature: float = 0.0
    timeout_seconds: int | None = None


class BaseGenerationConfig(BaseModel):
    """Shared generation settings used by both baseline and augmented generators."""
    prompt_version: str
    num_candidates: int


class BaselineGenerationConfig(BaseGenerationConfig):
    """Baseline generation configuration."""
    generator_type: Literal["baseline"] = Field(validation_alias="type")


class AugmentedGenerationConfig(BaseGenerationConfig):
    """Augmented generation configuration with few-shot retrieval settings."""
    generator_type: Literal["augmented"] = Field(validation_alias="type")
    few_shot_count: int


GenerationConfig = BaselineGenerationConfig | AugmentedGenerationConfig


class RetrievalConfig(BaseModel):
    """Retrieval-layer configuration placeholder for future retrieval strategies."""
    name: str = "default"


class BaseEvaluationConfig(BaseModel):
    """Shared evaluation fields used by metric-specific configs."""
    name: str
    prompt_version: str | None = None


class OxfordEvaluationConfig(BaseEvaluationConfig):
    """OxfordEval-specific judge configuration and ordering controls."""
    randomize_order: bool
    judge_model_name: str


class GdexEvaluationConfig(BaseEvaluationConfig):
    """GDEX rubric configuration."""
    criteria: list[str]


class ReadabilityEvaluationConfig(BaseEvaluationConfig):
    """Readability metric configuration."""
    measures: list[str]


EvaluationConfig = OxfordEvaluationConfig | GdexEvaluationConfig | ReadabilityEvaluationConfig


class RunSection(BaseModel):
    """Top-level run identity and seed."""
    run_id: str
    seed: int = 42


class DatasetConfig(BaseModel):
    """Dataset split and language used for the run."""
    split_path: Path
    language: str


class RunVersionsConfig(BaseModel):
    """Version keys that resolve the concrete configs used in the run."""
    model: str
    generator: str
    retrieval: str
    evaluations: list[str]


class RunOutputConfig(BaseModel):
    """Root directory for all artifacts emitted by the run."""
    root_dir: Path


class RunConfigReference(BaseModel):
    """Minimal run config before version-key resolution."""
    run: RunSection
    dataset: DatasetConfig
    versions: RunVersionsConfig
    output: RunOutputConfig


class ResolvedRunConfig(BaseModel):
    """Fully resolved run config with concrete model, generator, retrieval, and evaluation configs."""
    source: Path
    run: RunSection
    dataset: DatasetConfig
    versions: RunVersionsConfig
    output: RunOutputConfig
    model: ModelConfig
    generator: GenerationConfig
    retrieval: RetrievalConfig
    evaluations: list[EvaluationConfig]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def parse_version_key(version_key: str) -> tuple[str, str]:
    """Split a version key like `oxfordeval/v1` into config family and version."""
    parts = version_key.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Invalid version key '{version_key}'. Expected '<name>/<version>'."
        )
    return parts[0], parts[1]


def versioned_config_path(kind: str, version_key: str) -> Path:
    """Map a version key to its canonical config file path."""
    name, version = parse_version_key(version_key)
    directories = {
        "model": CONFIGS_DIR / "models",
        "generator": CONFIGS_DIR / "generator",
        "retrieval": CONFIGS_DIR / "retrieval",
        "evaluation": CONFIGS_DIR / "evaluation" / "approaches",
        "run": CONFIGS_DIR / "runs",
    }
    if kind not in directories:
        raise ValueError(f"Unsupported config kind '{kind}'.")
    return directories[kind] / name / f"{version}.yaml"


@lru_cache(maxsize=None)
def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config once and cache the parsed mapping."""
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {resolved}, got {type(data).__name__}")
    return data


def load_config(path: str | Path, model_type: type[BaseModel]) -> BaseModel:
    """Validate a YAML config into the requested Pydantic model."""
    return model_type.model_validate(load_yaml_config(path))


def load_evaluation_config(version_key: str) -> EvaluationConfig:
    """Resolve and load one metric-specific evaluation config."""
    name, _ = parse_version_key(version_key)
    path = versioned_config_path("evaluation", version_key)
    model_type: type[BaseModel]
    if name == "oxfordeval":
        model_type = OxfordEvaluationConfig
    elif name == "gdex":
        model_type = GdexEvaluationConfig
    elif name == "readability":
        model_type = ReadabilityEvaluationConfig
    else:
        raise ValueError(f"Unsupported evaluation config '{version_key}'.")
    return model_type.model_validate(load_yaml_config(path))


def resolve_run_config_path(path_or_key: str | Path) -> Path:
    """Allow run configs to be referenced either by file path or by version key."""
    candidate = Path(path_or_key)
    if candidate.exists():
        return resolve_project_path(candidate)
    return versioned_config_path("run", str(path_or_key))


def load_run_config(path_or_key: str | Path) -> ResolvedRunConfig:
    """Load a run config reference and resolve every module config it points to."""
    source = resolve_run_config_path(path_or_key)
    reference = RunConfigReference.model_validate(load_yaml_config(source))
    return ResolvedRunConfig(
        source=source,
        run=reference.run,
        dataset=reference.dataset,
        versions=reference.versions,
        output=reference.output,
        model=ModelConfig.model_validate(
            load_yaml_config(versioned_config_path("model", reference.versions.model))
        ),
        generator=_load_generator_config(reference.versions.generator),
        retrieval=RetrievalConfig.model_validate(
            load_yaml_config(versioned_config_path("retrieval", reference.versions.retrieval))
        ),
        evaluations=[
            load_evaluation_config(version_key)
            for version_key in reference.versions.evaluations
        ],
    )


def _load_generator_config(version_key: str) -> GenerationConfig:
    """Resolve the concrete generator config model from the version key."""
    name, _ = parse_version_key(version_key)
    path = versioned_config_path("generator", version_key)
    if name == "baseline":
        return BaselineGenerationConfig.model_validate(load_yaml_config(path))
    if name == "augmented":
        return AugmentedGenerationConfig.model_validate(load_yaml_config(path))
    raise ValueError(f"Unsupported generator config '{version_key}'.")


def build_run_context(config: ResolvedRunConfig) -> RunContext:
    """Derive run-wide artifact paths and shared identifiers from the resolved config."""
    root_dir = resolve_project_path(config.output.root_dir)
    artifacts = ArtifactPaths(
        root_dir=root_dir,
        generation_path=root_dir / "generations.jsonl",
        evaluation_path=root_dir / "evaluations.jsonl",
        manifest_path=root_dir / "manifest.json",
    )
    return RunContext(
        run_id=config.run.run_id,
        seed=config.run.seed,
        dataset_path=resolve_project_path(config.dataset.split_path),
        language=config.dataset.language,
        versions={
            "model": config.versions.model,
            "generator": config.versions.generator,
            "retrieval": config.versions.retrieval,
            "evaluations": list(config.versions.evaluations),
        },
        artifacts=artifacts,
    )
