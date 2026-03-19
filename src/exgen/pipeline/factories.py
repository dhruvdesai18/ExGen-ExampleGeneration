from __future__ import annotations

from exgen.config import (
    AugmentedGenerationConfig,
    BaselineGenerationConfig,
    EvaluationConfig,
    GdexEvaluationConfig,
    ModelConfig,
    OxfordEvaluationConfig,
    ReadabilityEvaluationConfig,
    RetrievalConfig,
)
from exgen.evaluation.base import Evaluator
from exgen.evaluation.oxfordeval import OxfordEvalEvaluator
from exgen.generation.augmented import AugmentedGenerator
from exgen.generation.baseline import BaselineGenerator
from exgen.models.base import Model
from exgen.models.gpt import GPT
from exgen.models.ministral_qft import MinistralQFT
from exgen.retrieval.fetcher import Fetcher


def create_model(config: ModelConfig) -> Model:
    if config.name == "gpt":
        return GPT(
            name=config.name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            model_id=config.model_id,
            timeout_seconds=config.timeout_seconds,
        )
    if config.name == "ministral_qft":
        return MinistralQFT(
            name=config.name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            model_id=config.model_id,
        )
    raise ValueError(f"Unsupported model config '{config.name}'.")


def create_fetcher(config: RetrievalConfig) -> Fetcher:
    _ = config
    return Fetcher()


def create_generator(config, model: Model, fetcher: Fetcher):
    if isinstance(config, BaselineGenerationConfig):
        return BaselineGenerator(
            model=model,
            num_candidates=config.num_candidates,
            prompt_version=config.prompt_version,
        )
    if isinstance(config, AugmentedGenerationConfig):
        return AugmentedGenerator(
            model=model,
            fetcher=fetcher,
            num_candidates=config.num_candidates,
            prompt_version=config.prompt_version,
            few_shot_examples=str(config.few_shot_count),
        )
    raise ValueError(f"Unsupported generation config '{type(config).__name__}'.")


def create_evaluator(config: EvaluationConfig) -> Evaluator:
    if isinstance(config, OxfordEvaluationConfig):
        return OxfordEvalEvaluator(config)
    if isinstance(config, GdexEvaluationConfig):
        raise NotImplementedError("GDEX evaluator is not implemented yet.")
    if isinstance(config, ReadabilityEvaluationConfig):
        raise NotImplementedError("Readability evaluator is not implemented yet.")
    raise ValueError(f"Unsupported evaluation config '{type(config).__name__}'.")
