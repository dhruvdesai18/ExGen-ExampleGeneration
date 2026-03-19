from __future__ import annotations

from exgen.config import ResolvedRunConfig
from exgen.pipeline.run_evaluation import EvaluationPipeline
from exgen.pipeline.run_generation import GenerationPipeline
from exgen.storage import write_manifest
from exgen.types import LexicalItem, RunContext, RunManifest


class FullPipeline:
    def __init__(self) -> None:
        self.generation_pipeline = GenerationPipeline()
        self.evaluation_pipeline = EvaluationPipeline()

    def run(
        self,
        config: ResolvedRunConfig,
        context: RunContext,
        lexical_items: list[LexicalItem],
    ) -> RunManifest:
        manifest = RunManifest(
            run_id=context.run_id,
            pipeline_mode="full",
            dataset=config.dataset.model_dump(mode="json"),
            versions=context.versions,
            configs={
                "model": config.model.model_dump(mode="json"),
                "generator": config.generator.model_dump(mode="json"),
                "retrieval": config.retrieval.model_dump(mode="json"),
                "evaluations": [
                    evaluation.model_dump(mode="json")
                    for evaluation in config.evaluations
                ],
            },
            artifacts=context.artifacts,
        )
        write_manifest(manifest, context.artifacts.manifest_path)
        generation_records = self.generation_pipeline.run(config, context, lexical_items)
        self.evaluation_pipeline.run(config, context, generation_records)
        return manifest
