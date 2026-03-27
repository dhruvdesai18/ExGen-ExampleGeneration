from __future__ import annotations

import argparse

from exgen.config import ResolvedRunConfig, build_run_context, load_run_config
from exgen.pipeline.factories import create_evaluator
from exgen.pipeline.records import build_evaluation_record, generation_record_to_example
from exgen.storage import (
    load_generation_records,
    persist_evaluation_record,
    reset_output_file,
)
from exgen.types import EvaluationRecord, GenerationRecord, RunContext


class EvaluationPipeline:
    def run(
        self,
        config: ResolvedRunConfig,
        context: RunContext,
        generation_records: list[GenerationRecord] | None = None,
    ) -> list[EvaluationRecord]:
        """Evaluate generated examples and persist normalized result rows."""
        source_records = generation_records or self._load_source_records(context)
        selected_records = [
            record
            for record in source_records
            if record.metadata.get("selected") is True
        ]
        reset_output_file(context.artifacts.evaluation_path)
        evaluators = [
            (evaluation_config, create_evaluator(evaluation_config))
            for evaluation_config in config.evaluations
        ]

        records: list[EvaluationRecord] = []
        for generation_record in selected_records:
            example = generation_record_to_example(generation_record)
            for evaluation_config, evaluator in evaluators:
                outcome = evaluator.evaluate(example)
                record = build_evaluation_record(
                    context=context,
                    evaluation_config=evaluation_config,
                    generation_record=generation_record,
                    outcome=outcome,
                )
                persist_evaluation_record(record, context.artifacts.evaluation_path)
                records.append(record)
        return records

    @staticmethod
    def _load_source_records(context: RunContext) -> list[GenerationRecord]:
        selected_records = load_generation_records(context.artifacts.selected_generation_path)
        if selected_records:
            return selected_records
        return load_generation_records(context.artifacts.generation_path)


def run_evaluation(run_config: str) -> None:
    """Load the run config, execute evaluation, and report the output location."""
    config = load_run_config(run_config)
    context = build_run_context(config)
    records = EvaluationPipeline().run(config, context)
    print(f"Evaluated {len(records)} records into {context.artifacts.evaluation_path}")


def main() -> None:
    """Parse command-line arguments for the standalone evaluation entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-config",
        help="Run config version key or path.",
    )
    args = parser.parse_args()
    run_evaluation(args.run_config)


if __name__ == "__main__":
    main()
