from __future__ import annotations

import argparse

from exgen.config import ResolvedRunConfig, build_run_context, load_run_config
from exgen.datasets.loader import load_lexical_items
from exgen.pipeline.factories import create_fetcher, create_generator, create_model
from exgen.pipeline.records import build_generation_record
from exgen.storage import persist_generation_record, reset_output_file
from exgen.types import GenerationRecord, LexicalItem, RunContext


class GenerationPipeline:
    def run(
        self,
        config: ResolvedRunConfig,
        context: RunContext,
        lexical_items: list[LexicalItem],
    ) -> list[GenerationRecord]:
        """Generate example sentences and persist the raw generation rows."""
        model = create_model(config.model)
        fetcher = create_fetcher(config.retrieval)
        generator = create_generator(config.generator, model, fetcher)
        reset_output_file(context.artifacts.generation_path)

        records: list[GenerationRecord] = []
        for lexical_item in lexical_items:
            generated_examples = generator.generate(lexical_item)
            for example in generated_examples:
                record = build_generation_record(context, example)
                persist_generation_record(record, context.artifacts.generation_path)
                records.append(record)
        return records


def run_generation(run_config: str) -> None:
    """Load the run config, execute generation, and report the output location."""
    config = load_run_config(run_config)
    context = build_run_context(config)
    lexical_items = load_lexical_items(context.dataset_path)
    records = GenerationPipeline().run(config, context, lexical_items)
    print(f"Generated {len(records)} records into {context.artifacts.generation_path}")


def main() -> None:
    """Parse command-line arguments for the standalone generation entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-config",
        help="Run config version key or path.",
    )
    args = parser.parse_args()
    run_generation(args.run_config)


if __name__ == "__main__":
    main()
