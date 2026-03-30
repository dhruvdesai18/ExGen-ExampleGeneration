from __future__ import annotations

import argparse
import sys

from exgen.config import ResolvedRunConfig, build_run_context, load_run_config
from exgen.datasets.loader import load_lexical_items
from exgen.datasets.splits import ensure_default_split_artifacts
from exgen.generation.ranking import MLMCandidateRanker
from exgen.pipeline.factories import create_fetcher, create_generator, create_model
from exgen.pipeline.records import build_generation_record
from exgen.storage import persist_generation_record, reset_output_file
from exgen.types import GenerationRecord, LexicalItem, RunContext

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for stale environments
    tqdm = None


class GenerationPipeline:
    def __init__(self) -> None:
        self.ranker = MLMCandidateRanker()

    def run(
        self,
        config: ResolvedRunConfig,
        context: RunContext,
        lexical_items: list[LexicalItem],
    ) -> list[GenerationRecord]:
        """Generate example sentences and persist the raw generation rows."""
        model = create_model(config.model)
        fetcher = create_fetcher(config.retrieval, context)
        generator = create_generator(config.generator, model, fetcher)
        reset_output_file(context.artifacts.generation_path)
        reset_output_file(context.artifacts.selected_generation_path)

        records: list[GenerationRecord] = []
        progress = _progress(
            total=len(lexical_items),
            desc=f"Generating {len(lexical_items)} items",
        )
        try:
            for lexical_item in lexical_items:
                _set_progress_status(progress, word=lexical_item.word)
                generated_examples = generator.generate(lexical_item)
                ranked_examples = self.ranker.annotate(lexical_item, generated_examples)
                for example in ranked_examples:
                    record = build_generation_record(context, example)
                    persist_generation_record(record, context.artifacts.generation_path)
                    if example.metadata.get("selected") is True:
                        persist_generation_record(
                            record,
                            context.artifacts.selected_generation_path,
                        )
                    records.append(record)
                _advance_progress(progress)
        finally:
            _close_progress(progress)
        return records


def run_generation(run_config: str) -> None:
    """Load the run config, execute generation, and report the output location."""
    config = load_run_config(run_config)
    context = build_run_context(config)
    ensure_default_split_artifacts(seed=context.seed)
    lexical_items = load_lexical_items(context.dataset_path)
    records = GenerationPipeline().run(config, context, lexical_items)
    print(f"Generated {len(records)} records into {context.artifacts.generation_path}")
    selected_count = sum(1 for record in records if record.metadata.get("selected") is True)
    print(
        f"Selected {selected_count} records into {context.artifacts.selected_generation_path}"
    )


def _progress(*, total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=desc,
        disable=not sys.stderr.isatty(),
        unit="item",
    )


def _set_progress_status(progress, *, word: str) -> None:
    if progress is None:
        return
    progress.set_postfix_str(f"word={word}", refresh=False)


def _advance_progress(progress) -> None:
    if progress is None:
        return
    progress.update(1)


def _close_progress(progress) -> None:
    if progress is None:
        return
    progress.close()


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
