from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from exgen.config import (
    ResolvedRunConfig,
    build_run_context,
    load_run_config,
    resolve_project_path,
)
from exgen.datasets.loader import (
    load_dataset_frame,
    normalize_dataset_frame,
    stable_lexical_item_id,
)
from exgen.pipeline.factories import create_evaluator
from exgen.pipeline.records import build_evaluation_record, generation_record_to_example
from exgen.storage import (
    load_generation_records,
    persist_evaluation_record,
    reset_output_file,
)
from exgen.types import EvaluationRecord, GenerationRecord, LexicalItemSnapshot, RunContext

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for stale environments
    tqdm = None


class EvaluationPipeline:
    def run(
        self,
        config: ResolvedRunConfig,
        context: RunContext,
        generation_records: list[GenerationRecord] | None = None,
        generation_path: str | Path | None = None,
        max_items: int | None = None,
    ) -> list[EvaluationRecord]:
        """Evaluate generated examples and persist normalized result rows."""
        source_records = generation_records or self._load_source_records(
            context,
            generation_path=generation_path,
        )
        selected_records = [
            record
            for record in source_records
            if record.metadata.get("selected") is True
        ]
        if max_items is not None:
            selected_records = _limit_records_per_language(selected_records, max_items)
        reset_output_file(context.artifacts.evaluation_path)
        evaluators = [
            (evaluation_config, create_evaluator(evaluation_config))
            for evaluation_config in config.evaluations
        ]
        total_evaluations = len(selected_records) * len(evaluators)
        progress = _progress(
            total=total_evaluations,
            desc=f"Evaluating {len(selected_records)} items x {len(evaluators)} metrics = {total_evaluations}",
        )

        records: list[EvaluationRecord] = []
        for generation_record in selected_records:
            example = generation_record_to_example(generation_record)
            for evaluation_config, evaluator in evaluators:
                _set_progress_status(
                    progress,
                    word=example.lexical_item.word,
                    metric=evaluation_config.name,
                )
                outcome = evaluator.evaluate(example)
                record = build_evaluation_record(
                    context=context,
                    evaluation_config=evaluation_config,
                    generation_record=generation_record,
                    outcome=outcome,
                )
                persist_evaluation_record(record, context.artifacts.evaluation_path)
                records.append(record)
                _advance_progress(progress)
        _close_progress(progress)
        return records

    @staticmethod
    def _load_source_records(
        context: RunContext,
        generation_path: str | Path | None = None,
    ) -> list[GenerationRecord]:
        if generation_path is not None:
            return _load_generation_records_for_evaluation(generation_path, context)
        return _load_generation_records_for_evaluation(context.artifacts.generation_path, context)


def _load_generation_records_for_evaluation(
    generation_path: str | Path,
    context: RunContext,
) -> list[GenerationRecord]:
    path = Path(generation_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Generation file not found: {path}")

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return []

    first_row = rows[0]
    if _looks_like_generation_record(first_row):
        return [GenerationRecord.model_validate(row) for row in rows]
    if _looks_like_flat_generation_row(first_row):
        return _adapt_flat_generation_rows(rows, context, source_path=path)
    raise ValueError(
        "Unsupported generation row format. Expected GenerationRecord rows or flat rows "
        "with language/word/pos/definition/example."
    )


def _looks_like_generation_record(row: dict[str, Any]) -> bool:
    return {"generated_example_id", "lexical_item", "example_text"} <= set(row)


def _looks_like_flat_generation_row(row: dict[str, Any]) -> bool:
    return {"language", "word", "pos", "definition", "example"} <= set(row)


def _adapt_flat_generation_rows(
    rows: list[dict[str, Any]],
    context: RunContext,
    *,
    source_path: Path,
) -> list[GenerationRecord]:
    frame = normalize_dataset_frame(load_dataset_frame(context.dataset_path))
    records_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in frame.reset_index(drop=True).to_dict("records"):
        key = _lexical_key(item)
        records_by_key.setdefault(key, item)

    adapted: list[GenerationRecord] = []
    missing_keys: list[tuple[str, str, str, str]] = []
    for row in rows:
        key = _lexical_key(row)
        matched = records_by_key.get(key)
        if matched is None:
            missing_keys.append(key)
            continue

        lexical_item_id = str(
            matched.get("id")
            or stable_lexical_item_id(
                language=str(matched.get("language", "")),
                word=str(matched.get("word", "")),
                pos=str(matched.get("pos", "")),
                definition=str(matched.get("definition", "")),
            )
        )
        adapted.append(
            GenerationRecord(
                run_id=context.run_id,
                generated_example_id=f"adapted-{uuid.uuid4()}",
                lexical_item_id=lexical_item_id,
                lexical_item=LexicalItemSnapshot(
                    id=lexical_item_id,
                    language=str(matched.get("language", "")),
                    word=str(matched.get("word", "")),
                    pos=str(matched.get("pos", "")),
                    definition=str(matched.get("definition", "")),
                    source=str(matched.get("source", "")),
                    sense=_coerce_optional_text(matched.get("sense")),
                    reference_sentence=_coerce_optional_text(
                        matched.get("reference_sentence", matched.get("sentence"))
                    ),
                    reference_sentence_id=_coerce_optional_text(
                        matched.get("reference_sentence_id", matched.get("sentence_id"))
                    ),
                    word_freq_rank=_coerce_optional_int(matched.get("word_freq_rank")),
                    sentence_count=_coerce_optional_int(matched.get("sentence_count")),
                ),
                example_text=str(row["example"]),
                metadata={
                    "selected": True,
                    "adapter_source": "flat_jsonl",
                    "adapter_input_path": str(source_path),
                },
            )
        )

    if missing_keys:
        preview = ", ".join(
            f"{language}/{word}/{pos}"
            for language, word, pos, _definition in missing_keys[:5]
        )
        hint = _suggest_matching_dataset_path(rows, context.dataset_path)
        raise ValueError(
            f"Could not match {len(missing_keys)} generated rows to dataset split items. "
            f"First unmatched keys: {preview}{hint}"
        )
    return adapted


def _lexical_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("language", "")).strip(),
        str(row.get("word", "")).strip(),
        str(row.get("pos", "")).strip(),
        str(row.get("definition", "")).strip(),
    )


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _suggest_matching_dataset_path(
    rows: list[dict[str, Any]],
    configured_dataset_path: Path,
) -> str:
    row_keys = {_lexical_key(row) for row in rows}
    candidates = [
        configured_dataset_path,
        configured_dataset_path.with_name("test_items.parquet"),
        configured_dataset_path.with_name("eval_items.parquet"),
        configured_dataset_path.with_name("train_items.parquet"),
    ]
    seen: set[Path] = set()
    best_path: Path | None = None
    best_overlap = 0
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            frame = normalize_dataset_frame(load_dataset_frame(candidate))
        except Exception:
            continue
        candidate_keys = {
            _lexical_key(record)
            for record in frame.reset_index(drop=True).to_dict("records")
        }
        overlap = len(row_keys & candidate_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_path = candidate

    if best_path and best_overlap == len(row_keys):
        return (
            f". The configured dataset is {configured_dataset_path}, but all generated rows "
            f"match {best_path}. Re-run with --dataset-path {best_path}"
        )
    return ""


def run_evaluation(
    run_config: str,
    generation_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    max_items: int | None = None,
) -> None:
    """Load the run config, execute evaluation, and report the output location."""
    config = load_run_config(run_config)
    context = build_run_context(config)
    if dataset_path is not None:
        context = context.model_copy(
            update={"dataset_path": resolve_project_path(dataset_path)}
        )
    if max_items is None:
        evaluation_runtime = getattr(config, "evaluation", None)
        max_items = getattr(evaluation_runtime, "max_items_per_language", None)
    records = EvaluationPipeline().run(
        config,
        context,
        generation_path=generation_path,
        max_items=max_items,
    )
    print(f"Evaluated {len(records)} records into {context.artifacts.evaluation_path}")


def main() -> None:
    """Parse command-line arguments for the standalone evaluation entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-config",
        help="Run config version key or path.",
    )
    parser.add_argument(
        "--generation-path",
        help="Optional generation JSONL path. Supports native GenerationRecord rows or flat inference rows.",
    )
    parser.add_argument(
        "--dataset-path",
        help="Optional dataset split path override used to join flat generation rows to reference items.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Optional maximum number of lexical items to evaluate per language after loading selected records.",
    )
    args = parser.parse_args()
    run_evaluation(
        args.run_config,
        generation_path=args.generation_path,
        dataset_path=args.dataset_path,
        max_items=args.max_items,
    )


def _progress(*, total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=desc,
        disable=not sys.stderr.isatty(),
        unit="eval",
    )


def _set_progress_status(progress, *, word: str, metric: str) -> None:
    if progress is None:
        return
    progress.set_postfix_str(f"metric={metric} word={word}", refresh=False)


def _advance_progress(progress) -> None:
    if progress is None:
        return
    progress.update(1)


def _close_progress(progress) -> None:
    if progress is None:
        return
    progress.close()


def _limit_records_per_language(
    records: list[GenerationRecord],
    limit: int,
) -> list[GenerationRecord]:
    kept: list[GenerationRecord] = []
    counts: dict[str, int] = {}
    for record in records:
        language = record.lexical_item.language
        current = counts.get(language, 0)
        if current >= limit:
            continue
        kept.append(record)
        counts[language] = current + 1
    return kept


if __name__ == "__main__":
    main()
