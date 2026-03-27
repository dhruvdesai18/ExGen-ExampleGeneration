from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from exgen.config import resolve_project_path
from exgen.datasets.loader import (
    load_dataset_frame,
    normalize_dataset_frame,
    stable_lexical_item_id,
)

DEFAULT_SOURCE_DATASET = "data/create_data/final_dataset/sampled_best_3000.csv"
DEFAULT_OUTPUT_DIR = "data/processed"
DEFAULT_EVAL_ITEMS_PER_LANGUAGE = 100
DEFAULT_TEST_ITEMS_PER_LANGUAGE = 100
DEFAULT_CANONICAL_ARTIFACTS = {
    "train_examples": "train_examples.parquet",
    "train_items": "train_items.parquet",
    "eval_examples": "eval_examples.parquet",
    "eval_items": "eval_items.parquet",
    "test_examples": "test_examples.parquet",
    "test_items": "test_items.parquet",
}


@dataclass(frozen=True)
class SplitEntry:
    lexical_item_id: str
    weight: int
    first_index: int
    primary_pos: str
    row_counts: dict[str, Counter[str]]
    item_counts: dict[str, Counter[str]]


def _import_pandas():
    import pandas as pd

    return pd


def ensure_default_split_artifacts(
    *,
    seed: int = 42,
    eval_items_per_language: int = DEFAULT_EVAL_ITEMS_PER_LANGUAGE,
    test_items_per_language: int = DEFAULT_TEST_ITEMS_PER_LANGUAGE,
) -> dict[str, Path]:
    output_dir = resolve_project_path(DEFAULT_OUTPUT_DIR)
    return build_split_artifacts(
        source_path=DEFAULT_SOURCE_DATASET,
        output_dir=output_dir,
        seed=seed,
        eval_items_per_language=eval_items_per_language,
        test_items_per_language=test_items_per_language,
    )


def build_split_artifacts(
    *,
    source_path: str | Path = DEFAULT_SOURCE_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    eval_items_per_language: int = DEFAULT_EVAL_ITEMS_PER_LANGUAGE,
    test_items_per_language: int = DEFAULT_TEST_ITEMS_PER_LANGUAGE,
) -> dict[str, Path]:
    pd = _import_pandas()

    resolved_output_dir = resolve_project_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    frame = normalize_dataset_frame(load_dataset_frame(source_path)).reset_index(drop=True)
    frame["lexical_item_id"] = frame.apply(
        lambda row: stable_lexical_item_id(
            language=str(row["language"]),
            word=str(row["word"]),
            pos=str(row["pos"]),
            definition=str(row["definition"]),
        ),
        axis=1,
    )
    frame["original_row_index"] = frame.index
    frame["freq_quartile"] = _assign_frequency_quartiles(frame)

    eval_ids: set[str] = set()
    test_ids: set[str] = set()
    train_ids: set[str] = set()
    item_frame = _prepare_item_rows(frame)
    for language, language_items in item_frame.groupby("language", sort=True):
        entries = _build_eval_entries(language_items, seed=seed)
        eval_desired = _build_desired_item_counts(entries, eval_items_per_language)
        selected_eval_ids = _select_exact_target(
            entries=entries,
            target_weight=eval_items_per_language,
            desired_counts=eval_desired,
            priorities=("pos", "freq", "source"),
        )

        remaining_entries = [
            entry
            for entry in entries
            if entry.lexical_item_id not in selected_eval_ids
        ]
        test_desired = _build_desired_item_counts(remaining_entries, test_items_per_language)
        selected_test_ids = _select_exact_target(
            entries=remaining_entries,
            target_weight=test_items_per_language,
            desired_counts=test_desired,
            priorities=("pos", "freq", "source"),
        )

        language_ids = {entry.lexical_item_id for entry in entries}
        eval_ids.update(selected_eval_ids)
        test_ids.update(selected_test_ids)
        train_ids.update(language_ids - selected_eval_ids - selected_test_ids)

    train_examples = _prepare_example_rows(frame[frame["lexical_item_id"].isin(train_ids)].copy())
    eval_examples = _prepare_example_rows(frame[frame["lexical_item_id"].isin(eval_ids)].copy())
    test_examples = _prepare_example_rows(frame[frame["lexical_item_id"].isin(test_ids)].copy())
    train_items = _prepare_item_rows(train_examples)
    eval_items = _prepare_item_rows(eval_examples)
    test_items = _prepare_item_rows(test_examples)

    artifact_frames = {
        "train_examples": train_examples,
        "train_items": train_items,
        "eval_examples": eval_examples,
        "eval_items": eval_items,
        "test_examples": test_examples,
        "test_items": test_items,
    }
    artifact_paths = {
        key: resolved_output_dir / name
        for key, name in DEFAULT_CANONICAL_ARTIFACTS.items()
    }
    for key, output_path in artifact_paths.items():
        artifact_frames[key].drop(columns=["original_row_index"], errors="ignore").to_parquet(
            output_path,
            index=False,
        )
    return artifact_paths


def _assign_frequency_quartiles(frame):
    pd = _import_pandas()
    ranked_groups = (
        frame.groupby(["language", "lexical_item_id"], as_index=False)
        .agg(
            word_freq_rank=("word_freq_rank", "first"),
            original_row_index=("original_row_index", "min"),
        )
        .sort_values(["language", "word_freq_rank", "original_row_index", "lexical_item_id"])
    )

    quartiles: dict[tuple[str, str], str] = {}
    for language, language_groups in ranked_groups.groupby("language", sort=False):
        total = len(language_groups)
        for offset, row in enumerate(language_groups.itertuples(index=False)):
            bucket = min(3, (offset * 4) // max(total, 1))
            quartiles[(language, row.lexical_item_id)] = f"q{bucket + 1}"

    return frame.apply(
        lambda row: quartiles[(str(row["language"]), str(row["lexical_item_id"]))],
        axis=1,
    )


def _build_split_entries(language_frame) -> list[SplitEntry]:
    entries: list[SplitEntry] = []
    grouped = language_frame.groupby("lexical_item_id", sort=False)
    pos_totals = language_frame["pos"].value_counts().to_dict()
    for lexical_item_id, group in grouped:
        reference_row = group.sort_values("original_row_index").iloc[0]
        row_counts = {
            "pos": Counter(group["pos"].astype(str)),
            "freq": Counter(group["freq_quartile"].astype(str)),
            "source": Counter(group["source"].astype(str)),
        }
        item_counts = {
            "pos": Counter({str(reference_row["pos"]): 1}),
            "freq": Counter({str(reference_row["freq_quartile"]): 1}),
            "source": Counter({str(reference_row["source"]): 1}),
        }
        entries.append(
            SplitEntry(
                lexical_item_id=str(lexical_item_id),
                weight=int(len(group)),
                first_index=int(group["original_row_index"].min()),
                primary_pos=str(reference_row["pos"]),
                row_counts=row_counts,
                item_counts=item_counts,
            )
        )
    entries.sort(
        key=lambda entry: (
            pos_totals.get(entry.primary_pos, 0),
            entry.weight,
            entry.first_index,
            entry.lexical_item_id,
        )
    )
    return entries


def _build_eval_entries(language_items, *, seed: int) -> list[SplitEntry]:
    pos_totals = language_items["pos"].value_counts().to_dict()
    entries: list[SplitEntry] = []
    for row in language_items.sort_values("original_row_index").to_dict("records"):
        lexical_item_id = str(row["lexical_item_id"])
        row_counts = {
            "pos": Counter({str(row["pos"]): 1}),
            "freq": Counter({str(row["freq_quartile"]): 1}),
            "source": Counter({str(row["source"]): 1}),
        }
        entries.append(
            SplitEntry(
                lexical_item_id=lexical_item_id,
                weight=1,
                first_index=int(row["original_row_index"]),
                primary_pos=str(row["pos"]),
                row_counts=row_counts,
                item_counts=row_counts,
            )
        )
    entries.sort(
        key=lambda entry: (
            pos_totals.get(entry.primary_pos, 0),
            _seeded_order(seed, entry.lexical_item_id),
            entry.first_index,
            entry.lexical_item_id,
        )
    )
    return entries


def _build_desired_row_counts(
    entries: Iterable[SplitEntry],
    target_weight: int,
) -> dict[str, dict[str, float]]:
    totals = defaultdict(Counter)
    total_weight = 0
    for entry in entries:
        total_weight += entry.weight
        for dimension, counts in entry.row_counts.items():
            totals[dimension].update(counts)
    if total_weight < target_weight:
        raise ValueError(
            f"Cannot allocate {target_weight} test rows from only {total_weight} rows."
        )
    ratio = target_weight / total_weight
    desired: dict[str, dict[str, float]] = {}
    for dimension, counts in totals.items():
        desired[dimension] = {
            key: value * ratio
            for key, value in counts.items()
            if dimension != "source" or len(counts) > 1
        }
    return desired


def _build_desired_item_counts(
    entries: Iterable[SplitEntry],
    target_weight: int,
) -> dict[str, dict[str, float]]:
    totals = defaultdict(Counter)
    total_items = 0
    for entry in entries:
        total_items += entry.weight
        for dimension, counts in entry.item_counts.items():
            totals[dimension].update(counts)
    if total_items < target_weight:
        raise ValueError(
            f"Cannot allocate {target_weight} eval items from only {total_items} items."
        )
    ratio = target_weight / total_items
    desired: dict[str, dict[str, float]] = {}
    for dimension, counts in totals.items():
        desired[dimension] = {
            key: value * ratio
            for key, value in counts.items()
            if dimension != "source" or len(counts) > 1
        }
    return desired


def _select_exact_target(
    *,
    entries: list[SplitEntry],
    target_weight: int,
    desired_counts: dict[str, dict[str, float]],
    priorities: tuple[str, ...],
) -> set[str]:
    if target_weight == 0:
        return set()

    weights = [entry.weight for entry in entries]
    reachable = _build_reachable_suffix(weights, target_weight)
    if not _bit_is_set(reachable[0], target_weight):
        raise ValueError(f"Target weight {target_weight} is not reachable from the provided entries.")

    selected_ids: set[str] = set()
    selected_weight = 0
    selected_counts = {dimension: Counter() for dimension in desired_counts}

    for index, entry in enumerate(entries):
        remaining_to_target = target_weight - selected_weight
        can_select = entry.weight <= remaining_to_target and _bit_is_set(
            reachable[index + 1],
            remaining_to_target - entry.weight,
        )
        can_skip = _bit_is_set(reachable[index + 1], remaining_to_target)

        if can_select and not can_skip:
            should_select = True
        elif can_skip and not can_select:
            should_select = False
        elif can_select and can_skip:
            select_score = _score_choice(
                selected_counts=selected_counts,
                entry_counts=entry.row_counts if entry.weight > 1 else entry.item_counts,
                desired_counts=desired_counts,
                priorities=priorities,
                selected_weight=selected_weight + entry.weight,
                target_weight=target_weight,
            )
            skip_score = _score_choice(
                selected_counts=selected_counts,
                entry_counts=None,
                desired_counts=desired_counts,
                priorities=priorities,
                selected_weight=selected_weight,
                target_weight=target_weight,
            )
            should_select = select_score < skip_score
        else:
            raise ValueError("No feasible split assignment remained while building dataset artifacts.")

        if should_select:
            selected_ids.add(entry.lexical_item_id)
            selected_weight += entry.weight
            for dimension, counts in (
                entry.row_counts if entry.weight > 1 else entry.item_counts
            ).items():
                if dimension in selected_counts:
                    selected_counts[dimension].update(counts)

    if selected_weight != target_weight:
        raise ValueError(f"Expected selected weight {target_weight}, got {selected_weight}.")
    return selected_ids


def _score_choice(
    *,
    selected_counts: dict[str, Counter[str]],
    entry_counts: dict[str, Counter[str]] | None,
    desired_counts: dict[str, dict[str, float]],
    priorities: tuple[str, ...],
    selected_weight: int,
    target_weight: int,
) -> tuple[float, ...]:
    score: list[float] = []
    for dimension in priorities:
        desired = desired_counts.get(dimension)
        if not desired:
            continue
        projected = Counter(selected_counts.get(dimension, Counter()))
        if entry_counts is not None and dimension in entry_counts:
            projected.update(entry_counts[dimension])
        keys = set(desired) | set(projected)
        score.append(sum(abs(projected.get(key, 0) - desired.get(key, 0.0)) for key in keys))
    score.append(abs(target_weight - selected_weight))
    return tuple(score)


def _build_reachable_suffix(weights: list[int], target: int) -> list[int]:
    mask = (1 << (target + 1)) - 1
    reachable = [0] * (len(weights) + 1)
    reachable[-1] = 1
    for index in range(len(weights) - 1, -1, -1):
        shifted = (reachable[index + 1] << weights[index]) & mask
        reachable[index] = reachable[index + 1] | shifted
    return reachable


def _bit_is_set(bits: int, index: int) -> bool:
    if index < 0:
        return False
    return ((bits >> index) & 1) == 1


def _seeded_order(seed: int, lexical_item_id: str) -> str:
    payload = f"{seed}:{lexical_item_id}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _prepare_example_rows(frame):
    prepared = frame.sort_values(["language", "original_row_index"]).copy()
    prepared["id"] = prepared["lexical_item_id"]
    return prepared


def _prepare_item_rows(frame):
    pd = _import_pandas()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "lexical_item_id",
                "language",
                "word",
                "pos",
                "definition",
                "source",
                "sense",
                "sentence",
                "sentence_id",
                "reference_sentence",
                "reference_sentence_id",
                "word_freq_rank",
                "sentence_count",
                "freq_quartile",
                "original_row_index",
            ]
        )

    grouped = []
    for lexical_item_id, group in frame.groupby("lexical_item_id", sort=False):
        reference = group.sort_values("original_row_index").iloc[0]
        grouped.append(
            {
                "id": lexical_item_id,
                "lexical_item_id": lexical_item_id,
                "language": reference["language"],
                "word": reference["word"],
                "pos": reference["pos"],
                "definition": reference["definition"],
                "source": reference["source"],
                "sense": reference.get("sense"),
                "sentence": reference["sentence"],
                "sentence_id": reference["sentence_id"],
                "reference_sentence": reference["sentence"],
                "reference_sentence_id": reference["sentence_id"],
                "word_freq_rank": reference["word_freq_rank"],
                "sentence_count": reference["sentence_count"],
                "freq_quartile": reference["freq_quartile"],
                "original_row_index": int(group["original_row_index"].min()),
            }
        )
    return pd.DataFrame(grouped).sort_values(["language", "original_row_index", "word", "pos"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-items-per-language",
        type=int,
        default=DEFAULT_EVAL_ITEMS_PER_LANGUAGE,
    )
    parser.add_argument(
        "--test-items-per-language",
        type=int,
        default=DEFAULT_TEST_ITEMS_PER_LANGUAGE,
    )
    args = parser.parse_args()

    artifact_paths = build_split_artifacts(
        source_path=args.source_path,
        output_dir=args.output_dir,
        seed=args.seed,
        eval_items_per_language=args.eval_items_per_language,
        test_items_per_language=args.test_items_per_language,
    )
    for name, path in artifact_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
