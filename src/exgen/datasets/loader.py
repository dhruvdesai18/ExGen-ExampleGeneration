from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from exgen.config import resolve_project_path
from exgen.types import LexicalItem

CANONICAL_COLUMNS = (
    "word",
    "sentence",
    "source",
    "pos",
    "sense",
    "sentence_id",
    "word_freq_rank",
    "sentence_count",
    "language",
    "definition",
)
REQUIRED_COLUMNS = {"language", "word", "pos", "definition"}


def _import_pandas():
    import pandas as pd

    return pd


def stable_lexical_item_id(
    *,
    language: str,
    word: str,
    pos: str,
    definition: str,
) -> str:
    payload = "||".join((language, word, pos, definition))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def normalize_dataset_columns(columns: list[str] | Any) -> list[str]:
    normalized: list[str] = []
    for column in columns:
        text = str(column).strip()
        if text == "\ufeffword":
            text = "word"
        normalized.append(text)
    return normalized


def load_dataset_frame(path: str | Path):
    pd = _import_pandas()
    resolved = resolve_project_path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(resolved)
    elif suffix == ".csv":
        frame = pd.read_csv(resolved, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported dataset format '{resolved.suffix}' for {resolved}.")
    frame.columns = normalize_dataset_columns(list(frame.columns))
    return frame


def normalize_dataset_frame(frame):
    pd = _import_pandas()
    normalized = frame.copy()
    normalized.columns = normalize_dataset_columns(list(normalized.columns))

    if "id" not in normalized.columns and "lexical_item_id" in normalized.columns:
        normalized["id"] = normalized["lexical_item_id"]

    if "reference_sentence" not in normalized.columns and "sentence" in normalized.columns:
        normalized["reference_sentence"] = normalized["sentence"]
    if "reference_sentence_id" not in normalized.columns and "sentence_id" in normalized.columns:
        normalized["reference_sentence_id"] = normalized["sentence_id"]

    missing = REQUIRED_COLUMNS - set(normalized.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if "source" not in normalized.columns:
        normalized["source"] = ""

    if "id" not in normalized.columns:
        normalized["id"] = normalized.apply(
            lambda row: stable_lexical_item_id(
                language=_coerce_text(row.get("language")),
                word=_coerce_text(row.get("word")),
                pos=_coerce_text(row.get("pos")),
                definition=_coerce_text(row.get("definition")),
            ),
            axis=1,
        )

    for column in ("reference_sentence", "reference_sentence_id", "sense"):
        if column not in normalized.columns:
            normalized[column] = None

    for column in ("word_freq_rank", "sentence_count"):
        if column not in normalized.columns:
            normalized[column] = None
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized


def load_lexical_items(path: str | Path) -> list[LexicalItem]:
    frame = normalize_dataset_frame(load_dataset_frame(path))
    items: list[LexicalItem] = []
    for row in frame.reset_index(drop=True).to_dict("records"):
        items.append(
            LexicalItem(
                id=_coerce_text(row.get("id")),
                language=_coerce_text(row.get("language")),
                word=_coerce_text(row.get("word")),
                pos=_coerce_text(row.get("pos")),
                definition=_coerce_text(row.get("definition")),
                source=_coerce_text(row.get("source")),
                sense=_coerce_optional_text(row.get("sense")),
                reference_sentence=_coerce_optional_text(
                    row.get("reference_sentence", row.get("sentence"))
                ),
                reference_sentence_id=_coerce_optional_text(
                    row.get("reference_sentence_id", row.get("sentence_id"))
                ),
                word_freq_rank=_coerce_optional_int(row.get("word_freq_rank")),
                sentence_count=_coerce_optional_int(row.get("sentence_count")),
            )
        )
    return items


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _coerce_optional_text(value: Any) -> str | None:
    text = _coerce_text(value).strip()
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
