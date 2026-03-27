from __future__ import annotations

from pathlib import Path

from exgen.config import resolve_project_path
from exgen.datasets.loader import load_dataset_frame, normalize_dataset_frame
from exgen.datasets.splits import ensure_default_split_artifacts
from exgen.types import LexicalItem, RetrievedExample


class Fetcher:
    def __init__(self, dataset_path: str | Path) -> None:
        resolved = resolve_project_path(dataset_path)
        if not resolved.exists():
            ensure_default_split_artifacts()
        self.dataset_path = resolved
        self.frame = normalize_dataset_frame(load_dataset_frame(resolved))
        self.frame["word_freq_rank"] = self.frame["word_freq_rank"].fillna(10**9)
        self.frame["sentence_count"] = self.frame["sentence_count"].fillna(0)

    def fetch(self, lexical_item: LexicalItem, limit: int = 3) -> list[RetrievedExample]:
        candidates = self.frame[self.frame["language"] == lexical_item.language].copy()
        if candidates.empty:
            return []

        lexical_key = (
            lexical_item.language,
            lexical_item.word,
            lexical_item.pos,
            lexical_item.definition,
        )
        candidate_keys = list(
            zip(
                candidates["language"],
                candidates["word"],
                candidates["pos"],
                candidates["definition"],
            )
        )
        candidates = candidates[[key != lexical_key for key in candidate_keys]].copy()
        if candidates.empty:
            return []

        target_rank = lexical_item.word_freq_rank or 10**9
        candidates["same_pos"] = (candidates["pos"] == lexical_item.pos).astype(int)
        candidates["freq_distance"] = (candidates["word_freq_rank"] - target_rank).abs()
        candidates = candidates.sort_values(
            by=[
                "same_pos",
                "freq_distance",
                "sentence_count",
                "sentence_id",
            ],
            ascending=[False, True, False, True],
        )
        candidates = candidates.drop_duplicates(subset=["lexical_item_id"], keep="first")

        rows = candidates.head(limit).to_dict("records")
        return [
            RetrievedExample(
                id=str(row.get("sentence_id") or row["lexical_item_id"]),
                lexical_item=LexicalItem(
                    id=str(row["lexical_item_id"]),
                    language=str(row["language"]),
                    word=str(row["word"]),
                    pos=str(row["pos"]),
                    definition=str(row["definition"]),
                    source=str(row["source"]),
                    sense=_optional_text(row.get("sense")),
                    reference_sentence=_optional_text(row.get("sentence")),
                    reference_sentence_id=_optional_text(row.get("sentence_id")),
                    word_freq_rank=_optional_int(row.get("word_freq_rank")),
                    sentence_count=_optional_int(row.get("sentence_count")),
                ),
                example_text=str(row["sentence"]),
                metadata={
                    "word_freq_rank": _optional_int(row.get("word_freq_rank")),
                    "sentence_count": _optional_int(row.get("sentence_count")),
                },
                source_name=str(row["source"]),
                source_url="",
            )
            for row in rows
        ]


def _optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value) -> int | None:
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
