from __future__ import annotations

from pathlib import Path

import pandas as pd

from exgen.types import LexicalItem


def load_lexical_items(path: str | Path) -> list[LexicalItem]:
    frame = pd.read_parquet(path)
    required = {"language", "word", "pos", "definition"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    source_default = Path(path).stem
    items: list[LexicalItem] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        items.append(
            LexicalItem(
                id=str(row.get("id", f"lex-{index}")),
                language=str(row["language"]),
                word=str(row["word"]),
                pos=str(row["pos"]),
                definition=str(row["definition"]),
                source=str(row.get("source", source_default)),
            )
        )
    return items
