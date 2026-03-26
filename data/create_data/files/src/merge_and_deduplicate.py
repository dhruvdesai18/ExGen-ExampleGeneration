"""
merge_and_deduplicate.py — Combine all processed source files into one
clean, deduplicated final dataset.

Pipeline:
    1. Load all processed parquet files from data/processed/
    2. Merge into one DataFrame
    3. Source prioritization: wiktionary > wordnet > tatoeba > leipzig
       (higher quality sources are kept over lower quality when same sentence
        appears in multiple sources)
    4. Per-word deduplication: if a word has >N sentences, sample to cap
       so the dataset is balanced and doesn't over-represent common words
    5. Compute per-word coverage statistics
    6. Save to data/final/english_word_examples.parquet

Output columns:
    word        : str — lemma
    sentence    : str — example sentence
    source      : str — originating dataset
    pos         : str — POS tag
    sense       : str — (if available) word sense / definition
    sentence_id : str — fingerprint for downstream dedup
    word_freq_rank : int — frequency rank (1 = most common word in dataset)

Usage:
    python src/merge_and_deduplicate.py
"""

import sys
import json
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import (
    PROC_DIR, FINAL_DIR, LOG_DIR,
    FINAL_DATASET_PATH, FINAL_DATASET_CSV, STATS_PATH,
    RANDOM_SEED,
)
from src.utils import get_logger

logger = get_logger("merge", LOG_DIR)
random.seed(RANDOM_SEED)

# ─── Per-word cap: max sentences per (word, source) combination ───────────────
# This prevents "the", "be", "have" from flooding the dataset.
# Adjust based on your memory constraints and training requirements.
MAX_SENTENCES_PER_WORD = 50

# ─── Source priority (higher = preferred when deduplicating) ─────────────────
SOURCE_PRIORITY = {
    "wiktionary":    3,   # wiktionary_hi and wiktionary_uk also match
    "wordnet":       2,
    "ud_ukrainian":  2,   # Gold-annotated UD = same quality as WordNet
    "brown_uk":      2,   # Curated corpus
    "iitb":          2,   # Parallel corpus, clean
    "ai4bharat":     1,
    "ubertext":      1,
}
# Any other source defaults to priority 1

def get_priority(source: str) -> int:
    for key, p in SOURCE_PRIORITY.items():
        if source.startswith(key):
            return p
    return 1


def load_all_processed() -> pd.DataFrame:
    """Load all parquet files from the processed directory."""
    parquet_files = list(PROC_DIR.glob("*_processed.parquet"))

    if not parquet_files:
        logger.error(f"No processed parquet files found in {PROC_DIR}.")
        logger.error("Run the collect_*.py scripts first.")
        sys.exit(1)

    dfs = []
    for path in parquet_files:
        # Skip any leftover tatoeba file (Tatoeba disabled)
        if "tatoeba" in path.name:
            logger.info(f"Skipping {path.name} (Tatoeba disabled - delete this file from data/processed/)")
            continue

        logger.info(f"Loading {path.name}...")
        df = pd.read_parquet(path)

        # Skip empty files or files missing expected columns
        if len(df) == 0 or "word" not in df.columns:
            logger.warning(f"  Skipping {path.name} - empty or missing 'word' column.")
            continue

        # Normalize column names across sources
        if "sense" not in df.columns:
            df["sense"] = ""
        if "synset_id" not in df.columns:
            df["synset_id"] = ""

        dfs.append(df)
        logger.info(f"  {len(df):,} rows, {df['word'].nunique():,} unique words")

    if not dfs:
        logger.error("All processed files were empty or invalid. Run collect scripts first.")
        sys.exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"\nTotal rows after merge: {len(merged):,}")
    logger.info(f"Total unique words: {merged['word'].nunique():,}")
    return merged


def deduplicate_by_sentence(df: pd.DataFrame) -> pd.DataFrame:
    """
    When the same sentence appears in multiple sources, keep only the
    highest-priority source version.
    """
    logger.info("Deduplicating by sentence_id (keeping highest-priority source)...")

    df["_priority"] = df["source"].apply(get_priority)
    df_sorted = df.sort_values("_priority", ascending=False)

    # Keep the first occurrence (highest priority) of each (word, sentence_id) pair
    df_dedup = df_sorted.drop_duplicates(subset=["word", "sentence_id"], keep="first")
    df_dedup = df_dedup.drop(columns=["_priority"])

    removed = len(df) - len(df_dedup)
    logger.info(f"Removed {removed:,} duplicate (word, sentence) pairs.")
    return df_dedup


def cap_per_word(df: pd.DataFrame, max_per_word: int = MAX_SENTENCES_PER_WORD) -> pd.DataFrame:
    """
    For words with more than max_per_word sentences, sample down.
    Prioritize: wiktionary > wordnet > tatoeba > leipzig (same priority order).
    """
    logger.info(f"Capping at {max_per_word} sentences per word...")

    df["_priority"] = df["source"].apply(get_priority)
    df_sorted = df.sort_values("_priority", ascending=False)

    # For each word, take up to max_per_word rows, prioritizing higher-quality sources
    capped = (
        df_sorted.groupby("word", group_keys=False)
        .apply(lambda g: g.head(max_per_word))
        .reset_index(drop=True)
    )
    capped = capped.drop(columns=["_priority"])

    removed = len(df) - len(capped)
    logger.info(f"Removed {removed:,} rows due to per-word cap.")
    return capped


def add_frequency_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add word_freq_rank column: rank 1 = word with the most example sentences.
    This is a proxy for how common the word is in the dataset.
    """
    word_counts = df["word"].value_counts().reset_index()
    word_counts.columns = ["word", "sentence_count"]
    word_counts["word_freq_rank"] = range(1, len(word_counts) + 1)

    df = df.merge(word_counts[["word", "word_freq_rank", "sentence_count"]], on="word", how="left")
    return df


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for logging and the stats JSON."""
    stats = {
        "total_pairs": len(df),
        "unique_words": df["word"].nunique(),
        "unique_sentences": df["sentence_id"].nunique(),
        "source_breakdown": df["source"].value_counts().to_dict(),
        "pos_breakdown": df["pos"].value_counts().to_dict(),
        "words_with_1_example": int((df.groupby("word").size() == 1).sum()),
        "words_with_5plus_examples": int((df.groupby("word").size() >= 5).sum()),
        "words_with_10plus_examples": int((df.groupby("word").size() >= 10).sum()),
        "avg_sentence_length_tokens": float(df["sentence"].str.split().apply(len).mean()),
        "median_sentence_length_tokens": float(df["sentence"].str.split().apply(len).median()),
    }
    return stats


def main():
    # 1. Load
    df = load_all_processed()

    # 2. Basic cleaning
    df["word"]     = df["word"].str.lower().str.strip()
    df["sentence"] = df["sentence"].str.strip()
    df = df.dropna(subset=["word", "sentence"])
    df = df[df["word"].str.len() > 1]          # drop single-char words
    df = df[df["word"].str.isalpha()]           # drop words with numbers/punctuation

    logger.info(f"After basic cleaning: {len(df):,} rows")

    # 3. Dedup by sentence content
    df = deduplicate_by_sentence(df)

    # 4. Cap per word
    df = cap_per_word(df, max_per_word=MAX_SENTENCES_PER_WORD)

    # 5. Add frequency rank
    df = add_frequency_rank(df)

    # 6. Final column selection & reorder
    final_cols = ["word", "sentence", "source", "pos", "sense", "sentence_id", "word_freq_rank", "sentence_count"]
    df = df[[c for c in final_cols if c in df.columns]]
    df = df.sort_values(["word", "source"]).reset_index(drop=True)

    # 7. Save
    logger.info(f"\nSaving final dataset → {FINAL_DATASET_PATH}")
    df.to_parquet(FINAL_DATASET_PATH, index=False)

    logger.info(f"Saving CSV copy → {FINAL_DATASET_CSV}")
    df.to_csv(FINAL_DATASET_CSV, index=False, encoding="utf-8")

    # 8. Stats
    stats = compute_stats(df)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n── Final Dataset Statistics ──────────────────────────────")
    for k, v in stats.items():
        if isinstance(v, dict):
            logger.info(f"  {k}:")
            for kk, vv in v.items():
                logger.info(f"    {kk}: {vv:,}" if isinstance(vv, int) else f"    {kk}: {vv}")
        else:
            logger.info(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    logger.info(f"\nStats saved → {STATS_PATH}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
