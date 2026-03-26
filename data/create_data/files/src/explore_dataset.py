"""
explore_dataset.py — Validate and explore the final English dataset.

Run this AFTER merge_and_deduplicate.py to:
  - Print summary statistics
  - Check for obvious quality issues
  - Show sample records per source and POS
  - Plot word frequency distribution (saved as PNG)
  - Flag potential problem words for manual review

Usage:
    python src/explore_dataset.py
"""

import sys
import json
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import FINAL_DATASET_PATH, STATS_PATH, FINAL_DIR, LOG_DIR
from src.utils import get_logger

logger = get_logger("explore", LOG_DIR)


def load_dataset() -> pd.DataFrame:
    if not FINAL_DATASET_PATH.exists():
        logger.error(f"Final dataset not found: {FINAL_DATASET_PATH}")
        logger.error("Run merge_and_deduplicate.py first.")
        sys.exit(1)
    return pd.read_parquet(FINAL_DATASET_PATH)


def print_summary(df: pd.DataFrame):
    logger.info("═" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("═" * 60)
    logger.info(f"Total (word, sentence) pairs : {len(df):,}")
    logger.info(f"Unique words                 : {df['word'].nunique():,}")
    logger.info(f"Unique sentences             : {df['sentence_id'].nunique():,}")
    logger.info(f"Avg sentence length (tokens) : {df['sentence'].str.split().apply(len).mean():.1f}")
    logger.info(f"Median sentence length       : {df['sentence'].str.split().apply(len).median():.1f}")
    logger.info("")

    logger.info("── Source breakdown ────────────────────────────────────")
    for source, count in df["source"].value_counts().items():
        pct = count / len(df) * 100
        logger.info(f"  {source:<35} {count:>8,}  ({pct:.1f}%)")
    logger.info("")

    logger.info("── POS breakdown ───────────────────────────────────────")
    for pos, count in df["pos"].value_counts().items():
        pct = count / len(df) * 100
        logger.info(f"  {pos:<10} {count:>8,}  ({pct:.1f}%)")
    logger.info("")

    logger.info("── Word coverage ───────────────────────────────────────")
    wc = df.groupby("word").size()
    for threshold in [1, 3, 5, 10, 20]:
        n = (wc >= threshold).sum()
        logger.info(f"  Words with >= {threshold:2d} examples: {n:,}")
    logger.info("")


def print_samples(df: pd.DataFrame, n: int = 3):
    """Print sample records for each source and POS for eyeballing."""
    logger.info("── Sample records by source ────────────────────────────")
    for source in df["source"].unique():
        subset = df[df["source"] == source]
        logger.info(f"\n  Source: {source}  ({len(subset):,} pairs)")
        for _, row in subset.sample(min(n, len(subset))).iterrows():
            logger.info(f"  [{row['pos']}] {row['word']!r:20s} → {row['sentence'][:100]}")

    logger.info("\n── Sample records by POS ───────────────────────────────")
    for pos in ["NOUN", "VERB", "ADJ", "ADV"]:
        subset = df[df["pos"] == pos]
        if len(subset) == 0:
            continue
        logger.info(f"\n  POS: {pos}  ({len(subset):,} pairs)")
        for _, row in subset.sample(min(n, len(subset))).iterrows():
            logger.info(f"  {row['word']!r:20s} | {row['source']:<20} | {row['sentence'][:80]}")


def flag_problems(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify potential quality issues for manual inspection.

    Returns a DataFrame of flagged records.
    """
    flags = []

    # Flag 1: Sentence doesn't contain the word (basic sanity check)
    mask_missing = ~df.apply(
        lambda row: row["word"] in row["sentence"].lower(), axis=1
    )
    flagged = df[mask_missing].copy()
    flagged["flag"] = "word_not_in_sentence"
    flags.append(flagged)

    # Flag 2: Very short sentences that passed the filter somehow
    mask_short = df["sentence"].str.split().apply(len) < 5
    flagged2 = df[mask_short & ~mask_missing].copy()
    flagged2["flag"] = "suspiciously_short"
    flags.append(flagged2)

    # Flag 3: Single-example words (might be noisy hapax legomena)
    word_counts = df.groupby("word").size()
    single_example_words = set(word_counts[word_counts == 1].index)
    flagged3 = df[df["word"].isin(single_example_words)].copy()
    flagged3["flag"] = "single_example_word"
    flags.append(flagged3)

    if flags:
        all_flags = pd.concat(flags, ignore_index=True)
        flag_path = FINAL_DIR / "flagged_records.csv"
        all_flags.to_csv(flag_path, index=False)
        logger.info(f"── Flagged records ─────────────────────────────────────")
        logger.info(f"  word_not_in_sentence : {len(flagged):,}")
        logger.info(f"  suspiciously_short   : {len(flagged2):,}")
        logger.info(f"  single_example_word  : {len(flagged3):,}")
        logger.info(f"  Saved to {flag_path} for manual review.")
        return all_flags

    return pd.DataFrame()


def plot_word_frequency(df: pd.DataFrame):
    """Save a word frequency distribution plot."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        word_counts = df.groupby("word").size().sort_values(ascending=False)
        ranks = np.arange(1, len(word_counts) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Log-log plot (Zipf's law check)
        axes[0].loglog(ranks, word_counts.values, ".", alpha=0.3, markersize=2)
        axes[0].set_xlabel("Word rank (log scale)")
        axes[0].set_ylabel("Number of examples (log scale)")
        axes[0].set_title("Word frequency distribution (Zipf plot)")
        axes[0].grid(True, alpha=0.3)

        # Histogram of examples per word
        axes[1].hist(word_counts.values, bins=50, color="steelblue", edgecolor="white")
        axes[1].set_xlabel("Number of examples per word")
        axes[1].set_ylabel("Number of words")
        axes[1].set_title("Distribution of examples per word")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = FINAL_DIR / "word_frequency_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Frequency distribution plot saved → {plot_path}")

    except ImportError:
        logger.info("  (matplotlib not installed — skipping frequency plot)")


def main():
    df = load_dataset()
    print_summary(df)
    print_samples(df, n=3)
    flag_problems(df)
    plot_word_frequency(df)

    logger.info("\nExploration complete.")


if __name__ == "__main__":
    main()
