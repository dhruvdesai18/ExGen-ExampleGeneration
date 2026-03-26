"""
collect_wordnet.py — Extract (word, example_sentence) pairs from WordNet
via NLTK.

WordNet provides:
  - ~117,000 synsets (groups of synonymous words)
  - Each synset may have example sentences (called "examples" in NLTK API)
  - Each lemma has a POS tag
  - Synsets have definitions (glosses)

This gives us HIGH-QUALITY, SENSE-DISAMBIGUATED examples — small in number
(~25,000 examples in total) but very clean.

No manual download required: NLTK downloads WordNet automatically on first run.

Output schema:
    word        : str — lemma
    sentence    : str — example sentence
    source      : str — "wordnet"
    pos         : str — POS (NOUN / VERB / ADJ / ADV)
    sense       : str — WordNet gloss / definition for context
    synset_id   : str — WordNet synset name (e.g. "dog.n.01")
    sentence_id : str — fingerprint

Usage:
    python src/collect_wordnet.py
"""

import sys
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import PROC_DIR, LOG_DIR, MIN_TOKENS, MAX_TOKENS, MIN_CHARS, MAX_CHARS
from src.utils import get_logger, is_valid_sentence, DeduplicatorSet, sentence_fingerprint

logger = get_logger("wordnet", LOG_DIR)


def download_nltk_data():
    """Download required NLTK data packages if not already present."""
    import nltk
    packages = ["wordnet", "omw-1.4", "averaged_perceptron_tagger"]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK package '{pkg}': {e}")


POS_MAP = {
    "n": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "s": "ADJ",   # adjective satellite
    "r": "ADV",
}


def extract_wordnet_pairs() -> list[dict]:
    """
    Iterate over all WordNet synsets and extract (lemma, example) pairs.

    For each synset:
      - Each example sentence is valid for ALL lemmas in that synset.
      - We emit one record per (lemma × example) pair.
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        raise ImportError("Install with: pip install nltk && python -m nltk.downloader wordnet omw-1.4")

    records = []
    synsets = list(wn.all_synsets())
    logger.info(f"Total WordNet synsets: {len(synsets):,}")

    dedup = DeduplicatorSet()

    for synset in tqdm(synsets, desc="WordNet synsets", unit="synset"):
        examples   = synset.examples()   # list of example sentences
        gloss      = synset.definition()
        pos_char   = synset.pos()
        pos_label  = POS_MAP.get(pos_char, "OTHER")
        synset_id  = synset.name()       # e.g. "dog.n.01"

        if not examples:
            continue

        lemma_names = [lemma.name().replace("_", " ").lower() for lemma in synset.lemmas()]

        for example in examples:
            example = example.strip()

            if dedup.is_duplicate(example):
                continue

            valid, _ = is_valid_sentence(
                example,
                min_tokens=MIN_TOKENS,
                max_tokens=MAX_TOKENS,
                min_chars=MIN_CHARS,
                max_chars=MAX_CHARS,
            )
            if not valid:
                continue

            fp = sentence_fingerprint(example)

            # Emit one record per lemma that appears in the example sentence
            for lemma in lemma_names:
                if lemma in example.lower():
                    records.append({
                        "word":       lemma,
                        "sentence":   example,
                        "source":     "wordnet",
                        "pos":        pos_label,
                        "sense":      gloss[:200],
                        "synset_id":  synset_id,
                        "sentence_id": fp,
                    })

    logger.info(f"Extracted {len(records):,} (word, sentence) pairs from WordNet.")
    return records


def main():
    download_nltk_data()
    records = extract_wordnet_pairs()

    if not records:
        logger.warning("No records extracted. Check NLTK WordNet installation.")
        return

    df = pd.DataFrame(records)

    out_path = PROC_DIR / "wordnet_processed.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"Saved {len(df):,} pairs → {out_path}")
    logger.info(f"Unique words: {df['word'].nunique():,}")
    logger.info(f"POS breakdown:\n{df['pos'].value_counts().to_string()}")

    # Show a few examples so you can eyeball quality
    logger.info("\nSample records:")
    for _, row in df.sample(min(5, len(df))).iterrows():
        logger.info(f"  [{row['pos']}] {row['word']!r:20s} → {row['sentence']}")


if __name__ == "__main__":
    main()
