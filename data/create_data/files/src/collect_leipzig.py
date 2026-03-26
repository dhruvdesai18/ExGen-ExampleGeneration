"""
collect_leipzig.py — Download and process English sentences from the
Leipzig Corpora Collection.

Leipzig provides large, register-diverse monolingual corpora (news, web,
Wikipedia). We mine them for (word, sentence) pairs using spaCy.

Unlike Wiktionary/WordNet, these are natural sentences NOT written as
examples — so they provide coverage breadth for rare/uncommon words.

Download happens automatically. Files are cached in data/raw/leipzig/.

Output schema:
    word        : str — lemma
    sentence    : str — natural sentence from Leipzig
    source      : str — "leipzig_{corpus_name}"
    pos         : str — POS of the target word in this sentence
    sentence_id : str — fingerprint

Usage:
    python src/collect_leipzig.py

Note on corpus URLs:
    Leipzig provides corpora at different sizes (10K, 30K, 100K, 300K, 1M).
    Update LEIPZIG_CORPORA in this file to choose the ones you want.
    Full URL list: https://wortschatz.uni-leipzig.de/en/download/English
"""

import sys
import tarfile
import hashlib
import requests
from pathlib import Path
from io import BytesIO

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import (
    PROC_DIR, LOG_DIR, LEIPZIG_RAW_DIR,
    MIN_TOKENS, MAX_TOKENS, MIN_CHARS, MAX_CHARS,
    MIN_ALPHA_RATIO, MAX_UPPERCASE_RATIO, MIN_UNIQUE_TOKENS,
    MIN_LANG_CONFIDENCE, CONTENT_POS, REQUIRE_EXACT_LEMMA_MATCH,
)
from src.utils import (
    get_logger, get_nlp, is_valid_sentence, is_english,
    DeduplicatorSet, sentence_fingerprint,
)

logger = get_logger("leipzig", LOG_DIR)

# ─── Choose which Leipzig corpora to download ─────────────────────────────────
# Format: (url, corpus_name_for_source_column)
# Smaller sizes (10K, 30K) are fine for testing. Use 300K or 1M for production.
# Leipzig DISABLED — using quality sources only (WordNet + Wiktionary)
# To re-enable, uncomment entries below.
LEIPZIG_CORPORA = [
    # (
    #     "https://downloads.wortschatz-leipzig.de/corpora/eng_news_2023_300K.tar.gz",
    #     "leipzig_news_2023",
    # ),
    # (
    #     "https://downloads.wortschatz-leipzig.de/corpora/eng_wikipedia_2016_300K.tar.gz",
    #     "leipzig_wikipedia_2016",
    # ),
]


def download_corpus(url: str, dest_dir: Path) -> Path | None:
    """
    Download a Leipzig .tar.gz corpus file, cache locally.
    Returns path to the downloaded .tar.gz file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest_path = dest_dir / filename

    if dest_path.exists():
        logger.info(f"  Already downloaded: {filename}")
        return dest_path

    logger.info(f"  Downloading {filename} ...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        logger.info(f"  Saved to {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"  Failed to download {url}: {e}")
        return None


def extract_sentences_from_tar(tar_path: Path) -> list[str]:
    """
    Leipzig .tar.gz files contain a directory with several files.
    The sentences file contains 'sentences' in its name and ends with .txt
    Format: tab-separated  id<TAB>sentence  (some versions have no ID)
    """
    sentences = []
    logger.info(f"  Extracting sentences from {tar_path.name}...")

    with tarfile.open(tar_path, "r:gz") as tar:
        all_names = [m.name for m in tar.getmembers()]
        logger.info(f"  Archive contains {len(all_names)} files.")

        # Find sentences file — Leipzig uses both _sentences.txt and -sentences.txt
        sentences_member = None
        for member in tar.getmembers():
            name_lower = member.name.lower()
            if "sentences" in name_lower and name_lower.endswith(".txt"):
                sentences_member = member
                logger.info(f"  Found sentences file: {member.name}")
                break

        if sentences_member is None:
            logger.warning(f"  No sentences file found! Files in archive: {all_names[:15]}")
            return []

        f = tar.extractfile(sentences_member)
        if f:
            for line in f:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    # Standard format: id<TAB>sentence
                    sentence = parts[1].strip()
                elif len(parts) == 1:
                    # No ID column
                    sentence = parts[0].strip()
                else:
                    continue
                if sentence:
                    sentences.append(sentence)
            logger.info(f"  Loaded {len(sentences):,} raw sentences.")

    return sentences

def process_sentences(
    sentences: list[str],
    source_name: str,
    global_dedup: DeduplicatorSet,
) -> list[dict]:
    """
    Run quality filters and extract (word, sentence) pairs via spaCy.
    Uses a shared DeduplicatorSet to deduplicate across all Leipzig corpora.
    """
    nlp = get_nlp()
    records = []
    stats = {
        "total": len(sentences),
        "rejected_dup": 0,
        "rejected_quality": 0,
        "rejected_language": 0,
    }

    for sentence in tqdm(sentences, desc=f"Processing {source_name}", unit="sent"):
        sentence = sentence.strip()

        if global_dedup.is_duplicate(sentence):
            stats["rejected_dup"] += 1
            continue

        valid, _ = is_valid_sentence(
            sentence,
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS,
            min_chars=MIN_CHARS,
            max_chars=MAX_CHARS,
            min_alpha_ratio=MIN_ALPHA_RATIO,
            max_uppercase_ratio=MAX_UPPERCASE_RATIO,
            min_unique_tokens=MIN_UNIQUE_TOKENS,
        )
        if not valid:
            stats["rejected_quality"] += 1
            continue

        en, conf = is_english(sentence, min_confidence=MIN_LANG_CONFIDENCE)
        if not en:
            stats["rejected_language"] += 1
            continue

        doc = nlp(sentence)
        fp = sentence_fingerprint(sentence)

        for token in doc:
            if token.pos_ not in CONTENT_POS:
                continue
            if token.is_stop or token.is_punct or not token.is_alpha:
                continue
            if len(token.lemma_) < 2:
                continue

            records.append({
                "word":       token.lemma_.lower(),
                "sentence":   sentence,
                "source":     source_name,
                "pos":        token.pos_,
                "sentence_id": fp,
            })

    logger.info(f"  Stats: {stats}")
    logger.info(f"  Generated {len(records):,} (word, sentence) pairs.")
    return records


def main():
    all_records = []
    global_dedup = DeduplicatorSet()   # Shared across all corpora to avoid cross-corpus dups

    for url, corpus_name in LEIPZIG_CORPORA:
        logger.info(f"\n── Corpus: {corpus_name} ──")
        tar_path = download_corpus(url, LEIPZIG_RAW_DIR)
        if tar_path is None:
            continue

        sentences = extract_sentences_from_tar(tar_path)
        records = process_sentences(sentences, corpus_name, global_dedup)
        all_records.extend(records)

    if not all_records:
        logger.warning("No records collected from Leipzig corpora.")
        return

    df = pd.DataFrame(all_records)
    out_path = PROC_DIR / "leipzig_processed.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"\nTotal Leipzig (word, sentence) pairs: {len(df):,}")
    logger.info(f"Unique words: {df['word'].nunique():,}")
    logger.info(f"Source breakdown:\n{df['source'].value_counts().to_string()}")
    logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
