"""
collect_hindi.py — Collect Hindi (word, sentence) pairs from:

    1. IIT Bombay English-Hindi Parallel Corpus  (Hindi side only)
    2. AI4Bharat IndicNLP Monolingual Corpus
    3. Hindi Wiktionary XML dump                 (optional, highest quality)

Expected files in data/raw/hindi/:
    iitb_hi.txt                                  ← one Hindi sentence per line
    ai4bharat_hi.txt                             ← one Hindi sentence per line
    hiwiktionary-latest-pages-articles.xml.bz2   ← optional

How to get the files:
    IIT Bombay:
        wget https://www.cfilt.iitb.ac.in/iitb_parallel/iitb_corpus_download/parallel.tgz
        tar -xzf parallel.tgz
        cp parallel/IITB.en-hi.hi  data/raw/hindi/iitb_hi.txt

    AI4Bharat:
        wget https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-1/hi.txt
        mv hi.txt data/raw/hindi/ai4bharat_hi.txt

    Hindi Wiktionary:
        wget https://dumps.wikimedia.org/hiwiktionary/latest/hiwiktionary-latest-pages-articles.xml.bz2
        mv hiwiktionary-latest-pages-articles.xml.bz2 data/raw/hindi/

Output schema (parquet):
    word        : str  — target token (surface form, lowercased)
    sentence    : str  — Hindi example sentence
    source      : str  — "iitb" | "ai4bharat" | "wiktionary_hi"
    pos         : str  — "UNKNOWN" (no Hindi spaCy model used; Stanza optional)
    sentence_id : str  — MD5 fingerprint

Usage:
    python src/collect_hindi.py

Notes on NLP tooling for Hindi:
    spaCy does not have a production Hindi model. We use simple whitespace
    tokenisation + the IndicNLP library for basic lemmatisation if available.
    If IndicNLP is not installed, we fall back to raw surface forms as words.
    Install IndicNLP with:
        pip install indic-nlp-library
        python -c "from indicnlp import common; common.set_resources_path('indic_nlp_resources')"
        # Download resources: https://github.com/anoopkunchukuttan/indic_nlp_resources
"""

import re
import bz2
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import PROC_DIR, LOG_DIR, MIN_CHARS, MAX_CHARS
from src.utils import get_logger, is_valid_sentence, DeduplicatorSet, sentence_fingerprint

logger = get_logger("hindi", LOG_DIR)

# ─── Paths ────────────────────────────────────────────────────────────────────

HINDI_RAW_DIR    = Path(__file__).parents[1] / "data" / "raw" / "hindi"
IITB_PATH        = HINDI_RAW_DIR / "IITB.en-hi.hi"        # parallel corpus Hindi side
MONOLINGUAL_PATH = HINDI_RAW_DIR / "monolingual.hi"        # IIT Bombay monolingual Hindi
WIKTIONARY_PATH  = HINDI_RAW_DIR / "hiwiktionary-latest-pages-articles.xml"  # uncompressed

# ─── Hindi-specific filter settings ──────────────────────────────────────────

# Hindi sentences can be shorter in token count due to agglutination.
# We relax the token minimum but keep character minimums.
MIN_TOKENS_HI   = 3
MAX_TOKENS_HI   = 40
MIN_CHARS_HI    = 10
MAX_CHARS_HI    = 400

# Devanagari Unicode block: U+0900–U+097F
# We require at least 60% of alpha chars to be Devanagari
MIN_DEVANAGARI_RATIO = 0.60

# ─── Devanagari detection ─────────────────────────────────────────────────────

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

def is_devanagari_char(c: str) -> bool:
    return bool(DEVANAGARI_RE.match(c))

def devanagari_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    deva = [c for c in alpha if is_devanagari_char(c)]
    return len(deva) / len(alpha)

def is_valid_hindi(sentence: str) -> tuple[bool, str]:
    """Hindi-specific validity check on top of the generic filter."""
    s = sentence.strip()

    if len(s) < MIN_CHARS_HI:
        return False, "too_short"
    if len(s) > MAX_CHARS_HI:
        return False, "too_long"

    tokens = s.split()
    if len(tokens) < MIN_TOKENS_HI:
        return False, "too_few_tokens"
    if len(tokens) > MAX_TOKENS_HI:
        return False, "too_many_tokens"

    # Must be predominantly Devanagari
    ratio = devanagari_ratio(s)
    if ratio < MIN_DEVANAGARI_RATIO:
        return False, f"low_devanagari_ratio_{ratio:.2f}"

    # Reject URLs
    if re.search(r"https?://|www\.", s):
        return False, "contains_url"

    return True, ""


# ─── Tokenisation / word extraction ──────────────────────────────────────────

def try_load_indicnlp():
    """
    Try to load IndicNLP tokeniser.
    NOTE: We no longer use IndicNLP for tokenisation — whitespace tokenisation
    works better in practice because IndicNLP's trivial_tokenize splits on
    punctuation in ways that leave too many non-Devanagari fragments.
    This function is kept for compatibility but always returns None.
    """
    try:
        from indicnlp.tokenize import indic_tokenize  # noqa: F401
        logger.info("IndicNLP available (not used for tokenisation — using robust whitespace method).")
    except ImportError:
        pass
    return None   # Always use our own tokeniser


def tokenize_hindi(sentence: str, tokeniser=None) -> list[str]:
    """
    Tokenise a Hindi sentence into word tokens.

    Strategy:
    1. Split on whitespace
    2. Strip ALL non-Devanagari characters from each token
    3. Keep tokens that are purely Devanagari and >= 2 chars

    This is more reliable than IndicNLP trivial_tokenize for our use case
    because it doesn't depend on external resources and handles the full
    Unicode Devanagari block correctly.

    Hindi does not use spaces within words (unlike English compounds),
    so whitespace splitting gives us real word boundaries.
    """
    tokens = sentence.split()
    clean = []
    seen = set()

    for t in tokens:
        # Strip everything that is not Devanagari unicode (U+0900-U+097F)
        # This removes punctuation, digits, Latin chars, danda (।), etc.
        deva_only = re.sub(r'[^\u0900-\u097F]', '', t).strip()

        if len(deva_only) < 2:
            continue

        # Skip function-word-like single characters after stripping
        if deva_only in seen:
            continue
        seen.add(deva_only)

        clean.append(deva_only)

    return clean


# ─── Source collectors ────────────────────────────────────────────────────────

def collect_from_text_file(
    path: Path,
    source_name: str,
    tokeniser,
    dedup: DeduplicatorSet,
    max_sentences: int = 500_000,  # Hard cap to prevent MemoryError on large files
) -> list[dict]:
    """
    Generic collector for plain text files (one sentence per line).
    Streams line-by-line — never loads the whole file into RAM.
    max_sentences caps how many sentences are processed (default 500k).
    """
    if not path.exists():
        logger.warning(f"File not found, skipping: {path}")
        return []

    logger.info(f"Loading {source_name} from {path} (max {max_sentences:,} sentences)...")
    records = []
    stats = {"total": 0, "rejected": 0, "duplicate": 0, "pairs": 0}

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc=source_name, unit="sent"):
            if stats["total"] >= max_sentences:
                logger.info(f"  Reached cap of {max_sentences:,} sentences.")
                break

            sentence = line.strip()
            if not sentence:
                continue

            stats["total"] += 1

            if dedup.is_duplicate(sentence):
                stats["duplicate"] += 1
                continue

            valid, reason = is_valid_hindi(sentence)
            if not valid:
                stats["rejected"] += 1
                continue

            tokens = tokenize_hindi(sentence, tokeniser)
            if len(tokens) < MIN_TOKENS_HI:
                stats["rejected"] += 1
                continue

            fp = sentence_fingerprint(sentence)

            # Emit one (word, sentence) pair per unique token in this sentence
            seen_in_sentence = set()
            for token in tokens:
                token_lower = token.strip()
                if len(token_lower) < 2:
                    continue
                if token_lower in seen_in_sentence:
                    continue
                seen_in_sentence.add(token_lower)

                records.append({
                    "word":        token_lower,
                    "sentence":    sentence,
                    "source":      source_name,
                    "pos":         "UNKNOWN",
                    "sentence_id": fp,
                    "sense":       "",
                })
                stats["pairs"] += 1

    logger.info(f"  {source_name} stats: {json.dumps(stats)}")
    return records


# ─── Wiktionary Hindi parser ──────────────────────────────────────────────────

HINDI_SEC_RE    = re.compile(r"^==\s*हिन्दी\s*==|^==\s*Hindi\s*==", re.MULTILINE)
EXAMPLE_LINE_RE = re.compile(r"^#:\s*(.+)$", re.MULTILINE)
WIKI_MARKUP_RE  = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]|\{\{[^}]+\}\}|'{2,3}|<[^>]+>")

def clean_wiki(text: str) -> str:
    text = WIKI_MARKUP_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

def parse_hindi_wiktionary_page(title: str, wikitext: str) -> list[dict]:
    """Extract (word, example) pairs from a Hindi Wiktionary page."""
    if ":" in title:
        return []

    # Find Hindi section
    match = HINDI_SEC_RE.search(wikitext)
    if not match:
        return []

    block = wikitext[match.start():]
    next_sec = re.search(r"\n==\s*[^=\n]", block[3:])
    if next_sec:
        block = block[:next_sec.start() + 3]

    results = []
    for ex_match in EXAMPLE_LINE_RE.finditer(block):
        example = clean_wiki(ex_match.group(1))
        if not example:
            continue

        valid, _ = is_valid_hindi(example)
        if not valid:
            continue

        # Check the headword appears in the example
        if title.lower() not in example:
            continue

        results.append({
            "word":       title.strip(),
            "sentence":   example,
            "source":     "wiktionary_hi",
            "pos":        "UNKNOWN",
            "sentence_id": sentence_fingerprint(example),
            "sense":      "",
        })

    return results


def collect_wiktionary_hindi(dedup: DeduplicatorSet) -> list[dict]:
    if not WIKTIONARY_PATH.exists():
        logger.warning(f"Hindi Wiktionary dump not found at {WIKTIONARY_PATH} — skipping.")
        return []

    logger.info(f"Parsing Hindi Wiktionary dump: {WIKTIONARY_PATH}")
    records = []
    count = 0
    ns = ""

    is_bz2 = str(WIKTIONARY_PATH).endswith(".bz2")
    opener = bz2.open if is_bz2 else open
    with opener(WIKTIONARY_PATH, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        title = wikitext = None

        for event, elem in tqdm(context, desc="Hindi Wiktionary", unit="elem"):
            if ns == "" and "}" in elem.tag:
                ns = elem.tag.split("}")[0] + "}"
            tag = elem.tag.replace(ns, "")

            if event == "end" and tag == "title":
                title = elem.text or ""
            if event == "end" and tag == "text":
                wikitext = elem.text or ""
            if event == "end" and tag == "page":
                if title and wikitext:
                    page_records = parse_hindi_wiktionary_page(title, wikitext)
                    for rec in page_records:
                        if not dedup.is_duplicate(rec["sentence"]):
                            records.append(rec)
                    count += 1
                elem.clear()
                title = wikitext = None

    logger.info(f"Hindi Wiktionary: parsed {count:,} pages, {len(records):,} pairs.")
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    HINDI_RAW_DIR.mkdir(parents=True, exist_ok=True)
    tokeniser = try_load_indicnlp()
    dedup = DeduplicatorSet()
    all_records = []

    # 1. IIT Bombay parallel corpus (Hindi side)
    records = collect_from_text_file(IITB_PATH, "iitb", tokeniser, dedup)
    all_records.extend(records)

    # 2. IIT Bombay monolingual — DISABLED: noisy web/news text
    # records = collect_from_text_file(MONOLINGUAL_PATH, "iitb_monolingual", tokeniser, dedup)
    # all_records.extend(records)

    # 3. Hindi Wiktionary — KEPT: purpose-written examples
    records = collect_wiktionary_hindi(dedup)
    all_records.extend(records)

    if not all_records:
        logger.error(
            "No Hindi records collected. Make sure at least one of these files exists:\n"
            f"  {IITB_PATH}\n  {WIKTIONARY_PATH}"
        )
        return

    df = pd.DataFrame(all_records)
    df["word"]     = df["word"].str.strip()
    df["sentence"] = df["sentence"].str.strip()
    df = df.dropna(subset=["word", "sentence"])
    df = df[df["word"].str.len() >= 2]

    out_path = PROC_DIR / "hindi_processed.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"\nSaved {len(df):,} Hindi (word, sentence) pairs → {out_path}")
    logger.info(f"Unique words    : {df['word'].nunique():,}")
    logger.info(f"Unique sentences: {df['sentence_id'].nunique():,}")
    logger.info(f"Source breakdown:\n{df['source'].value_counts().to_string()}")

    # Sample output
    logger.info("\nSample records:")
    for _, row in df.sample(min(5, len(df))).iterrows():
        logger.info(f"  [{row['source']}] {row['word'][:20]:<20} → {row['sentence'][:80]}")


if __name__ == "__main__":
    main()
