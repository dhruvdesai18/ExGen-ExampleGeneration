"""
collect_wiktionary.py — Extract (word, example_sentence) pairs from the
English Wiktionary XML dump.

Wiktionary is the HIGHEST QUALITY source in this pipeline because:
  - Examples are written BY LEXICOGRAPHERS for the purpose of illustrating word meaning.
  - Each example is directly linked to a specific word sense.
  - This is exactly the supervision signal we want.

Download the dump first (~900MB):
    wget https://dumps.wikimedia.org/enwiktionary/latest/enwiktionary-latest-pages-articles.xml.bz2 \
         -P data/raw/

Output schema:
    word        : str — headword (lemma)
    sentence    : str — example sentence from Wiktionary
    source      : str — "wiktionary"
    pos         : str — POS from Wiktionary markup (Noun, Verb, etc.)
    sense       : str — definition the example illustrates (useful for sense-aware training)
    sentence_id : str — fingerprint hash

Usage:
    python src/collect_wiktionary.py
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
from config import (
    PROC_DIR, LOG_DIR, WIKTIONARY_DUMP_PATH, WIKTIONARY_MAX_ENTRIES,
    MIN_TOKENS, MAX_TOKENS, MIN_CHARS, MAX_CHARS,
    MIN_ALPHA_RATIO, MAX_UPPERCASE_RATIO, MIN_UNIQUE_TOKENS,
)
from src.utils import (
    get_logger, is_valid_sentence, is_english,
    DeduplicatorSet, sentence_fingerprint,
)

logger = get_logger("wiktionary", LOG_DIR)


# ─── Wiktionary markup patterns ───────────────────────────────────────────────

# Wiktionary uses a custom markup. Key patterns we care about:
#
# == English ==          → language section header
# === Noun ===           → POS section header
# # Definition text      → a definition
# #: Example sentence    → example for the PRECEDING definition
# #* Quotation           → quotation (less useful, often archaic)
#
# We want: for each English headword, collect all (#, #:) pairs.

POS_HEADER_RE   = re.compile(r"^===+\s*(Noun|Verb|Adjective|Adverb|Pronoun|Interjection)\s*===+", re.MULTILINE)
ENGLISH_SEC_RE  = re.compile(r"^==\s*English\s*==", re.MULTILINE)
EXAMPLE_LINE_RE = re.compile(r"^#:\s*(.+)$", re.MULTILINE)
DEFN_LINE_RE    = re.compile(r"^#\s+([^:#\*].+)$", re.MULTILINE)

# Strip wikitext markup from examples: [[link|text]] → text, {{template}} → ''
WIKI_LINK_RE     = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
WIKI_TEMPLATE_RE = re.compile(r"\{\{[^}]+\}\}")
WIKI_BOLD_RE     = re.compile(r"'''?([^']+)'''?")
WIKI_TAG_RE      = re.compile(r"<[^>]+>")


def clean_wikitext(text: str) -> str:
    """Strip wikitext markup and return plain text."""
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = WIKI_TEMPLATE_RE.sub("", text)
    text = WIKI_BOLD_RE.sub(r"\1", text)
    text = WIKI_TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_english_block(wikitext: str) -> str | None:
    """
    Extract only the == English == section from a Wiktionary page.
    Returns None if the page has no English section.
    """
    match = ENGLISH_SEC_RE.search(wikitext)
    if not match:
        return None

    start = match.start()
    # Find the next top-level language section (== SomeLang ==) after English
    rest = wikitext[start + len(match.group()):]
    next_lang = re.search(r"\n==\s*[^=]", rest)
    end = start + len(match.group()) + (next_lang.start() if next_lang else len(rest))

    return wikitext[start:end]


def parse_page(title: str, wikitext: str) -> list[dict]:
    """
    Parse a single Wiktionary page and return a list of
    {word, sentence, pos, sense} dicts.
    """
    # Skip non-word pages
    if ":" in title or title.startswith("Wiktionary"):
        return []

    english_block = extract_english_block(wikitext)
    if not english_block:
        return []

    results = []
    current_pos = "UNKNOWN"
    current_sense = ""

    lines = english_block.split("\n")

    for i, line in enumerate(lines):
        # Track POS sections
        pos_match = POS_HEADER_RE.match(line)
        if pos_match:
            current_pos = pos_match.group(1).upper()
            continue

        # Track definition lines (# Definition...)
        defn_match = DEFN_LINE_RE.match(line)
        if defn_match:
            current_sense = clean_wikitext(defn_match.group(1))
            continue

        # Extract example lines (#: Example...)
        ex_match = EXAMPLE_LINE_RE.match(line)
        if ex_match:
            raw_example = ex_match.group(1)
            example = clean_wikitext(raw_example)

            if example and title.lower() in example.lower():
                results.append({
                    "word": title.lower(),
                    "sentence": example,
                    "source": "wiktionary",
                    "pos": current_pos,
                    "sense": current_sense[:200],   # truncate long definitions
                    "sentence_id": sentence_fingerprint(example),
                })

    return results


def parse_dump(dump_path: Path, max_entries: int | None = None) -> list[dict]:
    """
    Stream-parse the Wiktionary bz2 XML dump.

    We use iterative parsing (xml.etree.ElementTree.iterparse) to avoid
    loading the entire ~8GB uncompressed XML into memory.
    """
    logger.info(f"Parsing Wiktionary dump: {dump_path}")
    logger.info("This takes 10-30 minutes for the full dump. Set WIKTIONARY_MAX_ENTRIES in config.py to test faster.")

    records = []
    entry_count = 0
    ns = ""   # XML namespace prefix (populated from first element)

    opener = bz2.open if dump_path.suffix == ".bz2" else open

    with opener(dump_path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))

        title = None
        wikitext = None

        for event, elem in tqdm(context, desc="Wiktionary pages", unit="elem"):
            # Detect namespace from root element
            if ns == "" and "}" in elem.tag:
                ns = elem.tag.split("}")[0] + "}"

            tag = elem.tag.replace(ns, "")

            if event == "end" and tag == "title":
                title = elem.text or ""

            if event == "end" and tag == "text":
                wikitext = elem.text or ""

            if event == "end" and tag == "page":
                if title and wikitext and ":" not in title:
                    page_records = parse_page(title, wikitext)
                    records.extend(page_records)
                    entry_count += 1

                    if max_entries and entry_count >= max_entries:
                        logger.info(f"Stopping at {max_entries} entries (WIKTIONARY_MAX_ENTRIES limit).")
                        break

                # Free memory: clear processed elements
                elem.clear()
                title = None
                wikitext = None

    logger.info(f"Parsed {entry_count:,} pages, extracted {len(records):,} raw (word, example) pairs.")
    return records


def filter_records(records: list[dict]) -> pd.DataFrame:
    """Apply quality filters to extracted wiktionary examples."""
    dedup = DeduplicatorSet()
    clean = []
    stats = {"total": len(records), "rejected_quality": 0, "rejected_language": 0, "rejected_dup": 0}

    for rec in tqdm(records, desc="Filtering", unit="rec"):
        sentence = rec["sentence"]

        if dedup.is_duplicate(sentence):
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
        )
        if not valid:
            stats["rejected_quality"] += 1
            continue

        en, _ = is_english(sentence, min_confidence=0.85)
        if not en:
            stats["rejected_language"] += 1
            continue

        clean.append(rec)

    logger.info(f"Filter stats: {json.dumps(stats, indent=2)}")
    return pd.DataFrame(clean)


def main():
    if not WIKTIONARY_DUMP_PATH.exists():
        logger.error(
            f"Wiktionary dump not found at {WIKTIONARY_DUMP_PATH}.\n"
            f"Download it with:\n"
            f"  wget {WIKTIONARY_DUMP_PATH} -P data/raw/"
        )
        sys.exit(1)

    records = parse_dump(WIKTIONARY_DUMP_PATH, max_entries=WIKTIONARY_MAX_ENTRIES)
    df = filter_records(records)

    out_path = PROC_DIR / "wiktionary_processed.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"Saved {len(df):,} pairs → {out_path}")
    logger.info(f"Unique words: {df['word'].nunique():,}")
    logger.info(f"POS breakdown:\n{df['pos'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
