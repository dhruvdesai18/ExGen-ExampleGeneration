"""
collect_ukrainian.py — Collect Ukrainian (word, sentence) pairs from:

    1. UberText 2.0            — large diverse Ukrainian corpus (~5GB)
    2. Brown-UK / CNC-UA       — clean curated sentences (GitHub repo)
    3. Universal Dependencies  — syntactically annotated .conllu files
    4. Ukrainian Wiktionary    — optional, word-keyed examples

Expected files in data/raw/ukrainian/:
    ubertext/          ← extracted UberText 2.0 folder (contains .txt files)
    brown_uk/          ← cloned https://github.com/brown-uk/corpus (has .txt files)
    ud_ukrainian/      ← cloned https://github.com/UniversalDependencies/UD_Ukrainian-IU
    ukwiktionary-latest-pages-articles.xml.bz2   ← optional

How to get the files:
    UberText 2.0:
        Download from https://lang.org.ua/en/corpora/#anchor4
        Extract into data/raw/ukrainian/ubertext/

    Brown-UK:
        git clone https://github.com/brown-uk/corpus data/raw/ukrainian/brown_uk

    Universal Dependencies Ukrainian:
        git clone https://github.com/UniversalDependencies/UD_Ukrainian-IU data/raw/ukrainian/ud_ukrainian

    Ukrainian Wiktionary:
        wget https://dumps.wikimedia.org/ukwiktionary/latest/ukwiktionary-latest-pages-articles.xml.bz2
        mv ukwiktionary*.bz2 data/raw/ukrainian/

Output schema (parquet):
    word        : str  — target token (surface form, lowercased)
    sentence    : str  — Ukrainian example sentence
    source      : str  — "ubertext" | "brown_uk" | "ud_ukrainian" | "wiktionary_uk"
    pos         : str  — POS tag (from UD annotations where available, else UNKNOWN)
    sentence_id : str  — MD5 fingerprint
    sense       : str  — empty (populated by Wiktionary only)

Usage:
    python src/collect_ukrainian.py

Notes on NLP tooling for Ukrainian:
    spaCy has no Ukrainian model in the base install.
    We use simple tokenisation for plain text sources.
    For UD CoNLL-U files, we parse the annotations directly — these give us
    gold-standard lemmas and POS tags for free, making UD the highest-quality
    source despite being small.

    For better Ukrainian NLP, install:
        pip install stanza
        python -c "import stanza; stanza.download('uk')"
    Then set USE_STANZA = True below.
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
from config import PROC_DIR, LOG_DIR
from src.utils import get_logger, is_valid_sentence, DeduplicatorSet, sentence_fingerprint

logger = get_logger("ukrainian", LOG_DIR)

# ─── Paths ────────────────────────────────────────────────────────────────────

UK_RAW_DIR       = Path(__file__).parents[1] / "data" / "raw" / "ukrainian"
UBERTEXT_DIR     = UK_RAW_DIR / "ubertext"
BROWN_UK_DIR     = UK_RAW_DIR / "brown_uk"
UD_DIR           = UK_RAW_DIR / "ud_ukrainian"
WIKTIONARY_PATH  = UK_RAW_DIR / "ukwiktionary-latest-pages-articles.xml"

# ─── Ukrainian-specific filter settings ──────────────────────────────────────

MIN_TOKENS_UK   = 3
MAX_TOKENS_UK   = 45
MIN_CHARS_UK    = 10
MAX_CHARS_UK    = 500

# Ukrainian Cyrillic block: U+0400–U+04FF
# We require at least 60% of alpha chars to be Cyrillic
MIN_CYRILLIC_RATIO = 0.60

# Toggle Stanza NLP (better lemmatisation but slower)
USE_STANZA = False   # Set to True if you have installed stanza + uk model

# ─── Cyrillic detection ───────────────────────────────────────────────────────

CYRILLIC_RE = re.compile(r'[\u0400-\u04FF]')

def cyrillic_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    cyr = [c for c in alpha if CYRILLIC_RE.match(c)]
    return len(cyr) / len(alpha)

def is_valid_ukrainian(sentence: str) -> tuple[bool, str]:
    """Ukrainian-specific validity checks."""
    s = sentence.strip()

    if len(s) < MIN_CHARS_UK:
        return False, "too_short"
    if len(s) > MAX_CHARS_UK:
        return False, "too_long"

    tokens = s.split()
    if len(tokens) < MIN_TOKENS_UK:
        return False, "too_few_tokens"
    if len(tokens) > MAX_TOKENS_UK:
        return False, "too_many_tokens"

    ratio = cyrillic_ratio(s)
    if ratio < MIN_CYRILLIC_RATIO:
        return False, f"low_cyrillic_ratio_{ratio:.2f}"

    if re.search(r"https?://|www\.", s):
        return False, "contains_url"

    return True, ""


# ─── Tokenisation ─────────────────────────────────────────────────────────────

def try_load_stanza():
    """Attempt to load Stanza Ukrainian pipeline. Returns pipeline or None."""
    if not USE_STANZA:
        return None
    try:
        import stanza
        nlp = stanza.Pipeline("uk", processors="tokenize,pos,lemma", verbose=False)
        logger.info("Stanza Ukrainian pipeline loaded.")
        return nlp
    except Exception as e:
        logger.warning(f"Stanza not available: {e}. Using whitespace tokenisation.")
        return None


def tokenize_ukrainian(sentence: str, nlp_pipeline) -> list[tuple[str, str]]:
    """
    Tokenise a Ukrainian sentence.
    Returns list of (surface_form, pos_tag) tuples.

    If Stanza is available, returns lemmas + POS.
    Otherwise returns (surface_form, 'UNKNOWN').
    """
    if nlp_pipeline is not None:
        try:
            doc = nlp_pipeline(sentence)
            tokens = []
            for sent in doc.sentences:
                for word in sent.words:
                    if CYRILLIC_RE.search(word.text) and len(word.lemma) >= 2:
                        tokens.append((word.lemma.lower(), word.upos or "UNKNOWN"))
            return tokens
        except Exception:
            pass

    # Fallback: whitespace tokenisation
    tokens = []
    for t in sentence.split():
        t_clean = re.sub(r'[.,!?;:"\'()\[\]{}<>«»—–]', '', t).strip()
        if t_clean and CYRILLIC_RE.search(t_clean) and len(t_clean) >= 2:
            tokens.append((t_clean.lower(), "UNKNOWN"))
    return tokens


# ─── CoNLL-U parser (for Universal Dependencies) ─────────────────────────────

def parse_conllu_file(path: Path) -> list[dict]:
    """
    Parse a Universal Dependencies CoNLL-U file.

    CoNLL-U format (tab-separated columns):
        ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC

    We reconstruct sentences from token lines and extract
    (lemma, sentence, upos) for each content word.

    This is the highest-quality source because:
    - Sentences are grammatically well-formed (manually annotated)
    - We get gold-standard lemmas and POS tags for free
    """
    records = []
    dedup = DeduplicatorSet()

    CONTENT_UPOS = {"NOUN", "VERB", "ADJ", "ADV"}

    with open(path, encoding="utf-8") as f:
        current_tokens = []   # list of (form, lemma, upos)
        current_words  = []   # surface forms to reconstruct sentence

        for line in f:
            line = line.rstrip("\n")

            if line.startswith("#"):
                # Comment line — ignore
                continue

            if line.strip() == "":
                # End of sentence — process what we have
                if current_words:
                    sentence = " ".join(current_words)

                    valid, _ = is_valid_ukrainian(sentence)
                    if valid and not dedup.is_duplicate(sentence):
                        fp = sentence_fingerprint(sentence)
                        seen = set()
                        for form, lemma, upos in current_tokens:
                            if upos not in CONTENT_UPOS:
                                continue
                            key = (lemma, fp)
                            if key in seen:
                                continue
                            seen.add(key)
                            records.append({
                                "word":       lemma.lower(),
                                "sentence":   sentence,
                                "source":     "ud_ukrainian",
                                "pos":        upos,
                                "sentence_id": fp,
                                "sense":      "",
                            })

                current_tokens = []
                current_words  = []
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            token_id = parts[0]
            # Skip multi-word tokens (e.g. "1-2") and empty nodes (e.g. "1.1")
            if "-" in token_id or "." in token_id:
                continue

            form  = parts[1]
            lemma = parts[2]
            upos  = parts[3]

            current_words.append(form)
            current_tokens.append((form, lemma, upos))

    logger.info(f"  {path.name}: {len(records):,} (word, sentence) pairs")
    return records


def collect_universal_dependencies(dedup: DeduplicatorSet) -> list[dict]:
    """Collect from all .conllu files in the UD Ukrainian directory."""
    if not UD_DIR.exists():
        logger.warning(f"UD directory not found: {UD_DIR} — skipping.")
        return []

    conllu_files = list(UD_DIR.rglob("*.conllu"))
    if not conllu_files:
        logger.warning(f"No .conllu files found in {UD_DIR} — skipping.")
        return []

    logger.info(f"Found {len(conllu_files)} .conllu files in {UD_DIR}")
    all_records = []

    for path in conllu_files:
        records = parse_conllu_file(path)
        # Apply global dedup
        for rec in records:
            if not dedup.is_duplicate(rec["sentence"] + rec["word"]):
                all_records.append(rec)

    logger.info(f"UD Ukrainian total: {len(all_records):,} pairs")
    return all_records


# ─── Plain text file collector (UberText, Brown-UK) ──────────────────────────

def collect_from_text_files(
    source_dir: Path,
    source_name: str,
    nlp_pipeline,
    dedup: DeduplicatorSet,
    file_patterns: list[str] = ["*.txt"],
    max_sentences: int = 500_000,
) -> list[dict]:
    """
    Collect (word, sentence) pairs from all text files in a directory.
    Handles both single large files and directories of smaller files.
    """
    if not source_dir.exists():
        logger.warning(f"{source_name} directory not found: {source_dir} — skipping.")
        return []

    # Find all matching text files
    all_files = []
    for pattern in file_patterns:
        all_files.extend(source_dir.rglob(pattern))

    if not all_files:
        logger.warning(f"No text files found in {source_dir} — skipping.")
        return []

    logger.info(f"{source_name}: found {len(all_files)} file(s) in {source_dir}")

    records = []
    stats = {"total": 0, "rejected": 0, "duplicate": 0, "pairs": 0}
    sentence_count = 0

    for file_path in all_files:
        if sentence_count >= max_sentences:
            logger.info(f"  Reached max_sentences limit ({max_sentences:,}). Stopping.")
            break

        logger.info(f"  Processing: {file_path.name}")

        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"  Could not read {file_path}: {e}")
            continue

        for line in tqdm(lines, desc=file_path.name[:40], unit="line", leave=False):
            if sentence_count >= max_sentences:
                break

            sentence = line.strip()
            if not sentence:
                continue

            stats["total"] += 1

            if dedup.is_duplicate(sentence):
                stats["duplicate"] += 1
                continue

            valid, _ = is_valid_ukrainian(sentence)
            if not valid:
                stats["rejected"] += 1
                continue

            sentence_count += 1
            tokens = tokenize_ukrainian(sentence, nlp_pipeline)

            if len(tokens) < MIN_TOKENS_UK:
                stats["rejected"] += 1
                continue

            fp = sentence_fingerprint(sentence)
            seen_in_sent = set()

            for word, pos in tokens:
                if word in seen_in_sent or len(word) < 2:
                    continue
                seen_in_sent.add(word)
                records.append({
                    "word":       word,
                    "sentence":   sentence,
                    "source":     source_name,
                    "pos":        pos,
                    "sentence_id": fp,
                    "sense":      "",
                })
                stats["pairs"] += 1

    logger.info(f"  {source_name} stats: {json.dumps(stats)}")
    return records


# ─── Ukrainian Wiktionary parser ──────────────────────────────────────────────

UKRANIAN_SEC_RE = re.compile(r"^==\s*Українська\s*==|^==\s*Ukrainian\s*==", re.MULTILINE)
EXAMPLE_LINE_RE = re.compile(r"^#:\s*(.+)$", re.MULTILINE)
WIKI_MARKUP_RE  = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]|\{\{[^}]+\}\}|'{2,3}|<[^>]+>")

def clean_wiki(text: str) -> str:
    text = WIKI_MARKUP_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

def parse_ukrainian_wiktionary_page(title: str, wikitext: str) -> list[dict]:
    if ":" in title:
        return []

    match = UKRANIAN_SEC_RE.search(wikitext)
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
        valid, _ = is_valid_ukrainian(example)
        if not valid:
            continue
        results.append({
            "word":       title.strip().lower(),
            "sentence":   example,
            "source":     "wiktionary_uk",
            "pos":        "UNKNOWN",
            "sentence_id": sentence_fingerprint(example),
            "sense":      "",
        })
    return results


def collect_wiktionary_ukrainian(dedup: DeduplicatorSet) -> list[dict]:
    if not WIKTIONARY_PATH.exists():
        logger.warning(f"Ukrainian Wiktionary dump not found at {WIKTIONARY_PATH} — skipping.")
        return []

    logger.info(f"Parsing Ukrainian Wiktionary dump: {WIKTIONARY_PATH}")
    records = []
    count = 0
    ns = ""

    opener = bz2.open if str(WIKTIONARY_PATH).endswith(".bz2") else open
    with opener(WIKTIONARY_PATH, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        title = wikitext = None

        for event, elem in tqdm(context, desc="Ukrainian Wiktionary", unit="elem"):
            if ns == "" and "}" in elem.tag:
                ns = elem.tag.split("}")[0] + "}"
            tag = elem.tag.replace(ns, "")

            if event == "end" and tag == "title":
                title = elem.text or ""
            if event == "end" and tag == "text":
                wikitext = elem.text or ""
            if event == "end" and tag == "page":
                if title and wikitext:
                    page_records = parse_ukrainian_wiktionary_page(title, wikitext)
                    for rec in page_records:
                        if not dedup.is_duplicate(rec["sentence"]):
                            records.append(rec)
                    count += 1
                elem.clear()
                title = wikitext = None

    logger.info(f"Ukrainian Wiktionary: {count:,} pages parsed, {len(records):,} pairs.")
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    UK_RAW_DIR.mkdir(parents=True, exist_ok=True)
    nlp_pipeline = try_load_stanza()
    dedup = DeduplicatorSet()
    all_records = []

    # 1. Universal Dependencies — smallest but highest quality (gold lemmas + POS)
    logger.info("\n── Universal Dependencies Ukrainian ──")
    records = collect_universal_dependencies(dedup)
    all_records.extend(records)

    # 2. Brown-UK — DISABLED: small, lower quality relative to UD
    # logger.info("\n── Brown-UK Corpus ──")
    # records = collect_from_text_files(
    #     BROWN_UK_DIR, "brown_uk", nlp_pipeline, dedup,
    #     file_patterns=["*.txt"],
    #     max_sentences=100_000,
    # )
    # all_records.extend(records)

    # 3. UberText 2.0 — DISABLED: noisy web text
    # logger.info("\n── UberText 2.0 ──")
    # records = collect_from_text_files(
    #     UBERTEXT_DIR, "ubertext", nlp_pipeline, dedup,
    #     file_patterns=["*.txt"],
    #     max_sentences=500_000,
    # )
    # all_records.extend(records)

    # 4. Ukrainian Wiktionary
    logger.info("\n── Ukrainian Wiktionary ──")
    records = collect_wiktionary_ukrainian(dedup)
    all_records.extend(records)

    if not all_records:
        logger.error(
            "No Ukrainian records collected. Make sure at least one source exists:\n"
            f"  {UBERTEXT_DIR}\n  {BROWN_UK_DIR}\n  {UD_DIR}\n  {WIKTIONARY_PATH}"
        )
        return

    df = pd.DataFrame(all_records)
    df["word"]     = df["word"].str.strip()
    df["sentence"] = df["sentence"].str.strip()
    df = df.dropna(subset=["word", "sentence"])
    df = df[df["word"].str.len() >= 2]

    out_path = PROC_DIR / "ukrainian_processed.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"\nSaved {len(df):,} Ukrainian (word, sentence) pairs → {out_path}")
    logger.info(f"Unique words    : {df['word'].nunique():,}")
    logger.info(f"Unique sentences: {df['sentence_id'].nunique():,}")
    logger.info(f"Source breakdown:\n{df['source'].value_counts().to_string()}")
    logger.info(f"POS breakdown:\n{df['pos'].value_counts().to_string()}")

    logger.info("\nSample records:")
    for _, row in df.sample(min(5, len(df))).iterrows():
        logger.info(f"  [{row['source']}] {row['word'][:20]:<20} → {row['sentence'][:80]}")


if __name__ == "__main__":
    main()
