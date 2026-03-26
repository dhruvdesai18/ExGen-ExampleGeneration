"""
config.py — Central configuration for the English Dataset Pipeline.

Edit these values before running. All scripts import from here.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
FINAL_DIR  = DATA_DIR / "final"
LOG_DIR    = BASE_DIR / "logs"

for d in [RAW_DIR, PROC_DIR, FINAL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Sentence filtering thresholds ───────────────────────────────────────────

MIN_TOKENS        = 5      # Minimum tokens in a sentence
MAX_TOKENS        = 35     # Maximum tokens — longer = harder to learn from
MIN_CHARS         = 15     # Sanity check: at least 15 characters
MAX_CHARS         = 300    # Sanity check: cap on raw length

# ─── Quality filters ─────────────────────────────────────────────────────────

ALLOWED_LANGUAGES = {"en"}          # langdetect codes to keep
MIN_LANG_CONFIDENCE = 0.90          # Drop sentences below this confidence
MIN_ALPHA_RATIO     = 0.75          # Fraction of chars that must be alphabetic
MAX_UPPERCASE_RATIO = 0.30          # Drop ALL-CAPS heavy sentences
MIN_UNIQUE_TOKENS   = 3             # Sentence must have at least N distinct tokens

# ─── Target word inclusion ────────────────────────────────────────────────────

# When creating (word → sentence) pairs, the target word must appear in the
# sentence as its base lemma OR as a directly inflected surface form.
REQUIRE_EXACT_LEMMA_MATCH = True    # Use spaCy lemmatization to validate

# ─── POS tags to treat as content words (target word candidates) ──────────────
# spaCy universal POS tags
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# ─── Sources to collect ───────────────────────────────────────────────────────

SOURCES = {
    "tatoeba": False,
    "wiktionary": True,
    "wordnet": True,
    "leipzig": True,
}

# ─── Tatoeba settings ─────────────────────────────────────────────────────────

# HuggingFace dataset identifier
TATOEBA_HF_NAME    = "tatoeba"
TATOEBA_LANG_CODE  = "eng"          # Tatoeba uses ISO 639-2/3
TATOEBA_SPLIT      = "train"

# ─── Wiktionary settings ──────────────────────────────────────────────────────

# We parse the Wiktionary XML dump for English.
# Download from: https://dumps.wikimedia.org/enwiktionary/latest/
# File: enwiktionary-latest-pages-articles.xml.bz2 (~900MB compressed)
WIKTIONARY_DUMP_URL = (
    "https://dumps.wikimedia.org/enwiktionary/latest/"
    "enwiktionary-latest-pages-articles.xml.bz2"
)
WIKTIONARY_DUMP_PATH = RAW_DIR / "enwiktionary-latest-pages-articles.xml.bz2"

# Maximum entries to parse (set to None for full dump — takes ~30 min)
WIKTIONARY_MAX_ENTRIES = None

# ─── WordNet settings ─────────────────────────────────────────────────────────

# WordNet is downloaded via NLTK. No manual download needed.
# NLTK will download to ~/nltk_data on first run.

# ─── Leipzig Corpora settings ─────────────────────────────────────────────────

# Leipzig corpora page: https://wortschatz.uni-leipzig.de/en/download/English
# We download the 'news' and 'web' corpora (sentences files).
# Each file is ~50-200 MB. Format: TSV with (id, sentence).
LEIPZIG_URLS = [
    # 300k sentences from web sources, mixed register
    "https://downloads.wortschatz-leipzig.de/corpora/eng_news_2023_300K.tar.gz",
    # 300k sentences from Wikipedia
    "https://downloads.wortschatz-leipzig.de/corpora/eng_wikipedia_2021_300K.tar.gz",
]
LEIPZIG_RAW_DIR = RAW_DIR / "leipzig"

# ─── Output ───────────────────────────────────────────────────────────────────

FINAL_DATASET_PATH   = FINAL_DIR / "english_word_examples.parquet"
FINAL_DATASET_CSV    = FINAL_DIR / "english_word_examples.csv"
STATS_PATH           = FINAL_DIR / "dataset_stats.json"

# ─── Reproducibility ─────────────────────────────────────────────────────────

RANDOM_SEED = 42
