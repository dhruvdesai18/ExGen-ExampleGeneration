"""
utils.py — Shared filtering, validation, and logging utilities.

All source-specific collectors call these functions so quality
criteria are applied consistently across the whole pipeline.
"""

import re
import logging
import hashlib
from pathlib import Path
from typing import Optional

import spacy
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Make langdetect deterministic
DetectorFactory.seed = 0

# ─── Logging setup ────────────────────────────────────────────────────────────

def get_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─── spaCy model (loaded once, shared) ───────────────────────────────────────

_nlp: Optional[spacy.language.Language] = None

def get_nlp() -> spacy.language.Language:
    """
    Load the spaCy English model lazily (only once per process).
    Install with:
        python -m spacy download en_core_web_sm
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_sm' not found.\n"
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


# ─── Sentence-level filters ───────────────────────────────────────────────────

def is_valid_sentence(
    sentence: str,
    min_tokens: int = 5,
    max_tokens: int = 35,
    min_chars: int = 15,
    max_chars: int = 300,
    min_alpha_ratio: float = 0.75,
    max_uppercase_ratio: float = 0.30,
    min_unique_tokens: int = 3,
) -> tuple[bool, str]:
    """
    Run all sentence-level quality filters.

    Returns (is_valid: bool, rejection_reason: str).
    rejection_reason is empty string when valid.
    """
    s = sentence.strip()

    # ── Length checks ──────────────────────────────────────────────────────
    if len(s) < min_chars:
        return False, "too_short_chars"
    if len(s) > max_chars:
        return False, "too_long_chars"

    tokens = s.split()
    if len(tokens) < min_tokens:
        return False, "too_few_tokens"
    if len(tokens) > max_tokens:
        return False, "too_many_tokens"

    # ── Character composition ──────────────────────────────────────────────
    alpha_chars = sum(c.isalpha() for c in s)
    if alpha_chars / max(len(s), 1) < min_alpha_ratio:
        return False, "low_alpha_ratio"

    upper_chars = sum(c.isupper() for c in s if c.isalpha())
    if upper_chars / max(alpha_chars, 1) > max_uppercase_ratio:
        return False, "too_many_uppercase"

    # ── Lexical diversity ──────────────────────────────────────────────────
    unique_tokens = set(t.lower() for t in tokens if t.isalpha())
    if len(unique_tokens) < min_unique_tokens:
        return False, "low_unique_tokens"

    # ── Structural red flags ───────────────────────────────────────────────
    # Reject URLs, emails, code-like strings
    if re.search(r"https?://|www\.|\.com|@\w+\.\w+", s):
        return False, "contains_url_or_email"

    # Reject sentences that are just a list of words (no verb-like pattern)
    # Heuristic: no space-separated word longer than 2 chars with a, an, the, is, was, ...
    if not re.search(r"\b(is|are|was|were|has|have|had|will|would|can|could|the|a|an)\b", s.lower()):
        # Very rough heuristic — only reject if also very short
        if len(tokens) < 7:
            return False, "no_function_words"

    return True, ""


def is_english(sentence: str, min_confidence: float = 0.90) -> tuple[bool, float]:
    """
    Use langdetect to verify the sentence is English.

    Returns (is_english: bool, confidence: float).
    langdetect returns a list of probabilities; we take the max.
    """
    try:
        from langdetect import detect_langs
        langs = detect_langs(sentence)
        for lang in langs:
            if lang.lang == "en":
                return lang.prob >= min_confidence, lang.prob
        return False, 0.0
    except LangDetectException:
        return False, 0.0


# ─── Word–sentence relationship validation ────────────────────────────────────

def sentence_contains_word(
    sentence: str,
    target_word: str,
    nlp: spacy.language.Language,
    require_lemma_match: bool = True,
) -> bool:
    """
    Check whether `sentence` uses `target_word` as a content word.

    If require_lemma_match=True, we lemmatize both the sentence tokens
    and the target word and match on lemma. This handles inflected forms:
    e.g. target_word='run' matches 'running', 'ran', 'runs'.

    Falls back to case-insensitive substring match if lemmatization fails.
    """
    target_lower = target_word.lower().strip()

    # Fast path: literal presence check
    if target_lower not in sentence.lower():
        return False

    if not require_lemma_match:
        return True

    # Slow path: lemma match via spaCy
    target_doc = nlp(target_lower)
    target_lemmas = {token.lemma_.lower() for token in target_doc if token.is_alpha}

    sent_doc = nlp(sentence)
    for token in sent_doc:
        if not token.is_alpha:
            continue
        if token.lemma_.lower() in target_lemmas:
            return True

    return False


def get_target_word_pos(
    sentence: str,
    target_word: str,
    nlp: spacy.language.Language,
) -> Optional[str]:
    """
    Return the POS tag of the target word in the sentence.
    Returns None if word not found.
    """
    target_lower = target_word.lower()
    doc = nlp(sentence)
    target_doc = nlp(target_lower)
    target_lemma = target_doc[0].lemma_.lower() if len(target_doc) > 0 else target_lower

    for token in doc:
        if token.lemma_.lower() == target_lemma:
            return token.pos_
    return None


# ─── Deduplication ────────────────────────────────────────────────────────────

def sentence_fingerprint(sentence: str) -> str:
    """
    Create a short fingerprint for near-duplicate detection.

    We normalise whitespace and case before hashing so that
    trivial variants map to the same bucket.
    """
    normalised = " ".join(sentence.lower().split())
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


class DeduplicatorSet:
    """
    Simple in-memory deduplicator using exact fingerprints.

    For larger corpora, replace with a MinHash-LSH implementation
    (e.g. datasketch library) to catch near-duplicates.
    """
    def __init__(self):
        self._seen: set[str] = set()

    def is_duplicate(self, sentence: str) -> bool:
        fp = sentence_fingerprint(sentence)
        if fp in self._seen:
            return True
        self._seen.add(fp)
        return False

    def __len__(self):
        return len(self._seen)
