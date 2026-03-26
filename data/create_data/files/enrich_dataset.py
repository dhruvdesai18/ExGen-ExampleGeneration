"""
enrich_dataset_v3.py
====================
Enriches sampled_6000.parquet with:
  - Definitions from local Wiktionary XML dumps (offline, no API)
  - English fallback to WordNet if not in Wiktionary
  - Hindi POS via Stanza

Sources:
  English   → data/raw/enwiktionary-latest-pages-articles.xml.bz2 + WordNet fallback
  Hindi     → data/raw/hindi/hiwiktionary-latest-pages-articles.xml
  Ukrainian → data/raw/ukrainian/ukwiktionary-latest-pages-articles.xml

Run from files/ directory:
    python enrich_dataset_v3.py

Output:
    data/final/sampled_6000_enriched.parquet
    data/final/sampled_6000_enriched.csv
"""

import pandas as pd
import bz2
import re
import json
from pathlib import Path
from xml.etree import ElementTree as ET

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH   = Path("data/final/sampled_6000.parquet")
OUTPUT_PATH  = Path("data/final/sampled_6000_enriched.parquet")
OUTPUT_CSV   = Path("data/final/sampled_6000_enriched.csv")

EN_WIKT_PATH = Path("data/raw/enwiktionary-latest-pages-articles.xml.bz2")
HI_WIKT_PATH = Path("data/raw/hindi/hiwiktionary-latest-pages-articles.xml")
UK_WIKT_PATH = Path("data/raw/ukrainian/ukwiktionary-latest-pages-articles.xml")

# ── Language mapping ───────────────────────────────────────────────────────────
SOURCE_TO_LANG = {
    "wordnet":      "en",
    "wiktionary":   "en",
    "iitb":         "hi",
    "ud_ukrainian": "uk",
}

# ── Wiki markup cleaner ────────────────────────────────────────────────────────
WIKI_MARKUP_RE = re.compile(
    r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]"  # [[link|text]] → text
    r"|\{\{[^}]+\}\}"                   # {{template}} → remove
    r"|'{2,3}"                          # bold/italic → remove
    r"|<[^>]+>"                         # HTML tags → remove
    r"|\[https?://[^\]]*\]"             # [url] → remove
)

def clean_wiki(text: str) -> str:
    # [[link|display]] -> display, [[link]] -> link
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    # Remove templates {{...}}
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"'{2,3}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[https?://[^\]]*\]", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&[a-z]+;", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[#*:;]+\s*", "", text).strip()
    return text

def is_good_definition(text: str) -> bool:
    """Check if definition is valid — not empty, not just a redirect."""
    if not text or len(text) < 5:
        return False
    if text.startswith("#REDIRECT") or text.startswith("#redirect"):
        return False
    return True

# ── Wiktionary XML parser ─────────────────────────────────────────────────────
def parse_wiktionary_definitions(
    xml_path: Path,
    lang_section_re: re.Pattern,
    is_bz2: bool = False,
    max_pages: int = None,
) -> dict:
    """
    Parse a Wiktionary XML dump and extract definitions.
    Returns dict: {word: definition_string}

    Looks for:
      - Language section header matching lang_section_re
      - Lines starting with # (definitions, not #: which are examples)
    """
    definitions = {}

    if not xml_path.exists():
        print(f"  WARNING: File not found: {xml_path}")
        return definitions

    print(f"  Parsing {xml_path.name} ...")

    opener = bz2.open if is_bz2 else open
    mode   = "rb"

    title    = None
    wikitext = None
    ns       = ""
    count    = 0

    # Next-section detector — stops reading definitions when new == section starts
    ANY_SECTION_RE = re.compile(r"^==\s*\w", re.MULTILINE)

    try:
        with opener(xml_path, mode) as f:
            context = ET.iterparse(f, events=("start", "end"))
            for event, elem in context:
                # Detect namespace
                if ns == "" and "}" in elem.tag:
                    ns = elem.tag.split("}")[0] + "}"

                tag = elem.tag.replace(ns, "")

                if event == "end" and tag == "title":
                    title = elem.text or ""

                if event == "end" and tag == "text":
                    wikitext = elem.text or ""

                if event == "end" and tag == "page":
                    if title and wikitext and ":" not in title:
                        defn = extract_definition(
                            title, wikitext,
                            lang_section_re, ANY_SECTION_RE
                        )
                        if defn:
                            definitions[title.lower()] = defn
                        count += 1
                        if count % 200000 == 0:
                            print(f"    {count:,} pages processed, {len(definitions):,} definitions found")
                        if max_pages and count >= max_pages:
                            break

                    elem.clear()
                    title = wikitext = None

    except Exception as e:
        print(f"  Parse error: {e}")

    print(f"  Done: {count:,} pages → {len(definitions):,} definitions extracted")
    return definitions


def extract_definition(
    title: str,
    wikitext: str,
    lang_section_re: re.Pattern,
    any_section_re: re.Pattern,
) -> str:
    """
    Extract first valid definition. Handles:
    - English: ==English== section with # lines
    - Hindi:   {{-hi-}} marker with # lines, or shAbdAsAgar section
    - Ukrainian: {{-uk-}} marker + ====Значення==== subsection
    """

    def get_hash_lines(text):
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#:") or line.startswith("##") or line.startswith("#*"):
                continue
            if line.startswith("#"):
                clean = clean_wiki(line)
                if is_good_definition(clean):
                    return clean
        return ""

    # 1. Try standard ==Language== section (English)
    match = lang_section_re.search(wikitext)
    if match:
        section_text = wikitext[match.end():]
        next_sec = any_section_re.search(section_text)
        if next_sec:
            section_text = section_text[:next_sec.start()]
        result = get_hash_lines(section_text)
        if result:
            return result

    # 2. Try {{-hi-}} / {{-uk-}} language markers
    marker_match = re.search(r"\{\{-(hi|uk)-\}\}", wikitext)
    if marker_match:
        lang_code = marker_match.group(1)
        section_text = wikitext[marker_match.end():]
        # Stop at next top-level = section (another language)
        next_top = re.search(r"^= \{\{-(?!uk|hi)-\}\} =", section_text, re.MULTILINE)
        if next_top:
            section_text = section_text[:next_top.start()]

        if lang_code == "uk":
            # Ukrainian: find Значення subsection
            znamennya = re.search(r"={3,4}\s*Значення\s*={3,4}", section_text)
            if znamennya:
                sub_text = section_text[znamennya.end():]
                next_sub = re.search(r"^={3,4}[^=]", sub_text, re.MULTILINE)
                if next_sub:
                    sub_text = sub_text[:next_sub.start()]
                result = get_hash_lines(sub_text)
                if result:
                    return result
        else:
            # Hindi: definitions directly after marker
            result = get_hash_lines(section_text)
            if result:
                return result

    # 3. Hindi fallback: शब्दसागर dictionary
    shabda = re.search(r"={3,4}\s*शब्दसागर\s*={3,4}", wikitext)
    if shabda:
        section_text = wikitext[shabda.end():shabda.end()+600]
        for line in section_text.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("=") and not line.startswith("{") and not line.startswith("[") and not line.startswith("|"):
                clean = clean_wiki(line)
                clean = re.sub(r"^[०-९\d]+[\.\.।]\s*", "", clean).strip()
                if is_good_definition(clean) and len(clean) > 10:
                    return clean

    return ""


# ── WordNet fallback for English ──────────────────────────────────────────────
def build_wordnet_dict(words: list) -> dict:
    """Build WordNet definition lookup for a list of words."""
    try:
        from nltk.corpus import wordnet as wn
        result = {}
        for word in words:
            synsets = wn.synsets(word)
            if synsets:
                result[word.lower()] = synsets[0].definition()
        print(f"  WordNet: {len(result):,} definitions loaded")
        return result
    except Exception as e:
        print(f"  WordNet failed: {e}")
        return {}


# ── Stanza Hindi POS tagger ───────────────────────────────────────────────────
UPOS_MAP = {
    "NOUN": "NOUN", "VERB": "VERB", "AUX": "VERB",
    "ADJ":  "ADJ",  "ADV":  "ADV",  "PROPN": "NOUN",
    "PRON": "NOUN", "DET":  "ADJ",  "NUM":  "NOUN",
    "ADP":  "UNKNOWN", "CCONJ": "UNKNOWN", "SCONJ": "UNKNOWN",
    "PART": "UNKNOWN", "PUNCT": "UNKNOWN", "X": "UNKNOWN",
}

def load_stanza_hindi():
    try:
        import stanza
        print("  Loading Stanza Hindi model...")
        nlp = stanza.Pipeline('hi', processors='tokenize,pos',
                              download_method=None, verbose=False)
        print("  Stanza loaded.")
        return nlp
    except Exception as e:
        print(f"  Stanza failed: {e}")
        return None

def tag_hindi_pos_stanza(word: str, sentence: str, nlp) -> str:
    try:
        doc = nlp(sentence)
        for sent in doc.sentences:
            for token in sent.words:
                if token.text == word or token.lemma == word:
                    return UPOS_MAP.get(token.upos, "UNKNOWN")
        # Tag word alone if not found in sentence
        doc2 = nlp(word)
        for sent in doc2.sentences:
            for token in sent.words:
                return UPOS_MAP.get(token.upos, "UNKNOWN")
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} rows\n")

    # Add language column
    df["language"] = df["source"].map(SOURCE_TO_LANG)
    df["definition"] = ""

    # Normalise POS
    df["pos"] = df["pos"].replace({"ADJECTIVE": "ADJ", "ADVERB": "ADV"})

    # ── Step 1: Build definition dictionaries from XML ─────────────────────────

    # English Wiktionary
    print("── Building English definition dictionary ──")
    EN_SECTION_RE = re.compile(r"^==\s*English\s*==", re.MULTILINE)
    en_wikt_defs = parse_wiktionary_definitions(
        EN_WIKT_PATH,
        EN_SECTION_RE,
        is_bz2=True,
    )

    # WordNet fallback for English words not in Wiktionary
    print("\n── Building WordNet fallback dictionary ──")
    en_words = df[df["language"] == "en"]["word"].unique().tolist()
    wn_defs = build_wordnet_dict(en_words)

    # Hindi Wiktionary
    print("\n── Building Hindi definition dictionary ──")
    HI_SECTION_RE = re.compile(
        r"^==\s*(?:हिन्दी|हिंदी|Hindi)\s*==", re.MULTILINE
    )
    hi_wikt_defs = parse_wiktionary_definitions(
        HI_WIKT_PATH,
        HI_SECTION_RE,
        is_bz2=False,
    )

    # Ukrainian — parse from English Wiktionary (==Ukrainian== section)
    # Much better coverage: 1,360/1,642 words vs only 45 from ukwiktionary
    print("\n── Building Ukrainian definition dictionary (from EN Wiktionary) ──")
    UK_SECTION_RE = re.compile(r"^==\s*Ukrainian\s*==", re.MULTILINE)
    uk_wikt_defs = parse_wiktionary_definitions(
        EN_WIKT_PATH,
        UK_SECTION_RE,
        is_bz2=True,
    )

    print(f"\nDictionary sizes:")
    print(f"  English Wiktionary : {len(en_wikt_defs):,} words")
    print(f"  WordNet fallback   : {len(wn_defs):,} words")
    print(f"  Hindi Wiktionary   : {len(hi_wikt_defs):,} words")
    print(f"  Ukrainian (EN Wikt): {len(uk_wikt_defs):,} words")

    # ── Step 2: Apply definitions to dataset ───────────────────────────────────
    print("\n── Applying definitions to dataset ──")

    for idx, row in df.iterrows():
        lang = row["language"]
        word = str(row["word"]).lower().strip()

        if lang == "en":
            defn = en_wikt_defs.get(word, "") or wn_defs.get(word, "")
        elif lang == "hi":
            defn = hi_wikt_defs.get(word, "")
        elif lang == "uk":
            defn = uk_wikt_defs.get(word, "")
        else:
            defn = ""

        df.at[idx, "definition"] = defn

    # ── Step 3: Hindi POS via Stanza ──────────────────────────────────────────
    print("\n── Hindi POS tagging via Stanza ──")
    nlp_hi = load_stanza_hindi()

    if nlp_hi:
        hi_rows = df[df["language"] == "hi"]
        count = 0
        for idx, row in hi_rows.iterrows():
            pos = tag_hindi_pos_stanza(row["word"], row["sentence"], nlp_hi)
            df.at[idx, "pos"] = pos
            count += 1
            if count % 200 == 0:
                print(f"  Hindi POS: {count}/{len(hi_rows)} tagged")
        print(f"  Done: {count} rows tagged")

    # ── Step 4: Summary ────────────────────────────────────────────────────────
    print("\n── Enrichment Summary ──")
    print(f"Total rows          : {len(df):,}")
    print(f"With definition     : {(df['definition'] != '').sum():,}")
    print(f"Without definition  : {(df['definition'] == '').sum():,}")

    print(f"\nDefinition coverage by language:")
    for lang, label in [("en","English"), ("hi","Hindi"), ("uk","Ukrainian")]:
        lang_df = df[df["language"] == lang]
        covered = (lang_df["definition"] != "").sum()
        pct = 100 * covered / len(lang_df) if len(lang_df) > 0 else 0
        print(f"  {label:10}: {covered:,}/{len(lang_df):,} ({pct:.1f}%)")

    print(f"\nPOS breakdown:")
    print(df["pos"].value_counts().to_string())

    print(f"\nSample rows:")
    for lang in ["en", "hi", "uk"]:
        row = df[df["language"] == lang].iloc[0]
        print(f"\n  [{lang}] word={row['word']} | pos={row['pos']}")
        print(f"       def ={row['definition'][:100] if row['definition'] else 'EMPTY'}")
        print(f"       sent={row['sentence'][:80]}")

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    print(f"\nSaving to {OUTPUT_PATH} ...")
    df.to_parquet(OUTPUT_PATH, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Done!")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
