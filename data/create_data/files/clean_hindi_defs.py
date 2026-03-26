"""
clean_hindi_defs.py
===================
Cleans verbose Hindi definitions from शब्दसागर dictionary.
Strips grammatical annotations and keeps only the actual meaning.

Run from files/ directory:
    python clean_hindi_defs.py

Input:  data/final/sampled_6000_enriched.parquet
Output: data/final/sampled_6000_enriched.parquet (overwrites)
        data/final/sampled_6000_enriched.csv (overwrites)
"""

import pandas as pd
import re
from pathlib import Path

INPUT_PATH  = Path("data/final/sampled_6000_enriched.parquet")
OUTPUT_PATH = Path("data/final/sampled_6000_enriched_hindi_cleaned.parquet")
OUTPUT_CSV  = Path("data/final/sampled_6000_enriched_hindi_cleaned.csv")

def clean_hindi_definition(text):
    if not text or not isinstance(text, str):
        return text
    original = text.strip()

    # Skip already clean definitions
    GRAMMAR_MARKERS = ['संज्ञा', 'सर्व॰', 'वि॰', 'क्रि॰', 'अव्य॰', 'पुं॰', 'स्त्री॰', '^१', '^२']
    if not any(m in text for m in GRAMMAR_MARKERS):
        return original

    # Remove word + superscript at very start
    text = re.sub(r'^[^\s]+\s*\^?[०-९\d]*\s*', '', text).strip()

    # Remove POS tags with gender markers
    text = re.sub(r'(?:संज्ञा|सर्वनाम|विशेषण|क्रिया|अव्यय)\s*(?:पुं॰|स्त्री॰|नपुं॰)?\s*', '', text).strip()
    text = re.sub(r'(?:वि॰|क्रि॰|अव्य॰|सर्व॰|पुं॰|स्त्री॰|नपुं॰)\s*', '', text).strip()

    # Remove etymology brackets [सं॰], [अ॰], [हिं॰] etc
    text = re.sub(r'\[[\w॰\s\.\,\-\'\u0900-\u097F]+\]', '', text).strip()

    # Remove numbered sense at start
    text = re.sub(r'^[०-९\d]+[\.\।]\s*', '', text).strip()

    # Remove old citations उ॰—...
    text = re.sub(r'उ॰\s*—.*$', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'—[\u0900-\u097F\s]+\(शब्द[\w॰]*\).*$', '', text).strip()

    # Remove reference markers (को॰), (शब्द॰)
    text = re.sub(r'\([\w॰]+\)', '', text).strip()
    text = re.sub(r'दे॰\s*[\'"][\w\s\u0900-\u097F]+[\'"]', '', text).strip()

    # Keep only first sense (before । २.)
    parts = re.split(r'\s*।\s*[०-९\d]+[\.\s]', text)
    text = parts[0].strip() if parts else text

    # Clean trailing punctuation
    text = re.sub(r'[।\s]+$', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()

    return text if len(text) >= 3 else original


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} rows")

    hi_mask = (df['source'] == 'iitb') & df['definition'].notna() & (df['definition'].str.strip() != '')
    print(f"Hindi rows with definitions: {hi_mask.sum():,}")

    before = df.loc[hi_mask, 'definition'].copy()
    df.loc[hi_mask, 'definition'] = df.loc[hi_mask, 'definition'].apply(clean_hindi_definition)
    after = df.loc[hi_mask, 'definition']

    changed = (before != after).sum()
    print(f"Definitions cleaned: {changed:,}")

    print(f"\nAvg length before: {before.str.len().mean():.0f} chars")
    print(f"Avg length after : {after.str.len().mean():.0f} chars")

    print("\n── Before/After samples ──")
    changed_idx = before[before != after].index[:8]
    for idx in changed_idx:
        print(f"  BEFORE: {before[idx][:90]}")
        print(f"  AFTER:  {after[idx][:90]}")
        print()

    print(f"\nSaving to {OUTPUT_PATH} ...")
    df.to_parquet(OUTPUT_PATH, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("Done!")

    print("\n── Final Coverage ──")
    for src, label in [('wordnet','English WN'), ('wiktionary','English WK'),
                       ('iitb','Hindi'), ('ud_ukrainian','Ukrainian')]:
        sub = df[df['source'] == src]
        covered = (sub['definition'].notna() & (sub['definition'].str.strip() != '')).sum()
        print(f"  {label:15}: {covered}/{len(sub)} ({100*covered/len(sub):.1f}%)")

if __name__ == "__main__":
    main()
