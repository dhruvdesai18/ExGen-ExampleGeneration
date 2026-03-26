"""
create_best_datasets.py
=======================
Creates two best-quality datasets from sampled_6000_enriched_hindi_cleaned.parquet

File 1: sampled_best_unique.parquet  — no duplicate target words (~2,620 rows)
File 2: sampled_best_3000.parquet    — 1,000 per language, duplicates allowed (3,000 rows)

Run from files/ directory:
    python create_best_datasets.py
"""

import pandas as pd
import re
from pathlib import Path

INPUT_PATH = Path("data/final/sampled_6000_enriched_hindi_cleaned.parquet")

# Output paths
UNIQUE_PATH     = Path("data/final/sampled_best_unique.parquet")
UNIQUE_CSV      = Path("data/final/sampled_best_unique.csv")
DUPES_PATH      = Path("data/final/sampled_best_3000.parquet")
DUPES_CSV       = Path("data/final/sampled_best_3000.csv")

# ── Scoring functions ─────────────────────────────────────────────────────────

def score_english(row):
    score = 0
    defn = str(row['definition']) if pd.notna(row['definition']) else ''
    if defn.strip(): score += 3
    if 'or a .' in defn or 'of a .' in defn: score -= 5
    if 30 <= len(defn) <= 150: score += 2
    if row['pos'] in ['NOUN','VERB','ADJ','ADV']: score += 1
    if 6 <= len(str(row['sentence']).split()) <= 20: score += 1
    return score

def score_hindi(row):
    score = 0
    defn = str(row['definition']) if pd.notna(row['definition']) else ''
    if defn.strip(): score += 3
    if any(m in defn for m in ['संज्ञा', 'सर्व॰', 'मुहा॰', '= दे॰']): score -= 2
    if 5 <= len(defn) <= 100: score += 2
    if re.search(r'ऋ{2,}|ऊण्', str(row['word'])): score -= 10
    if re.search(r'%s|%d|\{0\}|\\n|<s>', str(row['sentence'])): score -= 5
    if re.search(r'[a-zA-Z]', str(row['word'])): score -= 3
    if row['pos'] in ['NOUN','VERB','ADJ','ADV']: score += 1
    if len(str(row['word'])) >= 3: score += 1
    return score

def score_ukrainian(row):
    score = 0
    defn = str(row['definition']) if pd.notna(row['definition']) else ''
    if defn.strip(): score += 3
    if 3 <= len(defn) <= 80: score += 2
    if row['pos'] in ['NOUN','VERB','ADJ','ADV']: score += 1
    if 6 <= len(str(row['sentence']).split()) <= 25: score += 1
    return score

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_summary(df, label):
    has_def = df['definition'].notna() & (df['definition'].str.strip() != '')
    print(f"\n── {label} ──")
    print(f"  Total rows   : {len(df):,}")
    print(f"  Unique words : {df['word'].nunique():,}")
    print(f"  With def     : {has_def.sum():,} ({100*has_def.sum()/len(df):.1f}%)")
    print(f"  Source       : {df['source'].value_counts().to_dict()}")
    print(f"  POS          : {df['pos'].value_counts().to_dict()}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} rows")

    # Score all rows
    en_mask = df['source'].isin(['wordnet', 'wiktionary'])
    hi_mask = df['source'] == 'iitb'
    uk_mask = df['source'] == 'ud_ukrainian'

    df.loc[en_mask, 'score'] = df[en_mask].apply(score_english, axis=1)
    df.loc[hi_mask, 'score'] = df[hi_mask].apply(score_hindi, axis=1)
    df.loc[uk_mask, 'score'] = df[uk_mask].apply(score_ukrainian, axis=1)

    # ── FILE 1: No duplicate words ────────────────────────────────────────────
    print("\n=== Building File 1: No duplicate words ===")

    en_unique = (df[en_mask].sort_values('score', ascending=False)
                 .drop_duplicates(subset='word', keep='first')
                 .head(1000))

    hi_unique = (df[hi_mask].sort_values('score', ascending=False)
                 .drop_duplicates(subset='word', keep='first'))
    # Hindi only has 620 unique words — take all

    uk_unique = (df[uk_mask].sort_values('score', ascending=False)
                 .drop_duplicates(subset='word', keep='first')
                 .head(1000))

    unique_df = pd.concat([en_unique, hi_unique, uk_unique], ignore_index=True)
    unique_df = unique_df.drop(columns=['score'])
    unique_df = unique_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print_summary(unique_df, "sampled_best_unique")

    unique_df.to_parquet(UNIQUE_PATH, index=False)
    unique_df.to_csv(UNIQUE_CSV, index=False, encoding='utf-8-sig')
    print(f"\n  Saved: {UNIQUE_PATH}")
    print(f"  Saved: {UNIQUE_CSV}")

    # ── FILE 2: 1000 per language, duplicates allowed ─────────────────────────
    print("\n=== Building File 2: 1000 per language (3,000 total) ===")

    en_3k = (df[en_mask].sort_values('score', ascending=False)
             .head(1000))

    hi_3k = (df[hi_mask].sort_values('score', ascending=False)
             .head(1000))

    uk_3k = (df[uk_mask].sort_values('score', ascending=False)
             .head(1000))

    df_3k = pd.concat([en_3k, hi_3k, uk_3k], ignore_index=True)
    df_3k = df_3k.drop(columns=['score'])
    df_3k = df_3k.sample(frac=1, random_state=42).reset_index(drop=True)

    print_summary(df_3k, "sampled_best_3000")

    df_3k.to_parquet(DUPES_PATH, index=False)
    df_3k.to_csv(DUPES_CSV, index=False, encoding='utf-8-sig')
    print(f"\n  Saved: {DUPES_PATH}")
    print(f"  Saved: {DUPES_CSV}")

    print("\n✓ All done!")

if __name__ == "__main__":
    main()
