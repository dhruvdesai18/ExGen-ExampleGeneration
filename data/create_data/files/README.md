# Multilingual Word-Example Generation Dataset
### MSc Artificial Intelligence — NLP Project
**National University of Ireland, Galway**

---

## Overview

This project builds a multilingual dataset of **(word, example sentence)** pairs for training a language learning model. Given a target word, the model learns to generate a natural, contextually appropriate example sentence in the same language.

**Languages:** English · Hindi · Ukrainian  
**Task:** Controlled example sentence generation for language learners  
**Research Question:** Can a fine-tuned multilingual model generate high-quality, word-specific example sentences across typologically diverse languages?

---

## Repository Structure

```
├── run_pipeline.py              # Master pipeline runner
├── config.py                    # All settings and paths
├── requirements.txt             # Dependencies
├── enrich_dataset.py            # Adds definitions + Hindi POS (Stanza)
├── clean_hindi_defs.py          # Cleans verbose Hindi definitions
├── create_best_datasets.py      # Creates final training datasets
│
├── src/
│   ├── collect_wiktionary.py    # English Wiktionary collector
│   ├── collect_wordnet.py       # WordNet collector
│   ├── collect_leipzig.py       # Leipzig collector (disabled)
│   ├── collect_hindi.py         # Hindi IITB collector
│   ├── collect_ukrainian.py     # Ukrainian UD collector
│   ├── merge_and_deduplicate.py # Merges all sources
│   ├── explore_dataset.py       # Statistics and quality checks
│   └── utils.py                 # Shared filters and utilities
│
└── data/
    ├── raw/                     # Raw source files (not on GitHub)
    ├── processed/               # Per-source parquet files (not on GitHub)
    └── final/                   # Final datasets ✓
        ├── sampled_best_unique.parquet   # 2,620 rows, no duplicate words
        ├── sampled_best_3000.parquet     # 3,000 rows, best quality
        ├── sampled_6000_enriched_hindi_cleaned.parquet  # Full enriched dataset
        └── english_word_examples.parquet               # Complete raw dataset
```

---

## Dataset

### Sources

| Language | Source | Pairs | Quality |
|---|---|---|---|
| English | Princeton WordNet | 23,237 | ★★★★★ Lexicographer-written |
| English | English Wiktionary | 1,533 | ★★★★★ Purpose-written examples |
| Hindi | IIT Bombay Parallel Corpus (IITB) | 13,503 | ★★★★ Verified parallel text |
| Ukrainian | Universal Dependencies Ukrainian-IU | 46,192 | ★★★★★ Gold-annotated treebank |

### Final Training Datasets

**`sampled_best_3000.parquet`** — Primary training dataset
- 3,000 rows (1,000 per language)
- Best quality examples selected by scoring
- Definitions and POS tags for all rows

**`sampled_best_unique.parquet`** — No-duplicate dataset
- 2,620 rows (all unique target words)
- 1,000 English + 620 Hindi + 1,000 Ukrainian

### Schema

| Column | Type | Description |
|---|---|---|
| word | str | Target word (lemma) |
| sentence | str | Example sentence containing the word |
| source | str | Data source identifier |
| pos | str | Part of speech (NOUN/VERB/ADJ/ADV) |
| definition | str | Word definition (same language for EN/HI, English for UK) |
| language | str | Language code (en/hi/uk) |
| sentence_id | str | MD5 fingerprint of sentence |

### Input Format for Training

```
[lang=en] word: run → She runs every morning to stay fit.
[lang=hi] word: किताब → मैंने पुस्तकालय से एक किताब उधार ली।
[lang=uk] word: бігти → Він біжить щодня вранці.
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_REPO_URL
cd nlp_project/files

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK WordNet
python -c "import nltk; nltk.download('wordnet')"

# 4. Download Stanza Hindi model
python -c "import stanza; stanza.download('hi')"
```

---

## Running the Pipeline

```bash
# Full pipeline (regenerates everything from scratch)
python run_pipeline.py

# Individual stages
python run_pipeline.py --stage wordnet
python run_pipeline.py --stage wiktionary
python run_pipeline.py --stage hindi
python run_pipeline.py --stage ukrainian
python run_pipeline.py --stage merge
python run_pipeline.py --stage explore
```

### Raw Data Required

Place these files in `data/raw/` before running:

| File | Source | Language |
|---|---|---|
| `enwiktionary-latest-pages-articles.xml.bz2` | dumps.wikimedia.org | English |
| `hindi/IITB.en-hi.hi` | IIT Bombay | Hindi |
| `hindi/hiwiktionary-latest-pages-articles.xml` | dumps.wikimedia.org | Hindi |
| `ukrainian/ud_ukrainian/*.conllu` | github.com/UniversalDependencies | Ukrainian |
| `ukrainian/ukwiktionary-latest-pages-articles.xml` | dumps.wikimedia.org | Ukrainian |

---

## Enrichment Pipeline

After running the main pipeline, enrich the dataset with definitions and POS:

```bash
# Step 1 — Add definitions + Hindi POS (takes ~40 mins due to Wiktionary XML parsing)
python enrich_dataset.py

# Step 2 — Clean verbose Hindi definitions
python clean_hindi_defs.py

# Step 3 — Create final best-quality datasets
python create_best_datasets.py
```

---

## Known Limitations

- **Hindi POS:** Tagged using Stanza `hdtb` model — not gold standard
- **Hindi definitions:** From 19th century शब्दसागर dictionary — verbose style, 33% missing
- **Ukrainian definitions:** In English (from English Wiktionary `==Ukrainian==` section)
- **word_not_in_sentence:** 31% of rows flagged — expected for Hindi/Ukrainian due to morphological inflection (lemma ≠ surface form)
- **Ukrainian vocabulary:** Limited to ~17,000 unique words vs ~56,000 for Hindi

---

## License

- **WordNet:** Princeton WordNet License
- **IIT Bombay Corpus:** Available for research use
- **Universal Dependencies Ukrainian-IU:** CC BY-NC-SA 4.0
- **Wiktionary:** CC BY-SA 3.0
- **Pipeline code:** MIT License
