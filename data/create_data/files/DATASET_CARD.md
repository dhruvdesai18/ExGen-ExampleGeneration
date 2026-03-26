---
language:
- en
- hi
- uk
license: cc-by-sa-4.0
task_categories:
- text-generation
- conditional-text-generation
task_ids:
- language-modeling
pretty_name: Multilingual Word-Example Dataset
size_categories:
- 1K<n<10K
tags:
- language-learning
- example-generation
- multilingual
- wordnet
- universal-dependencies
---

# Multilingual Word-Example Dataset

## Dataset Description

A multilingual dataset of **(word, definition, example sentence)** triples for training language learning models. The dataset covers three typologically diverse languages — English, Hindi, and Ukrainian — and is designed for the task of **controlled example sentence generation**: given a target word, generate a natural example sentence that uses the word correctly in context.

### Languages

| Language | Script | Family | Pairs |
|---|---|---|---|
| English | Latin | Indo-European / Germanic | ~2,000 |
| Hindi | Devanagari | Indo-European / Indo-Aryan | ~1,000 |
| Ukrainian | Cyrillic | Indo-European / Slavic | ~1,000 |

---

## Dataset Files

| File | Rows | Description |
|---|---|---|
| `sampled_best_3000.parquet` | 3,000 | Primary training dataset (1,000/language) |
| `sampled_best_unique.parquet` | 2,620 | No duplicate target words |
| `sampled_6000_enriched_hindi_cleaned.parquet` | 6,000 | Full enriched dataset |
| `english_word_examples.parquet` | 84,465 | Complete raw dataset (all sources) |

---

## Dataset Structure

### Data Fields

```python
{
    "word":        str,   # Target word (lemma form)
    "sentence":    str,   # Example sentence
    "source":      str,   # Data source (wordnet/wiktionary/iitb/ud_ukrainian)
    "pos":         str,   # Part of speech (NOUN/VERB/ADJ/ADV/UNKNOWN)
    "definition":  str,   # Word definition
    "language":    str,   # Language code (en/hi/uk)
    "sentence_id": str,   # MD5 fingerprint of sentence
}
```

### Example Rows

```python
# English
{
    "word": "run",
    "sentence": "She runs every morning to stay fit.",
    "source": "wordnet",
    "pos": "VERB",
    "definition": "Move fast by using one's feet.",
    "language": "en"
}

# Hindi
{
    "word": "किताब",
    "sentence": "मैंने पुस्तकालय से एक किताब उधार ली।",
    "source": "iitb",
    "pos": "NOUN",
    "definition": "पुस्तक; ग्रंथ",
    "language": "hi"
}

# Ukrainian
{
    "word": "бігти",
    "sentence": "Він біжить щодня вранці.",
    "source": "ud_ukrainian",
    "pos": "VERB",
    "definition": "to run, to move fast on foot",
    "language": "uk"
}
```

---

## Data Sources

### English
- **Princeton WordNet** — Sense-disambiguated examples written by lexicographers. Gold standard for English lexical semantics.
- **English Wiktionary** — Purpose-written examples from collaborative dictionary.

### Hindi
- **IIT Bombay English-Hindi Parallel Corpus (IITB)** — The largest clean Hindi parallel corpus. Only the Hindi side used. Sentences verified through parallel alignment.
- **Hindi Wiktionary** — Very sparse, minimal contribution.

### Ukrainian
- **Universal Dependencies Ukrainian-IU** — Hand-annotated gold treebank. Provides gold lemmas and Universal POS tags directly. Covers fiction, news, Wikipedia, legal documents.

---

## Data Collection Pipeline

```
Raw sources → Quality filters → Deduplication → 50/word cap → Merge → Enrich
```

**Quality filters applied:**
- Sentence length: 5–35 tokens
- Character count: 15–300 chars
- Alpha ratio: ≥75%
- Script ratio: ≥60% Devanagari (Hindi), ≥60% Cyrillic (Ukrainian)
- Language confidence: ≥90% (English, via langdetect)

**Enrichment:**
- Definitions: Wiktionary XML dumps (local, offline)
- English fallback: WordNet glosses
- Ukrainian definitions: from English Wiktionary `==Ukrainian==` sections (in English)
- Hindi POS: Stanza `hdtb` model

---

## Intended Use

**Primary use:** Fine-tuning multilingual sequence-to-sequence models for example sentence generation.

**Suggested input format:**
```
[lang=en] word: run → <generated sentence>
[lang=hi] word: किताब → <generated sentence>
[lang=uk] word: бігти → <generated sentence>
```

**Recommended models:** mT5, mBART, NLLB, or any multilingual encoder-decoder.

---

## Limitations

- **Hindi definitions** come from 19th century शब्दसागर dictionary — verbose and old-fashioned style. 33% of Hindi words have no definition.
- **Ukrainian definitions** are in English, not Ukrainian.
- **word_not_in_sentence** is flagged for 31% of rows — this is expected for Hindi and Ukrainian due to morphological inflection. The stored word is the lemma; the sentence contains the inflected surface form.
- **Hindi POS** tagged by Stanza model, not gold standard.
- **Ukrainian vocabulary** limited to ~17,000 unique words.
- **Class imbalance** in raw dataset — Ukrainian has more pairs (54.7%) due to richer UD annotations.

---

## Citation

```bibtex
@dataset{multilingual_word_examples_2026,
  title     = {Multilingual Word-Example Dataset for Language Learning},
  author    = {MSc AI Student, National University of Ireland Galway},
  year      = {2026},
  language  = {en, hi, uk},
  note      = {MSc Artificial Intelligence Project}
}
```

---

## License

The dataset is a compilation from multiple sources with different licenses:

| Source | License |
|---|---|
| WordNet | Princeton WordNet License (free for research) |
| IIT Bombay Corpus | Free for research use |
| Universal Dependencies Ukrainian-IU | CC BY-NC-SA 4.0 |
| Wiktionary | CC BY-SA 3.0 |
