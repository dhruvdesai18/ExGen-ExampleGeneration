import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/final/sampled_best_unique.parquet")
TRAIN_PATH = Path("data/final/train.parquet")
TEST_PATH = Path("data/final/test.parquet")
TRAIN_CSV = Path("data/final/train.csv")
TEST_CSV = Path("data/final/test.csv")

SEED = 42
TEST_RATIO = 0.20


def split_group(group, test_ratio, seed):
    test_n = max(1, round(len(group) * test_ratio))
    test = group.sample(n=test_n, random_state=seed)
    train = group.drop(test.index)
    return train, test


def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} rows\n")

    # Language column
    SOURCE_TO_LANG = {
        "wordnet": "en",
        "wiktionary": "en",
        "iitb": "hi",
        "ud_ukrainian": "uk",
    }
    df["language"] = df["source"].map(SOURCE_TO_LANG)

    # Normalise POS
    KEEP_POS = {"NOUN", "VERB", "ADJ", "ADV"}
    df["pos_group"] = df["pos"].apply(lambda p: p if p in KEEP_POS else "UNKNOWN")

    # Stratify by language + POS
    train_parts = []
    test_parts = []

    print("── Stratified split ──")
    for lang in ["en", "hi", "uk"]:
        lang_df = df[df["language"] == lang]
        for pos in sorted(lang_df["pos_group"].unique()):
            group = lang_df[lang_df["pos_group"] == pos]
            tr, te = split_group(group, TEST_RATIO, SEED)
            train_parts.append(tr)
            test_parts.append(te)
            print(
                f"  {lang} / {pos:8}: {len(group):4} total → "
                f"{len(tr):4} train + {len(te):3} test"
            )

    train = (
        pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    )
    test = (
        pd.concat(test_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    )

    train = train.drop(columns=["pos_group"])
    test = test.drop(columns=["pos_group"])

    # Verify no word leakage
    overlap = set(train["word"].str.lower()) & set(test["word"].str.lower())
    assert len(overlap) == 0, f"Word leakage: {overlap}"

    # Summary
    print(f"\n── Final Split ──")
    print(f"  Train : {len(train):,} rows")
    print(f"  Test  : {len(test):,} rows")
    print(f"  Ratio : {len(test)/len(df)*100:.1f}% test")
    print(f"  Word leakage: None ✓")

    print(f"\n── Train language breakdown ──")
    print(train["language"].value_counts().to_string())
    print(f"\n── Test language breakdown ──")
    print(test["language"].value_counts().to_string())
    print(f"\n── Train POS breakdown ──")
    print(train["pos"].value_counts().to_string())
    print(f"\n── Test POS breakdown ──")
    print(test["pos"].value_counts().to_string())

    has_def_train = (
        train["definition"].notna() & (train["definition"].str.strip() != "")
    ).sum()
    has_def_test = (
        test["definition"].notna() & (test["definition"].str.strip() != "")
    ).sum()
    print(f"\n── Definition coverage ──")
    print(
        f"  Train: {has_def_train}/{len(train)} ({100*has_def_train/len(train):.1f}%)"
    )
    print(f"  Test : {has_def_test}/{len(test)} ({100*has_def_test/len(test):.1f}%)")

    # Save
    print(f"\nSaving ...")
    train.to_parquet(TRAIN_PATH, index=False)
    train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    test.to_parquet(TEST_PATH, index=False)
    test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")
    print(f"  {TRAIN_PATH}  ({len(train):,} rows)")
    print(f"  {TEST_PATH}   ({len(test):,} rows)")
    print("Done!")


if __name__ == "__main__":
    main()
