from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HAS_DATA_DEPS = (
    importlib.util.find_spec("pandas") is not None
    and importlib.util.find_spec("pyarrow") is not None
)


@unittest.skipUnless(HAS_DATA_DEPS, "pandas and pyarrow are required for split tests")
class SplitArtifactsTest(unittest.TestCase):
    def test_build_split_artifacts_creates_balanced_outputs(self) -> None:
        from exgen.datasets.splits import build_split_artifacts

        import pandas as pd

        rows = []
        for language, source in (("en", "wordnet"), ("hi", "iitb"), ("uk", "ud_ukrainian")):
            rows.extend(
                [
                    {
                        "word": f"{language}_alpha",
                        "sentence": f"{language} alpha sentence 1",
                        "source": source,
                        "pos": "NOUN",
                        "sense": "",
                        "sentence_id": f"{language}-alpha-1",
                        "word_freq_rank": 10,
                        "sentence_count": 5,
                        "language": language,
                        "definition": f"{language} alpha definition",
                    },
                    {
                        "word": f"{language}_alpha",
                        "sentence": f"{language} alpha sentence 2",
                        "source": source,
                        "pos": "NOUN",
                        "sense": "",
                        "sentence_id": f"{language}-alpha-2",
                        "word_freq_rank": 10,
                        "sentence_count": 4,
                        "language": language,
                        "definition": f"{language} alpha definition",
                    },
                    {
                        "word": f"{language}_beta",
                        "sentence": f"{language} beta sentence 1",
                        "source": source,
                        "pos": "VERB",
                        "sense": "",
                        "sentence_id": f"{language}-beta-1",
                        "word_freq_rank": 20,
                        "sentence_count": 3,
                        "language": language,
                        "definition": f"{language} beta definition",
                    },
                    {
                        "word": f"{language}_beta",
                        "sentence": f"{language} beta sentence 2",
                        "source": source,
                        "pos": "VERB",
                        "sense": "",
                        "sentence_id": f"{language}-beta-2",
                        "word_freq_rank": 20,
                        "sentence_count": 2,
                        "language": language,
                        "definition": f"{language} beta definition",
                    },
                ]
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "fixture.csv"
            with source_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            artifact_paths = build_split_artifacts(
                source_path=source_path,
                output_dir=temp_path,
                seed=7,
                eval_items_per_language=1,
                test_items_per_language=1,
            )

            train_examples = pd.read_parquet(artifact_paths["train_examples"])
            eval_examples = pd.read_parquet(artifact_paths["eval_examples"])
            test_examples = pd.read_parquet(artifact_paths["test_examples"])
            eval_items = pd.read_parquet(artifact_paths["eval_items"])
            test_items = pd.read_parquet(artifact_paths["test_items"])

            self.assertEqual(len(train_examples), 0)
            self.assertEqual(
                eval_examples.groupby("language").size().to_dict(),
                {"en": 2, "hi": 2, "uk": 2},
            )
            self.assertEqual(
                test_examples.groupby("language").size().to_dict(),
                {"en": 2, "hi": 2, "uk": 2},
            )
            self.assertEqual(
                eval_items.groupby("language").size().to_dict(),
                {"en": 1, "hi": 1, "uk": 1},
            )
            self.assertEqual(
                test_items.groupby("language").size().to_dict(),
                {"en": 1, "hi": 1, "uk": 1},
            )

            self.assertTrue(
                set(eval_examples["lexical_item_id"]).isdisjoint(set(test_examples["lexical_item_id"]))
            )

            for lexical_item_id, group in test_examples.groupby("lexical_item_id"):
                first_sentence = group.iloc[0]["sentence"]
                item_row = test_items[test_items["lexical_item_id"] == lexical_item_id].iloc[0]
                self.assertEqual(item_row["reference_sentence"], first_sentence)


if __name__ == "__main__":
    unittest.main()
