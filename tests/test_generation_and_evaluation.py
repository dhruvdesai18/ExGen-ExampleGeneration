from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class FakeRankerBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_exemplification_score(self, sentence: str, target_word: str) -> dict[str, object]:
        if "best" in sentence:
            return {"status": "SUCCESS", "score": 0.9, "log_prob": -0.1}
        if "good" in sentence:
            return {"status": "SUCCESS", "score": 0.5, "log_prob": -0.5}
        return {"status": "failure", "reason": f"{target_word} missing"}


class FakeGenerator:
    def generate(self, lexical_item):
        from exgen.types import GeneratedExample

        return [
            GeneratedExample(
                id=f"gen-{index}",
                lexical_item=lexical_item,
                example_text=text,
                metadata={"model_name": "fake", "generator_type": "augmented"},
            )
            for index, text in enumerate(
                [
                    "plain fallback sentence",
                    "good candidate sentence",
                    "best candidate sentence",
                    "another sentence",
                    "last sentence",
                ]
            )
        ]


class FakeEvaluator:
    def evaluate(self, example):
        from exgen.types import EvaluationOutcome

        return EvaluationOutcome(
            score=1.0,
            result={"seen_reference_sentence": example.lexical_item.reference_sentence},
            metadata={"evaluated_word": example.lexical_item.word},
            provenance={"metric_name": "fake"},
        )


class RankingAndPipelineTests(unittest.TestCase):
    def test_mlm_ranker_marks_single_selected_candidate(self) -> None:
        from exgen.generation.ranking import MLMCandidateRanker
        from exgen.types import GeneratedExample, LexicalItem

        lexical_item = LexicalItem(
            id="lex-1",
            language="en",
            word="bank",
            pos="NOUN",
            definition="a financial institution",
            source="wordnet",
        )
        examples = [
            GeneratedExample(id="1", lexical_item=lexical_item, example_text="plain sentence"),
            GeneratedExample(id="2", lexical_item=lexical_item, example_text="good sentence"),
            GeneratedExample(id="3", lexical_item=lexical_item, example_text="best sentence"),
        ]

        ranker = MLMCandidateRanker(ranker_factory=FakeRankerBackend)
        annotated = ranker.annotate(lexical_item, examples)

        selected_indexes = [
            example.metadata["candidate_index"]
            for example in annotated
            if example.metadata["selected"] is True
        ]
        self.assertEqual(selected_indexes, [2])
        self.assertFalse(annotated[2].metadata["mlm_fallback"])

    def test_generation_pipeline_persists_all_and_selected_records(self) -> None:
        from exgen.pipeline.run_generation import GenerationPipeline
        from exgen.storage import load_generation_records
        from exgen.types import ArtifactPaths, LexicalItem, RunContext

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            context = RunContext(
                run_id="run-1",
                seed=42,
                dataset_path=temp_path / "dataset.parquet",
                language="multilingual",
                versions={},
                artifacts=ArtifactPaths(
                    root_dir=temp_path,
                    generation_path=temp_path / "generations.jsonl",
                    selected_generation_path=temp_path / "selected_generations.jsonl",
                    evaluation_path=temp_path / "evaluations.jsonl",
                    manifest_path=temp_path / "manifest.json",
                ),
            )
            lexical_item = LexicalItem(
                id="lex-1",
                language="en",
                word="bank",
                pos="NOUN",
                definition="a financial institution",
                source="wordnet",
                reference_sentence="The bank opens at nine.",
                reference_sentence_id="sent-1",
            )
            config = SimpleNamespace(model=None, retrieval=None, generator=None)

            pipeline = GenerationPipeline()
            pipeline.ranker = SimpleNamespace(
                annotate=lambda lexical_item, examples: [
                    example.model_copy(
                        update={
                            "metadata": {
                                **example.metadata,
                                "candidate_index": index,
                                "selected": index == 2,
                                "mlm_fallback": False,
                            }
                        }
                    )
                    for index, example in enumerate(examples)
                ]
            )

            with patch("exgen.pipeline.run_generation.create_model", return_value=object()):
                with patch("exgen.pipeline.run_generation.create_fetcher", return_value=object()):
                    with patch(
                        "exgen.pipeline.run_generation.create_generator",
                        return_value=FakeGenerator(),
                    ):
                        records = pipeline.run(config, context, [lexical_item])

            self.assertEqual(len(records), 5)
            self.assertEqual(len(load_generation_records(context.artifacts.generation_path)), 5)
            selected_records = load_generation_records(context.artifacts.selected_generation_path)
            self.assertEqual(len(selected_records), 1)
            self.assertEqual(selected_records[0].metadata["candidate_index"], 2)

    def test_evaluation_pipeline_uses_selected_rows_only(self) -> None:
        from exgen.pipeline.run_evaluation import EvaluationPipeline
        from exgen.types import (
            ArtifactPaths,
            EvaluationRecord,
            GenerationRecord,
            LexicalItemSnapshot,
            RunContext,
        )

        lexical_snapshot = LexicalItemSnapshot(
            id="lex-1",
            language="en",
            word="bank",
            pos="NOUN",
            definition="a financial institution",
            source="wordnet",
            reference_sentence="The bank opens at nine.",
            reference_sentence_id="sent-1",
        )
        selected = GenerationRecord(
            run_id="run-1",
            generated_example_id="gen-1",
            lexical_item_id="lex-1",
            lexical_item=lexical_snapshot,
            example_text="Generated best sentence",
            metadata={"selected": True},
        )
        rejected = GenerationRecord(
            run_id="run-1",
            generated_example_id="gen-2",
            lexical_item_id="lex-1",
            lexical_item=lexical_snapshot,
            example_text="Generated rejected sentence",
            metadata={"selected": False},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            context = RunContext(
                run_id="run-1",
                seed=42,
                dataset_path=temp_path / "dataset.parquet",
                language="multilingual",
                versions={},
                artifacts=ArtifactPaths(
                    root_dir=temp_path,
                    generation_path=temp_path / "generations.jsonl",
                    selected_generation_path=temp_path / "selected_generations.jsonl",
                    evaluation_path=temp_path / "evaluations.jsonl",
                    manifest_path=temp_path / "manifest.json",
                ),
            )
            config = SimpleNamespace(
                evaluations=[SimpleNamespace(name="fake-metric", prompt_version=None)]
            )

            with patch("exgen.pipeline.run_evaluation.create_evaluator", return_value=FakeEvaluator()):
                records = EvaluationPipeline().run(config, context, [selected, rejected])

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].generated_example_id, "gen-1")
            self.assertEqual(records[0].result["seen_reference_sentence"], "The bank opens at nine.")

    def test_oxford_eval_uses_reference_sentence(self) -> None:
        from exgen.config import OxfordEvaluationConfig
        from exgen.evaluation.oxfordeval import OxfordEvalEvaluator
        from exgen.types import GeneratedExample, LexicalItem

        class FakeVerifier:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def verify(self, system_prompt: str, user_prompt: str) -> str:
                return json.dumps(
                    {
                        "preferred": "A",
                        "confidence": "high",
                        "reason": "Generated sentence is clearer.",
                    }
                )

        lexical_item = LexicalItem(
            id="lex-1",
            language="en",
            word="bank",
            pos="NOUN",
            definition="a financial institution",
            source="wordnet",
            reference_sentence="The bank opens at nine.",
            reference_sentence_id="sent-1",
        )
        example = GeneratedExample(
            id="gen-1",
            lexical_item=lexical_item,
            example_text="The bank closes at five.",
        )
        config = OxfordEvaluationConfig(
            name="oxfordeval",
            prompt_version="oxfordeval_1",
            randomize_order=False,
            judge_model_name="fake-judge",
        )

        with patch("exgen.evaluation.oxfordeval.GeminiVerifier", FakeVerifier):
            evaluator = OxfordEvalEvaluator(config)
            outcome = evaluator.evaluate(example)

        self.assertEqual(outcome.metadata["sentence_a"], "The bank closes at five.")
        self.assertEqual(outcome.metadata["sentence_b"], "The bank opens at nine.")
        self.assertEqual(outcome.metadata["reference_sentence_id"], "sent-1")


if __name__ == "__main__":
    unittest.main()
