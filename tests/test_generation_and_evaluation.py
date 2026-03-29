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
    def test_llm_json_parser_handles_trailing_commas(self) -> None:
        from exgen.evaluation.json_utils import parse_llm_json

        parsed = parse_llm_json(
            """```json
            {
              "preferred": "A",
              "confidence": "high",
              "reason": "Clearer sentence",
            }
            ```"""
        )

        self.assertEqual(parsed["preferred"], "A")
        self.assertEqual(parsed["confidence"], "high")

    def test_gemini_verifier_falls_back_when_thinking_is_unsupported(self) -> None:
        from exgen.evaluation.verifier import GeminiVerifier

        class UnsupportedThinkingError(Exception):
            pass

        class FakeModels:
            def __init__(self) -> None:
                self.calls = 0

            def generate_content(self, *, model, contents, config):
                self.calls += 1
                if self.calls == 1:
                    raise UnsupportedThinkingError(
                        "400 INVALID_ARGUMENT. Thinking level is not supported for this model."
                    )
                return SimpleNamespace(text='{"preferred":"A","confidence":"high","reason":"ok"}')

        fake_models = FakeModels()
        fake_client = SimpleNamespace(models=fake_models)

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            with patch("google.genai.Client", return_value=fake_client):
                verifier = GeminiVerifier(model_name="gemini-2.5-flash")
                raw = verifier.verify("system", "user")

        self.assertIn('"preferred":"A"', raw)
        self.assertEqual(fake_models.calls, 2)
        self.assertFalse(verifier._thinking_supported)

    def test_gemini_verifier_passes_response_schema(self) -> None:
        from exgen.evaluation.verifier import GeminiVerifier

        captured = {}

        class FakeModels:
            def generate_content(self, *, model, contents, config):
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                return SimpleNamespace(text='{"ok":true}')

        fake_client = SimpleNamespace(models=FakeModels())

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
            with patch("google.genai.Client", return_value=fake_client):
                verifier = GeminiVerifier(model_name="gemini-3-flash-preview")
                verifier.verify(
                    "system",
                    "user",
                    response_schema={
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                    },
                )

        self.assertEqual(captured["model"], "gemini-3-flash-preview")
        self.assertEqual(captured["contents"], "user")
        self.assertEqual(
            captured["config"].response_json_schema["properties"]["ok"]["type"],
            "boolean",
        )
        self.assertEqual(captured["config"].temperature, 0.0)

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

    def test_evaluation_pipeline_respects_max_items(self) -> None:
        from exgen.pipeline.run_evaluation import EvaluationPipeline
        from exgen.types import (
            ArtifactPaths,
            GenerationRecord,
            LexicalItemSnapshot,
            RunContext,
        )

        snapshot = LexicalItemSnapshot(
            id="lex-1",
            language="en",
            word="bank",
            pos="NOUN",
            definition="a financial institution",
            source="wordnet",
            reference_sentence="The bank opens at nine.",
            reference_sentence_id="sent-1",
        )
        records_in = []
        for language in ("en", "hi"):
            for index in range(3):
                records_in.append(
                    GenerationRecord(
                        run_id="run-1",
                        generated_example_id=f"{language}-gen-{index}",
                        lexical_item_id=f"{language}-lex-{index}",
                        lexical_item=snapshot.model_copy(
                            update={
                                "id": f"{language}-lex-{index}",
                                "language": language,
                                "word": f"{language}-word-{index}",
                                "reference_sentence_id": f"{language}-sent-{index}",
                            }
                        ),
                        example_text=f"{language} generated sentence {index}",
                        metadata={"selected": True},
                    )
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
                records = EvaluationPipeline().run(
                    config,
                    context,
                    generation_records=records_in,
                    max_items=2,
                )

            self.assertEqual(len(records), 4)
            self.assertEqual(
                [record.generated_example_id for record in records],
                ["en-gen-0", "en-gen-1", "hi-gen-0", "hi-gen-1"],
            )

    def test_evaluation_pipeline_adapts_flat_generation_rows(self) -> None:
        from exgen.pipeline.run_evaluation import EvaluationPipeline
        from exgen.types import ArtifactPaths, RunContext

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "eval_items.csv"
            dataset_path.write_text(
                "\n".join(
                    [
                        "word,sentence,source,pos,sense,sentence_id,word_freq_rank,sentence_count,language,definition",
                        "bank,The bank opens at nine.,wordnet,NOUN,,sent-1,100,5,en,a financial institution",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            generation_path = temp_path / "generated_examples.jsonl"
            generation_path.write_text(
                json.dumps(
                    {
                        "language": "en",
                        "word": "bank",
                        "pos": "NOUN",
                        "definition": "a financial institution",
                        "example": "The bank closes at five.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            context = RunContext(
                run_id="run-1",
                seed=42,
                dataset_path=dataset_path,
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
                records = EvaluationPipeline().run(
                    config,
                    context,
                    generation_path=generation_path,
                )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].result["seen_reference_sentence"], "The bank opens at nine.")

    def test_evaluation_pipeline_uses_generations_jsonl_by_default(self) -> None:
        from exgen.pipeline.run_evaluation import EvaluationPipeline
        from exgen.storage import persist_generation_record
        from exgen.types import ArtifactPaths, GenerationRecord, LexicalItemSnapshot, RunContext

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
            stale_snapshot = LexicalItemSnapshot(
                id="lex-stale",
                language="en",
                word="stale",
                pos="NOUN",
                definition="stale definition",
                source="wordnet",
                reference_sentence="Stale reference.",
                reference_sentence_id="sent-stale",
            )
            fresh_snapshot = LexicalItemSnapshot(
                id="lex-fresh",
                language="en",
                word="fresh",
                pos="NOUN",
                definition="fresh definition",
                source="wordnet",
                reference_sentence="Fresh reference.",
                reference_sentence_id="sent-fresh",
            )
            persist_generation_record(
                GenerationRecord(
                    run_id="run-1",
                    generated_example_id="stale-gen",
                    lexical_item_id="lex-stale",
                    lexical_item=stale_snapshot,
                    example_text="Stale sentence",
                    metadata={"selected": True},
                ),
                context.artifacts.selected_generation_path,
            )
            persist_generation_record(
                GenerationRecord(
                    run_id="run-1",
                    generated_example_id="fresh-gen",
                    lexical_item_id="lex-fresh",
                    lexical_item=fresh_snapshot,
                    example_text="Fresh sentence",
                    metadata={"selected": True},
                ),
                context.artifacts.generation_path,
            )
            config = SimpleNamespace(
                evaluations=[SimpleNamespace(name="fake-metric", prompt_version=None)]
            )

            with patch("exgen.pipeline.run_evaluation.create_evaluator", return_value=FakeEvaluator()):
                records = EvaluationPipeline().run(config, context)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].generated_example_id, "fresh-gen")
            self.assertEqual(records[0].result["seen_reference_sentence"], "Fresh reference.")

    def test_evaluation_pipeline_resumes_without_rewriting_completed_rows(self) -> None:
        from exgen.pipeline.run_evaluation import EvaluationPipeline
        from exgen.storage import load_evaluation_records, persist_evaluation_record
        from exgen.types import (
            ArtifactPaths,
            EvaluationRecord,
            GenerationRecord,
            LexicalItemSnapshot,
            RunContext,
        )

        snapshot = LexicalItemSnapshot(
            id="lex-1",
            language="en",
            word="bank",
            pos="NOUN",
            definition="a financial institution",
            source="wordnet",
            reference_sentence="The bank opens at nine.",
            reference_sentence_id="sent-1",
        )
        first = GenerationRecord(
            run_id="run-1",
            generated_example_id="gen-1",
            lexical_item_id="lex-1",
            lexical_item=snapshot,
            example_text="Generated sentence one",
            metadata={"selected": True},
        )
        second = GenerationRecord(
            run_id="run-1",
            generated_example_id="gen-2",
            lexical_item_id="lex-2",
            lexical_item=snapshot.model_copy(
                update={
                    "id": "lex-2",
                    "word": "loan",
                    "reference_sentence": "The loan was approved.",
                    "reference_sentence_id": "sent-2",
                }
            ),
            example_text="Generated sentence two",
            metadata={"selected": True},
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
            persist_evaluation_record(
                EvaluationRecord(
                    id="existing",
                    run_id="run-1",
                    metric_name="fake-metric",
                    metric_config_name="fake-metric",
                    language="en",
                    lexical_item_id="lex-1",
                    lexical_item=snapshot,
                    generated_example_id="gen-1",
                    score=1.0,
                    result={"seen_reference_sentence": "The bank opens at nine."},
                    metadata={"evaluated_word": "bank"},
                    provenance={"metric_name": "fake"},
                ),
                context.artifacts.evaluation_path,
            )
            config = SimpleNamespace(
                evaluations=[SimpleNamespace(name="fake-metric", prompt_version=None)]
            )

            with patch("exgen.pipeline.run_evaluation.create_evaluator", return_value=FakeEvaluator()):
                records = EvaluationPipeline().run(
                    config,
                    context,
                    generation_records=[first, second],
                    resume=True,
                )

            self.assertEqual(len(records), 2)
            self.assertEqual(
                [record.generated_example_id for record in records],
                ["gen-1", "gen-2"],
            )
            persisted = load_evaluation_records(context.artifacts.evaluation_path)
            self.assertEqual(len(persisted), 2)
            self.assertEqual(
                [record.generated_example_id for record in persisted],
                ["gen-1", "gen-2"],
            )

    def test_run_evaluation_accepts_dataset_override_for_flat_rows(self) -> None:
        from exgen.pipeline.run_evaluation import run_evaluation
        from exgen.types import ArtifactPaths, RunContext

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_dataset_path = temp_path / "eval_items.csv"
            test_dataset_path = temp_path / "test_items.csv"
            generations_path = temp_path / "generated_examples.jsonl"
            eval_dataset_path.write_text(
                "\n".join(
                    [
                        "word,sentence,source,pos,sense,sentence_id,word_freq_rank,sentence_count,language,definition",
                        "river,The river is wide.,wordnet,NOUN,,sent-eval,10,4,en,a natural watercourse",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            test_dataset_path.write_text(
                "\n".join(
                    [
                        "word,sentence,source,pos,sense,sentence_id,word_freq_rank,sentence_count,language,definition",
                        "bank,The bank opens at nine.,wordnet,NOUN,,sent-test,100,5,en,a financial institution",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            generations_path.write_text(
                json.dumps(
                    {
                        "language": "en",
                        "word": "bank",
                        "pos": "NOUN",
                        "definition": "a financial institution",
                        "example": "The bank closes at five.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = SimpleNamespace(
                evaluations=[
                    SimpleNamespace(name="oxfordeval", prompt_version=None),
                    SimpleNamespace(name="gdex", prompt_version=None),
                ]
            )
            context = RunContext(
                run_id="test-run",
                seed=42,
                dataset_path=eval_dataset_path,
                language="multilingual",
                versions={},
                artifacts=ArtifactPaths(
                    root_dir=temp_path / "outputs",
                    generation_path=temp_path / "outputs" / "generations.jsonl",
                    selected_generation_path=temp_path / "outputs" / "selected_generations.jsonl",
                    evaluation_path=temp_path / "outputs" / "evaluations.jsonl",
                    manifest_path=temp_path / "outputs" / "manifest.json",
                ),
            )

            with patch("exgen.pipeline.run_evaluation.load_run_config", return_value=config):
                with patch("exgen.pipeline.run_evaluation.build_run_context", return_value=context):
                    with patch("exgen.pipeline.run_evaluation.create_evaluator", return_value=FakeEvaluator()):
                        run_evaluation(
                            "ignored.yaml",
                            generation_path=generations_path,
                            dataset_path=test_dataset_path,
                        )

            evaluations_path = temp_path / "outputs" / "evaluations.jsonl"
            self.assertTrue(evaluations_path.exists())
            records = [json.loads(line) for line in evaluations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 2)

    def test_oxford_eval_uses_reference_sentence(self) -> None:
        from exgen.config import OxfordEvaluationConfig
        from exgen.evaluation.oxfordeval import OxfordEvalEvaluator
        from exgen.types import GeneratedExample, LexicalItem

        class FakeVerifier:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def verify(
                self,
                system_prompt: str,
                user_prompt: str,
                response_schema: dict | None = None,
            ) -> str:
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

    def test_oxford_eval_order_is_deterministic_per_lexical_item(self) -> None:
        from exgen.config import OxfordEvaluationConfig
        from exgen.evaluation.oxfordeval import OxfordEvalEvaluator
        from exgen.types import GeneratedExample, LexicalItem

        class FakeVerifier:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def verify(
                self,
                system_prompt: str,
                user_prompt: str,
                response_schema: dict | None = None,
            ) -> str:
                return json.dumps(
                    {
                        "preferred": "A",
                        "confidence": "high",
                        "reason": "Stable choice.",
                    }
                )

        lexical_item = LexicalItem(
            id="lex-stable",
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
            randomize_order=True,
            judge_model_name="fake-judge",
        )

        with patch("exgen.evaluation.oxfordeval.GeminiVerifier", FakeVerifier):
            evaluator = OxfordEvalEvaluator(config)
            outcome_one = evaluator.evaluate(example)
            outcome_two = evaluator.evaluate(example)

        self.assertEqual(
            outcome_one.metadata["generated_first"],
            outcome_two.metadata["generated_first"],
        )
        self.assertEqual(outcome_one.metadata["sentence_a"], outcome_two.metadata["sentence_a"])
        self.assertEqual(outcome_one.metadata["sentence_b"], outcome_two.metadata["sentence_b"])


if __name__ == "__main__":
    unittest.main()
