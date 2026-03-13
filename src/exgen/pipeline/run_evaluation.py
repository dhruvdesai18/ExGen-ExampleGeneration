from __future__ import annotations

from src.exgen.evaluation.oxfordeval import OxfordEvalEvaluator
from src.exgen.evaluation.verifier import GeminiVerifier
from src.exgen.types import GeneratedExample, LexicalItem


def build_mock_generated_example() -> GeneratedExample:
    lexical_item = LexicalItem(
        id="lex-001",
        language="english",
        word="balance",
        pos="verb",
        definition="to keep or put something in a steady position so that it does not fall",
        source="mock",
    )
    return GeneratedExample(
        id="gen-001",
        lexical_item=lexical_item,
        example_text="She balanced the tray carefully while walking to the table.",
        metadata={
            "model_name": "mock-generator",
            "generator_type": "baseline",
        },
    )


def run_evaluation() -> None:
    verifier = GeminiVerifier()
    evaluator = OxfordEvalEvaluator(verifier=verifier)

    generated_example = build_mock_generated_example()
    record = evaluator.evaluate(generated_example)

    print(record.model_dump_json(indent=2))


if __name__ == "__main__":
    run_evaluation()
