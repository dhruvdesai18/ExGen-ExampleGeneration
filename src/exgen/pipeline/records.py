from __future__ import annotations

import uuid

from exgen.config import BaseEvaluationConfig
from exgen.types import (
    EvaluationOutcome,
    EvaluationRecord,
    GeneratedExample,
    GenerationRecord,
    LexicalItem,
    LexicalItemSnapshot,
    RunContext,
)


def lexical_item_snapshot(lexical_item: LexicalItem) -> LexicalItemSnapshot:
    """Build a compact lexical snapshot for persistence and analysis."""
    return LexicalItemSnapshot(
        id=lexical_item.id,
        language=lexical_item.language,
        word=lexical_item.word,
        pos=lexical_item.pos,
        definition=lexical_item.definition,
        source=lexical_item.source,
        sense=lexical_item.sense,
        reference_sentence=lexical_item.reference_sentence,
        reference_sentence_id=lexical_item.reference_sentence_id,
        word_freq_rank=lexical_item.word_freq_rank,
        sentence_count=lexical_item.sentence_count,
    )


def build_generation_record(
    context: RunContext,
    example: GeneratedExample,
) -> GenerationRecord:
    """Normalize a generated example into the persisted generation row shape."""
    return GenerationRecord(
        run_id=context.run_id,
        generated_example_id=example.id,
        lexical_item_id=example.lexical_item.id,
        lexical_item=lexical_item_snapshot(example.lexical_item),
        example_text=example.example_text,
        metadata=example.metadata,
    )


def generation_record_to_example(record: GenerationRecord) -> GeneratedExample:
    """Rehydrate a persisted generation row back into the in-memory example shape."""
    lexical_item = LexicalItem(
        id=record.lexical_item.id or record.lexical_item_id,
        language=record.lexical_item.language,
        word=record.lexical_item.word,
        pos=record.lexical_item.pos,
        definition=record.lexical_item.definition,
        source=record.lexical_item.source,
        sense=record.lexical_item.sense,
        reference_sentence=record.lexical_item.reference_sentence,
        reference_sentence_id=record.lexical_item.reference_sentence_id,
        word_freq_rank=record.lexical_item.word_freq_rank,
        sentence_count=record.lexical_item.sentence_count,
    )
    return GeneratedExample(
        id=record.generated_example_id,
        lexical_item=lexical_item,
        example_text=record.example_text,
        metadata=record.metadata,
    )


def build_evaluation_record(
    context: RunContext,
    evaluation_config: BaseEvaluationConfig,
    generation_record: GenerationRecord,
    outcome: EvaluationOutcome,
) -> EvaluationRecord:
    """Combine run context, generation linkage, and evaluator output into one normalized result row."""
    return EvaluationRecord(
        id=str(uuid.uuid4()),
        run_id=context.run_id,
        metric_name=evaluation_config.name,
        metric_config_name=evaluation_config.name,
        metric_prompt_version=evaluation_config.prompt_version,
        language=generation_record.lexical_item.language,
        lexical_item_id=generation_record.lexical_item_id,
        lexical_item=generation_record.lexical_item,
        generated_example_id=generation_record.generated_example_id,
        score=outcome.score,
        result=outcome.result,
        metadata=outcome.metadata,
        provenance=outcome.provenance,
    )
