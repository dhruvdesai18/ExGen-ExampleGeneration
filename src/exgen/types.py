from typing import Any, Literal
from pydantic import BaseModel, Field

class LexicalItem(BaseModel):
    id: str
    language: str
    word: str
    pos: str
    definition: str
    source: str

class GeneratedExample(BaseModel):
    id: str
    # request_id: str
    lexical_item: LexicalItem
    example_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    quality_scores: dict[str,float] = Field(default_factory=dict)

class RetrievedExample(BaseModel):
    id: str
    lexical_item: LexicalItem
    example_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_name: str
    source_url: str

class GenerationRequest(BaseModel):
    id: str
    lexical_item: LexicalItem
    generation_type: Literal["baseline", "augmented"]
    run_id: str

class EvaluationRecord(BaseModel):
    id: str
    metric_name: str
    lexical_item_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
