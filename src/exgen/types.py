from datetime import datetime, timezone
from pathlib import Path
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

class LexicalItemSnapshot(BaseModel):
    language: str
    word: str
    pos: str
    definition: str

class ArtifactPaths(BaseModel):
    root_dir: Path
    generation_path: Path
    evaluation_path: Path
    manifest_path: Path


class RunContext(BaseModel):
    run_id: str
    seed: int
    dataset_path: Path
    language: str
    versions: dict[str, Any] = Field(default_factory=dict)
    artifacts: ArtifactPaths


class GenerationRecord(BaseModel):
    run_id: str
    generated_example_id: str
    lexical_item_id: str
    lexical_item: LexicalItemSnapshot
    example_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationOutcome(BaseModel):
    score: float | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    id: str
    run_id: str
    metric_name: str
    metric_config_name: str | None = None
    metric_prompt_version: str | None = None
    language: str
    lexical_item_id: str
    lexical_item: LexicalItemSnapshot
    generated_example_id: str
    score: float | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunManifest(BaseModel):
    run_id: str
    pipeline_mode: Literal["full", "generation", "evaluation"]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset: dict[str, Any]
    versions: dict[str, Any]
    configs: dict[str, Any]
    artifacts: ArtifactPaths
