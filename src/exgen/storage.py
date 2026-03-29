from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from exgen.config import resolve_project_path
from exgen.types import EvaluationRecord, GenerationRecord, RunManifest


def append_jsonl(path: str | Path, record: BaseModel) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return resolved


def persist_generation_record(record: GenerationRecord, output_path: str | Path) -> Path:
    return append_jsonl(output_path, record)


def persist_evaluation_record(
    record: EvaluationRecord,
    output_path: str | Path,
) -> Path:
    return append_jsonl(output_path, record)


def write_manifest(manifest: RunManifest, output_path: str | Path) -> Path:
    resolved = resolve_project_path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(manifest.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return resolved


def load_generation_records(path: str | Path) -> list[GenerationRecord]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8") as handle:
        return [
            GenerationRecord.model_validate(json.loads(line))
            for line in handle
            if line.strip()
        ]


def load_evaluation_records(path: str | Path) -> list[EvaluationRecord]:
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return []
    with resolved.open("r", encoding="utf-8") as handle:
        return [
            EvaluationRecord.model_validate(json.loads(line))
            for line in handle
            if line.strip()
        ]


def reset_output_file(path: str | Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("", encoding="utf-8")
    return resolved
