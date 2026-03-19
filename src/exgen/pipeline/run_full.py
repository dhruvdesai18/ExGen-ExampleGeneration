from __future__ import annotations

import argparse

from exgen.config import build_run_context, load_run_config
from exgen.datasets.loader import load_lexical_items
from exgen.pipeline.full_pipeline import FullPipeline


def run_full(run_config: str) -> None:
    config = load_run_config(run_config)
    context = build_run_context(config)
    lexical_items = load_lexical_items(context.dataset_path)
    manifest = FullPipeline().run(config, context, lexical_items)
    print(f"Wrote manifest to {context.artifacts.manifest_path}")
    print(manifest.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-config",
        help="Run config version key or path.",
    )
    args = parser.parse_args()
    run_full(args.run_config)


if __name__ == "__main__":
    main()
