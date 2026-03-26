"""
run_pipeline.py — Master script to run the entire English dataset pipeline.

Runs stages in order:
    1. Tatoeba
    2. Wiktionary      (skipped if dump not found — logs a warning)
    3. WordNet
    4. Leipzig
    5. Merge & Deduplicate
    6. Explore & Validate

Usage:
    # Full pipeline:
    python run_pipeline.py

    # Single stage:
    python run_pipeline.py --stage tatoeba
    python run_pipeline.py --stage wiktionary
    python run_pipeline.py --stage wordnet
    python run_pipeline.py --stage leipzig
    python run_pipeline.py --stage merge
    python run_pipeline.py --stage explore
"""

import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import WIKTIONARY_DUMP_PATH, LOG_DIR
from src.utils import get_logger

logger = get_logger("pipeline", LOG_DIR)


def run_stage(name: str, module_path: str) -> bool:
    """Import and run a collector module's main() function."""
    logger.info(f"\n{'═'*60}")
    logger.info(f" STAGE: {name.upper()}")
    logger.info(f"{'═'*60}")
    t0 = datetime.now()

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(f" ✓ {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = (datetime.now() - t0).total_seconds()
        logger.error(f" ✗ {name} FAILED after {elapsed:.1f}s: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Run English dataset pipeline.")
    parser.add_argument(
        "--stage",
        choices=["wiktionary", "wordnet", "leipzig", "hindi", "ukrainian", "merge", "explore", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    stages = {
        "wiktionary": str(base / "src/collect_wiktionary.py"),
        "wordnet":    str(base / "src/collect_wordnet.py"),
        "leipzig":    str(base / "src/collect_leipzig.py"),
        "hindi":      str(base / "src/collect_hindi.py"),
        "ukrainian":  str(base / "src/collect_ukrainian.py"),
        "merge":      str(base / "src/merge_and_deduplicate.py"),
        "explore":    str(base / "src/explore_dataset.py"),
    }

    run_order = ["wiktionary", "wordnet", "leipzig", "hindi", "ukrainian", "merge", "explore"]

    if args.stage != "all":
        run_order = [args.stage]

    t_total = datetime.now()
    results = {}

    for stage in run_order:
        # Special case: skip wiktionary if dump not downloaded
        if stage == "wiktionary" and not WIKTIONARY_DUMP_PATH.exists():
            logger.warning(
                f"\nSkipping Wiktionary: dump not found at {WIKTIONARY_DUMP_PATH}\n"
                f"To include it, download:\n"
                f"  wget https://dumps.wikimedia.org/enwiktionary/latest/"
                f"enwiktionary-latest-pages-articles.xml.bz2 -P data/raw/"
            )
            results[stage] = "skipped"
            continue

        success = run_stage(stage, stages[stage])
        results[stage] = "ok" if success else "failed"

        # Stop pipeline if a critical stage fails
        if not success and stage == "merge":
            logger.error(f"Critical stage '{stage}' failed. Stopping pipeline.")
            break

    total_elapsed = (datetime.now() - t_total).total_seconds()
    logger.info(f"\n{'═'*60}")
    logger.info(f" PIPELINE COMPLETE  ({total_elapsed:.0f}s total)")
    logger.info(f"{'═'*60}")
    for stage, result in results.items():
        icon = {"ok": "✓", "failed": "✗", "skipped": "○"}.get(result, "?")
        logger.info(f"  {icon} {stage:<20} {result}")


if __name__ == "__main__":
    main()
