from __future__ import annotations

import argparse

from exgen.datasets.loader import load_lexical_items
from exgen.generation.baseline import BaselineGenerator
from exgen.models.ministral_qft import MinistralQFT


def run_inference(
    *,
    dataset_path: str,
    row_index: int,
    model_id: str,
    adapter_path: str,
    num_candidates: int,
) -> None:
    lexical_items = load_lexical_items(dataset_path)
    lexical_item = lexical_items[row_index]
    model = MinistralQFT(
        name="ministral_qft",
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
        model_id=model_id,
        adapter_path=adapter_path,
    )
    generator = BaselineGenerator(
        model=model,
        num_candidates=num_candidates,
        prompt_version="baseline_v1",
    )
    print(
        f"Target: {lexical_item.language} | {lexical_item.word} | {lexical_item.pos} | "
        f"{lexical_item.definition}"
    )
    for example in generator.generate(lexical_item):
        print(example.example_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="data/create_data/final_dataset/sampled_best_3000.csv",
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--model-id", default="Ministral-3-3B-Instruct-2512-bnb-nf4")
    parser.add_argument("--adapter-path", default="exgen-qlora-adapter")
    parser.add_argument("--num-candidates", type=int, default=1)
    args = parser.parse_args()
    run_inference(
        dataset_path=args.dataset_path,
        row_index=args.row_index,
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        num_candidates=args.num_candidates,
    )


if __name__ == "__main__":
    main()
