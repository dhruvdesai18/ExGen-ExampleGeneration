from __future__ import annotations

import argparse

from exgen.datasets.loader import load_lexical_items


def run_inference(
    *,
    dataset_path: str,
    row_index: int,
    model_id: str,
    max_new_tokens: int,
) -> None:
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

    lexical_items = load_lexical_items(dataset_path)
    lexical_item = lexical_items[row_index]
    model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    tokenizer = MistralCommonBackend.from_pretrained(model_id)

    messages = [
        {
            "role": "system",
            "content": (
                "Generate a clear dictionary example sentence. "
                "Return exactly one sentence wrapped in <EXAMPLE> tags."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Language: {lexical_item.language}\n"
                f"Word: {lexical_item.word}\n"
                f"Part of speech: {lexical_item.pos}\n"
                f"Definition: {lexical_item.definition}\n"
                "Generate one natural sentence and return only <EXAMPLE>...</EXAMPLE>."
            ),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(model.device)
    input_ids = inputs["input_ids"][0]
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )[0]
    decoded = tokenizer.decode(output[len(input_ids) :], skip_special_tokens=True).strip()
    print(decoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="data/create_data/final_dataset/sampled_best_3000.csv",
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--model-id", default="Ministral-3-3B-Instruct-2512-bnb-nf4")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    run_inference(
        dataset_path=args.dataset_path,
        row_index=args.row_index,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
