from __future__ import annotations

from typing import Any


class MaskedLMRanker:
    def __init__(self, model_name: str):
        self.model_name = model_name

        import numpy as np
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._np = np
        self._torch = torch
        self._log_softmax = F.log_softmax
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.mask_token = self.tokenizer.mask_token

    def get_exemplification_score(self, sentence: str, target_word: str) -> dict[str, Any]:
        candidate_forms = {
            target_word,
            target_word.lower(),
            target_word.capitalize(),
            f" {target_word}",
            f" {target_word.lower()}",
            f" {target_word.capitalize()}",
        }

        target_variants: list[tuple[list[str], list[int]]] = []
        seen_id_tuples: set[tuple[int, ...]] = set()
        for form in candidate_forms:
            tokens = self.tokenizer.tokenize(form)
            ids = tuple(self.tokenizer.convert_tokens_to_ids(tokens))
            if ids not in seen_id_tuples:
                seen_id_tuples.add(ids)
                target_variants.append((tokens, list(ids)))

        inputs = self.tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"][0]

        occurrences: list[tuple[list[int], list[int]]] = []
        for _, token_ids in target_variants:
            token_count = len(token_ids)
            for start_index in range(len(input_ids) - token_count + 1):
                if input_ids[start_index : start_index + token_count].tolist() == token_ids:
                    occurrences.append((list(range(start_index, start_index + token_count)), token_ids))

        if not occurrences:
            return {
                "status": "failure",
                "reason": f"target_word '{target_word}' not found in the tokenized sentence.",
            }

        masked_input_ids = input_ids.clone()
        for token_indexes, _ in occurrences:
            for token_index in token_indexes:
                masked_input_ids[token_index] = self.tokenizer.mask_token_id

        with self._torch.no_grad():
            outputs = self.model(
                masked_input_ids.unsqueeze(0),
                attention_mask=inputs["attention_mask"],
            )
            logits = outputs.logits[0]

        log_probs = self._log_softmax(logits, dim=-1)
        total_log_prob = 0.0
        total_tokens_evaluated = 0
        for token_indexes, token_ids in occurrences:
            for offset, token_index in enumerate(token_indexes):
                target_token_id = token_ids[offset]
                total_log_prob += log_probs[token_index, target_token_id].item()
                total_tokens_evaluated += 1

        mean_log_prob = total_log_prob / total_tokens_evaluated
        return {
            "status": "SUCCESS",
            "score": float(self._np.exp(mean_log_prob)),
            "log_prob": float(mean_log_prob),
        }
