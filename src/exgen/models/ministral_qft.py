from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from .base import Model


@dataclass
class MinistralQFT(Model):
    model_id: str
    adapter_path: str = "exgen-qlora-adapter"
    _model: Any | None = field(default=None, init=False, repr=False)
    _tokeniser: Any | None = field(default=None, init=False, repr=False)

    def _ensure_loaded(self):
        if self._model is not None and self._tokeniser is not None:
            return

        from peft import PeftModel
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

        # Load base quantised model
        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto"
        )
        # Load trained LoRA adapter and tokeniser
        self._model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self._tokeniser = MistralCommonBackend.from_pretrained(self.adapter_path)

    def generate(
        self,
        messages: list[dict[str, str]],
        num_candidates: int,
    ) -> Iterator[str]:
        if num_candidates < 1:
            return
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokeniser is not None

        for _ in range(num_candidates):
            inputs = self._tokeniser.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).to(self._model.device)
            input_ids = inputs["input_ids"][0]
            do_sample = self.temperature > 0
            generation_kwargs = {
                "max_new_tokens": self.max_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_kwargs["temperature"] = self.temperature
                generation_kwargs["top_p"] = self.top_p
            output = self._model.generate(**inputs, **generation_kwargs)[0]
            response = self._tokeniser.decode(
                output[len(input_ids) :], skip_special_tokens=True
            ).strip()
            yield self._extract_example(response)

    @staticmethod
    def _extract_example(text: str) -> str:
        start_tag = "<EXAMPLE>"
        end_tag = "</EXAMPLE>"
        start = text.find(start_tag)
        if start == -1:
            return text
        end = text.find(end_tag, start)
        if end == -1:
            return text
        return text[start + len(start_tag) : end].strip()
