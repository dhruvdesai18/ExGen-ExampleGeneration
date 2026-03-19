from dataclasses import dataclass
from typing import Iterator

from .base import Model


@dataclass
class MinistralQFT(Model):
    model_id: str

    def generate(
        self,
        messages: list[dict[str, str]],
        num_candidates: int,
    ) -> Iterator[str]:
        raise NotImplementedError(
            f"Ministral generation is not implemented for model '{self.model_id}'."
        )
