from dataclasses import dataclass
from typing import Iterator

from .base import Model


@dataclass
class GPT(Model):
    model_id: str
    timeout_seconds: int | None = None

    def generate(
        self,
        messages: list[dict[str, str]],
        num_candidates: int,
    ) -> Iterator[str]:
        raise NotImplementedError(
            f"GPT generation is not implemented for model '{self.model_id}'."
        )
