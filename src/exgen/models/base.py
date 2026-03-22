from dataclasses import dataclass
from typing import Iterator
from abc import ABC, abstractmethod


@dataclass
class Model(ABC):
    name: str
    temperature: float
    top_p: float
    max_tokens: int

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        num_candidates: int,
    ) -> Iterator[str]:
        for i in range(num_candidates):
            yield ...
