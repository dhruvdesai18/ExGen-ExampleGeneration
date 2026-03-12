from dataclasses import dataclass
from typing import Iterator
from abc import ABC, abstractmethod


@dataclass
class Model(ABC):
    name: str

    @abstractmethod
    def generate(
            self,
            messages: list[dict[str, str]],
            num_candidates: int,
            temperature: float,
            max_tokens: int
    ) -> Iterator[str]:
        for i in range(num_candidates):
            yield ...
