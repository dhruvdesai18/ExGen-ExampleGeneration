from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterator


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
        raise NotImplementedError
