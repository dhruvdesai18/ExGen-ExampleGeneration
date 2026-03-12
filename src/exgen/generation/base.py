from abc import ABC, abstractmethod

from src.exgen.types import GenerationRequest, GeneratedExample


class Generator(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> list[GeneratedExample]:
        ...
