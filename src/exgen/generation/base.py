from abc import ABC, abstractmethod

from exgen.types import GenerationRequest, GeneratedExample


class Generator(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> list[GeneratedExample]:
        ...
