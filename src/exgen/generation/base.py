from abc import ABC, abstractmethod

from exgen.types import GeneratedExample, LexicalItem


class Generator(ABC):
    @abstractmethod
    def generate(self, lexical_item: LexicalItem) -> list[GeneratedExample]:
        ...
