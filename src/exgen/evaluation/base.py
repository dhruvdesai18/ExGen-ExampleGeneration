from abc import ABC, abstractmethod

from src.exgen.types import GeneratedExample


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, example: GeneratedExample):
        pass