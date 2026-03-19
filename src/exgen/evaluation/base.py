from abc import ABC, abstractmethod

from exgen.types import EvaluationRecord, GeneratedExample


class Evaluator(ABC):
    metric_name: str

    @abstractmethod
    def evaluate(self, example: GeneratedExample) -> EvaluationRecord:
        raise NotImplementedError


class Verifier(ABC):
    @abstractmethod
    def verify(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError
