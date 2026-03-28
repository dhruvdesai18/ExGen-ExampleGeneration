from abc import ABC, abstractmethod

from exgen.types import EvaluationOutcome, GeneratedExample


class Evaluator(ABC):
    metric_name: str

    @abstractmethod
    def evaluate(self, example: GeneratedExample) -> EvaluationOutcome:
        raise NotImplementedError


class Verifier(ABC):
    @abstractmethod
    def verify(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict | None = None,
    ) -> str:
        raise NotImplementedError
