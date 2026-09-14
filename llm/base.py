from abc import ABC, abstractmethod
from .model import GenerationRequest


class LLM(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        """Generate text for a validated request."""
        raise NotImplementedError
