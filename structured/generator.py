from typing import Any, Type
from .outlines_adapter import create_structured_model


class StructuredGeneration:
    """Thin adapter around an optional structured-generation backend."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = create_structured_model(model, tokenizer)

    def generate(self, prompt: str, schema: Type[Any]) -> Any:
        return self.model(prompt, schema)
