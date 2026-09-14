from ..base import LLM
from ..model import GenerationRequest
from src.inference.generate import generate_text


class LocalLLM(LLM):
    """LLM adapter around the project's custom local generation engine."""

    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, request: GenerationRequest) -> str:
        prompt = "\n".join(
            f"{message.role.value}: {message.content}" for message in request.messages
        )
        return generate_text(
            tokenizer=self.tokenizer,
            model=self.model,
            prompt=prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
