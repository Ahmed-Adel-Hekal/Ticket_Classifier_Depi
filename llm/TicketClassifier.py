from structured.parser import parse_and_validate_ticket
from structured.schemas import TicketOutput
from prompts.classifier import build_ticket_classification_prompt
from .base import LLM
from .model import GenerationRequest, Message, MessageRole


class TicketClassifier:
    """Application service for classifying and summarizing support tickets."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def classify(self, ticket: str) -> TicketOutput:
        if not isinstance(ticket, str) or not ticket.strip():
            raise ValueError("ticket must be a non-empty string")

        prompt = build_ticket_classification_prompt(ticket)
        request = GenerationRequest(
            messages=[
                Message(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You classify support tickets. Return only valid JSON "
                        "matching the requested schema."
                    ),
                ),
                Message(role=MessageRole.USER, content=prompt),
            ],
            temperature=0.0,
            max_new_tokens=200,
        )
        raw_output = self.llm.generate(request)
        return parse_and_validate_ticket(raw_output)
