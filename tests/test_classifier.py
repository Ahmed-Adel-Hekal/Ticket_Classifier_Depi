import json
from llm.TicketClassifier import TicketClassifier
from llm.base import LLM
from llm.model import GenerationRequest


class FakeLLM(LLM):
    def generate(self, request: GenerationRequest) -> str:
        return json.dumps({
            "category": "billing",
            "sentiment": "negative",
            "urgency": "high",
            "summary": "Duplicate charge reported.",
        })


def test_classifier_returns_structured_result():
    result = TicketClassifier(FakeLLM()).classify("I was charged twice")
    assert result.category == "billing"
    assert result.urgency == "high"
