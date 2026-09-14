import pytest
from llm.base import LLM
from llm.exceptions import ConnectionError
from llm.gateway import LLMGateway
from llm.model import GenerationRequest, Message, MessageRole


class FlakyLLM(LLM):
    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.calls < 3:
            raise ConnectionError("temporary")
        return "ok"


def request():
    return GenerationRequest([Message(MessageRole.USER, "hello")])


def test_gateway_retries_then_succeeds():
    provider = FlakyLLM()
    sleeps = []
    gateway = LLMGateway(provider, max_attempts=3, sleep=sleeps.append, random_uniform=lambda a, b: b)
    assert gateway.generate(request()) == "ok"
    assert provider.calls == 3
    assert len(sleeps) == 2


def test_gateway_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        LLMGateway(FlakyLLM(), max_attempts=0)
