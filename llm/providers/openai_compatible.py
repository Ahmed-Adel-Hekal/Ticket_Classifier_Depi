from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError as OpenAIAuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError as OpenAIRateLimitError,
)

from ..base import LLM
from ..config import LLMSettings
from ..exceptions import (
    AuthenticationError,
    ConnectionError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)
from ..model import GenerationRequest


class OpenAICompatible(LLM):
    """Adapter for OpenAI-compatible chat-completions providers."""

    def __init__(self, settings: LLMSettings) -> None:
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
        )
        self.model = settings.model

    def generate(self, request: GenerationRequest) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_new_tokens,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            raise InvalidRequestError("top_k is not supported by the OpenAI-compatible chat adapter")

        try:
            response = self.client.chat.completions.create(**payload)
            content = response.choices[0].message.content
            if content is None:
                raise ProviderError("provider returned empty content")
            return content
        except OpenAIAuthenticationError as exc:
            raise AuthenticationError("provider authentication failed") from exc
        except BadRequestError as exc:
            raise InvalidRequestError("provider rejected the request") from exc
        except OpenAIRateLimitError as exc:
            raise RateLimitError("provider rate limit reached") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise ConnectionError("provider connection failed") from exc
        except InternalServerError as exc:
            raise ProviderError("provider server error") from exc
