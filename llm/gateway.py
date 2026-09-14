import logging
import random
import time
from collections.abc import Callable

from .base import LLM
from .decorators import log_llm_call
from .exceptions import ConnectionError, LLMError, ProviderError, RateLimitError
from .model import GenerationRequest

logger = logging.getLogger(__name__)


class LLMGateway(LLM):
    """Reliability wrapper that retries transient provider failures."""

    def __init__(
        self,
        llm: LLM,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        self.llm = llm
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self._sleep = sleep
        self._random_uniform = random_uniform

    @log_llm_call
    def generate(self, request: GenerationRequest) -> str:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.llm.generate(request)
            except (RateLimitError, ConnectionError, ProviderError) as exc:
                if attempt == self.max_attempts:
                    logger.error("llm_retry_exhausted attempts=%s", attempt)
                    raise
                ceiling = self.base_delay * (2 ** (attempt - 1))
                delay = self._random_uniform(0, ceiling)
                logger.warning(
                    "llm_retrying attempt=%s next_attempt=%s delay_seconds=%.3f error=%s",
                    attempt, attempt + 1, delay, type(exc).__name__,
                )
                self._sleep(delay)
            except LLMError:
                raise
        raise RuntimeError("unreachable retry state")
