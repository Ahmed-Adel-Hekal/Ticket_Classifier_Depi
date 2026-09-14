import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


def log_llm_call(function: Callable[..., T]) -> Callable[..., T]:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        started = time.perf_counter()
        logger.info("llm_call_started function=%s", function.__qualname__)
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            logger.exception(
                "llm_call_failed function=%s error_type=%s elapsed_ms=%.2f",
                function.__qualname__,
                type(exc).__name__,
                (time.perf_counter() - started) * 1000,
            )
            raise
        logger.info(
            "llm_call_succeeded function=%s elapsed_ms=%.2f",
            function.__qualname__,
            (time.perf_counter() - started) * 1000,
        )
        return result

    return wrapper
