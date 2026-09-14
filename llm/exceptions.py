class LLMError(Exception):
    """Base exception for LLM-related failures."""


class AuthenticationError(LLMError):
    """The provider rejected authentication."""


class InvalidRequestError(LLMError):
    """The provider rejected the request as invalid."""


class RateLimitError(LLMError):
    """The provider rate limit was reached."""


class ConnectionError(LLMError):
    """A network or timeout error occurred."""


class ProviderError(LLMError):
    """The provider returned an unexpected server-side error."""
