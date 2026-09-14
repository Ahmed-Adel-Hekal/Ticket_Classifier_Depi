from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    api_key: str = Field(..., min_length=1, validation_alias="LLM_API_KEY")
    base_url: str = Field(
        "https://api.groq.com/openai/v1",
        min_length=1,
        validation_alias="LLM_BASE_URL",
    )
    model: str = Field(..., min_length=1, validation_alias="LLM_MODEL")
    max_attempts: int = Field(3, ge=1, validation_alias="LLM_MAX_ATTEMPTS")
    base_delay: float = Field(1.0, ge=0, validation_alias="LLM_BASE_DELAY")
    timeout: float = Field(30.0, gt=0, validation_alias="LLM_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
