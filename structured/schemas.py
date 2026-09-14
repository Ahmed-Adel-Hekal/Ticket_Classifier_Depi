from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class TicketOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["technical", "account", "delivery", "billing", "subscription"]
    sentiment: Literal["positive", "negative", "neutral"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(min_length=1, max_length=1000)
