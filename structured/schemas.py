from pydantic import BaseModel
from typing import Literal

class TicketOutput(BaseModel):

    category: Literal[
        "technical",
        "account",
        "delivery",
        "billing",
        "subscription",
    ]

    sentiment: Literal[
        "positive",
        "negative",
        "neutral",
    ]

    urgency: Literal[
        "low",
        "medium",
        "high",
    ]

    summary: str