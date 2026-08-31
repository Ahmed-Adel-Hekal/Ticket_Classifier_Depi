from typing import Optional

from pydantic import BaseModel

from structured.schemas import TicketOutput


class EvaluationResult(BaseModel):
    """
    Stores the result of evaluating one support ticket.

    This model contains:
    - the ticket identity
    - the ground-truth labels
    - the model prediction
    - correctness information
    - generation status
    - error information
    """

    ticket_id: int

    # Did the model successfully produce a structured output?
    success: bool

    # Ground-truth values from the evaluation dataset
    expected_category: str
    expected_sentiment: str
    expected_urgency: str

    # Model prediction
    # None when generation fails
    result: Optional[TicketOutput] = None

    # Field-level correctness
    category_correct: bool = False
    sentiment_correct: bool = False
    urgency_correct: bool = False

    # Error information when evaluation fails
    error: Optional[str] = None