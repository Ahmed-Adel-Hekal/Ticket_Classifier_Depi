import json
from pydantic import ValidationError
from .schemas import TicketOutput


def parse_and_validate_ticket(raw_output: str) -> TicketOutput:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("model output must be a non-empty string")
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("model output is not valid JSON") from exc
    try:
        return TicketOutput.model_validate(data)
    except ValidationError as exc:
        raise ValueError("model output does not match TicketOutput") from exc
