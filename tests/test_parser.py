import json
import pytest
from structured.parser import parse_and_validate_ticket


def valid_payload():
    return json.dumps({
        "category": "billing",
        "sentiment": "negative",
        "urgency": "high",
        "summary": "The customer reports a duplicate charge.",
    })


def test_parser_returns_typed_output():
    result = parse_and_validate_ticket(valid_payload())
    assert result.category == "billing"


def test_parser_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_and_validate_ticket("not-json")


def test_parser_rejects_unknown_fields():
    payload = json.loads(valid_payload())
    payload["extra"] = "not allowed"
    with pytest.raises(ValueError):
        parse_and_validate_ticket(json.dumps(payload))
