import os
import sys

import pytest

# Allow imports from the project root
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from structured.parser import parse_and_validate_ticket


def test_valid_ticket_output():
    # Arrange
    data = """{
        "sentiment": "positive",
        "urgency": "high",
        "category": "billing",
        "summary": "I have a question about my bill and need assistance."
    }"""

    # Act
    result = parse_and_validate_ticket(data)

    # Assert
    assert result.sentiment == "positive"
    assert result.urgency == "high"
    assert result.category == "billing"
    assert result.summary == (
        "I have a question about my bill and need assistance."
    )


def test_invalid_json_is_rejected():
    # Arrange
    data = """{
        "sentiment": "positive",
        "urgency": "high",
        "category": "billing",
        "summary": "I have a question about my bill and need assistance."
    """

    # Act + Assert
    with pytest.raises(ValueError):
        parse_and_validate_ticket(data)


def test_missing_required_field_is_rejected():
    # Arrange
    data = """{
        "sentiment": "positive",
        "urgency": "high",
        "category": "billing"
    }"""

    # Act + Assert
    with pytest.raises(ValueError):
        parse_and_validate_ticket(data)


def test_invalid_category_is_rejected():
    # Arrange
    data = """{
        "sentiment": "positive",
        "urgency": "high",
        "category": "complaint",
        "summary": "I have a question about my bill and need assistance."
    }"""

    # Act + Assert
    with pytest.raises(ValueError):
        parse_and_validate_ticket(data)