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


def test_category_accuracy():

    expected = [
        "technical",
        "account",
        "delivery",
        "billing",
        "subscription",
    ]

    predicted = [
        "technical",
        "account",
        "billing",
        "billing",
        "subscription",
    ]

    correct = 0

    for expected_value, predicted_value in zip(
        expected,
        predicted,
    ):
        if expected_value == predicted_value:
            correct += 1

    accuracy = correct / len(expected)

    assert accuracy == 0.8