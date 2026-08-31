from evaluation.metrics import (
    validity_rate,
    category_accuracy,
    sentiment_accuracy,
    urgency_accuracy,
    overall_success_rate,
    failure_count,
)
from evaluation.models import EvaluationResult


def make_result(
    ticket_id=1,
    success=True,
    category_correct=True,
    sentiment_correct=True,
    urgency_correct=True,
):
    return EvaluationResult(
        ticket_id=ticket_id,
        success=success,
        expected_category="billing",
        expected_sentiment="negative",
        expected_urgency="high",
        category_correct=category_correct,
        sentiment_correct=sentiment_correct,
        urgency_correct=urgency_correct,
    )


def test_validity_rate():
    results = [
        make_result(ticket_id=1, success=True),
        make_result(ticket_id=2, success=True),
        make_result(ticket_id=3, success=False),
        make_result(ticket_id=4, success=True),
    ]

    assert validity_rate(results) == 0.75


def test_validity_rate_empty_results():
    assert validity_rate([]) == 0.0


def test_category_accuracy():
    results = [
        make_result(ticket_id=1, category_correct=True),
        make_result(ticket_id=2, category_correct=True),
        make_result(ticket_id=3, category_correct=False),
        make_result(ticket_id=4, category_correct=True),
    ]

    assert category_accuracy(results) == 0.75


def test_category_accuracy_ignores_generation_failures():
    results = [
        make_result(
            ticket_id=1,
            success=True,
            category_correct=True,
        ),
        make_result(
            ticket_id=2,
            success=False,
            category_correct=False,
        ),
    ]

    assert category_accuracy(results) == 1.0


def test_sentiment_accuracy():
    results = [
        make_result(ticket_id=1, sentiment_correct=True),
        make_result(ticket_id=2, sentiment_correct=False),
        make_result(ticket_id=3, sentiment_correct=True),
        make_result(ticket_id=4, sentiment_correct=True),
    ]

    assert sentiment_accuracy(results) == 0.75


def test_urgency_accuracy():
    results = [
        make_result(ticket_id=1, urgency_correct=True),
        make_result(ticket_id=2, urgency_correct=False),
        make_result(ticket_id=3, urgency_correct=False),
        make_result(ticket_id=4, urgency_correct=True),
    ]

    assert urgency_accuracy(results) == 0.50


def test_overall_success_rate():
    results = [
        make_result(
            ticket_id=1,
            success=True,
            category_correct=True,
            sentiment_correct=True,
            urgency_correct=True,
        ),
        make_result(
            ticket_id=2,
            success=True,
            category_correct=True,
            sentiment_correct=True,
            urgency_correct=False,
        ),
        make_result(
            ticket_id=3,
            success=True,
            category_correct=False,
            sentiment_correct=True,
            urgency_correct=True,
        ),
        make_result(
            ticket_id=4,
            success=False,
        ),
    ]

    assert overall_success_rate(results) == 0.25


def test_failure_count():
    results = [
        make_result(ticket_id=1, success=True),
        make_result(ticket_id=2, success=False),
        make_result(ticket_id=3, success=True),
        make_result(ticket_id=4, success=False),
    ]

    assert failure_count(results) == 2


def test_failure_count_when_no_failures():
    results = [
        make_result(ticket_id=1, success=True),
        make_result(ticket_id=2, success=True),
    ]

    assert failure_count(results) == 0

