from collections.abc import Sequence

from .models import EvaluationResult


def validity_rate(
    results: Sequence[EvaluationResult],
) -> float:
    """
    Percentage of evaluation cases where
    structured generation succeeded.
    """

    if not results:
        return 0.0

    return sum(
        result.success
        for result in results
    ) / len(results)


def _successful_results(
    results: Sequence[EvaluationResult],
) -> list[EvaluationResult]:
    """
    Return only cases where structured generation succeeded.
    """

    return [
        result
        for result in results
        if result.success
    ]


def category_accuracy(
    results: Sequence[EvaluationResult],
) -> float:
    """
    Accuracy of the predicted category among
    successfully generated structured outputs.
    """

    successful = _successful_results(results)

    if not successful:
        return 0.0

    return sum(
        result.category_correct
        for result in successful
    ) / len(successful)


def sentiment_accuracy(
    results: Sequence[EvaluationResult],
) -> float:
    """
    Accuracy of the predicted sentiment among
    successfully generated structured outputs.
    """

    successful = _successful_results(results)

    if not successful:
        return 0.0

    return sum(
        result.sentiment_correct
        for result in successful
    ) / len(successful)


def urgency_accuracy(
    results: Sequence[EvaluationResult],
) -> float:
    """
    Accuracy of the predicted urgency among
    successfully generated structured outputs.
    """

    successful = _successful_results(results)

    if not successful:
        return 0.0

    return sum(
        result.urgency_correct
        for result in successful
    ) / len(successful)


def overall_success_rate(
    results: Sequence[EvaluationResult],
) -> float:
    """
    Percentage of cases where:

    1. Structured generation succeeded
    2. Category was correct
    3. Sentiment was correct
    4. Urgency was correct
    """

    if not results:
        return 0.0

    successful_cases = sum(
        result.success
        and result.category_correct
        and result.sentiment_correct
        and result.urgency_correct
        for result in results
    )

    return successful_cases / len(results)


def failure_count(
    results: Sequence[EvaluationResult],
) -> int:
    """
    Number of cases where structured generation failed.
    """

    return sum(
        not result.success
        for result in results
    )