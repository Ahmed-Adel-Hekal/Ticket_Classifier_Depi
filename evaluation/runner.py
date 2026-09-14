from collections import Counter
from typing import Iterable, Callable

from .models import ClassificationMetrics


def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def evaluate_classifier(rows: Iterable[dict], classify: Callable[[str], object]) -> ClassificationMetrics:
    total = valid = 0
    correct = 0
    true_categories: list[str] = []
    predicted_categories: list[str] = []

    for row in rows:
        total += 1
        try:
            result = classify(row["message"])
            valid += 1
            predicted = result.category
            true = row["category"]
            true_categories.append(true)
            predicted_categories.append(predicted)
            correct += int(predicted == true)
        except (KeyError, ValueError, TypeError, AttributeError):
            continue

    return ClassificationMetrics(
        total=total,
        valid_outputs=valid,
        accuracy=correct / valid if valid else None,
        macro_f1=_macro_f1(true_categories, predicted_categories) if valid else None,
    )
