from types import SimpleNamespace

from evaluation.runner import evaluate_classifier


def test_evaluate_classifier_metrics():
    rows = [
        {"message": "a", "category": "technical"},
        {"message": "b", "category": "billing"},
    ]

    def classify(message):
        return SimpleNamespace(category="technical" if message == "a" else "billing")

    metrics = evaluate_classifier(rows, classify)
    assert metrics.total == 2
    assert metrics.valid_outputs == 2
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
