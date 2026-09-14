from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    total: int
    valid_outputs: int
    accuracy: float | None = None
    macro_f1: float | None = None
