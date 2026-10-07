"""Selective prediction: abstain on low-confidence fields and measure the trade-off.

A classifier that must answer on every field spends its error budget on the
fields it understands least. Abstaining instead hands those fields to a human.
The quantity that matters is not accuracy alone but accuracy *at a stated
coverage*: what fraction of fields the model answered, and how often it was right
when it did.

Coverage and selective accuracy here are measured on the synthetic generator's
holdout. They describe this software, not iPSC cultures, and an abstention is a
request for human review rather than a statement about a culture.
"""

from __future__ import annotations

import numpy as np

DEFAULT_ABSTAIN_THRESHOLD = 0.60
ABSTAIN_LABEL = "abstain_low_confidence"


def confidence(probabilities: np.ndarray) -> np.ndarray:
    """Top-label probability per row, the score the abstention rule thresholds."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or not probabilities.size:
        raise ValueError("probabilities must be a nonempty two-dimensional array")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    return probabilities.max(axis=1)


def selective_calls(labels: list[str], probabilities: np.ndarray,
                    threshold: float = DEFAULT_ABSTAIN_THRESHOLD) -> list[str]:
    """Replace a predicted label with an abstention when confidence is below threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    scores = confidence(probabilities)
    if len(labels) != len(scores):
        raise ValueError("labels and probabilities must be aligned")
    return [label if score >= threshold else ABSTAIN_LABEL for label, score in zip(labels, scores)]


def selective_metrics(truth: list[str], predicted: list[str], probabilities: np.ndarray,
                      threshold: float = DEFAULT_ABSTAIN_THRESHOLD) -> dict:
    """Coverage, selective accuracy, and the error rate among answered fields.

    ``selective_accuracy`` is accuracy among answered fields only. It is not
    comparable with full-coverage accuracy: a model can raise it to 1.0 by
    answering almost nothing, which is why coverage is reported beside it and
    never omitted.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    scores = confidence(probabilities)
    if not (len(truth) == len(predicted) == len(scores)):
        raise ValueError("truth, predictions and probabilities must be aligned")
    answered = scores >= threshold
    n_answered = int(answered.sum())
    correct = sum(1 for index in range(len(truth))
                  if answered[index] and truth[index] == predicted[index])
    deferred_correct = sum(1 for index in range(len(truth))
                           if not answered[index] and truth[index] == predicted[index])
    return {
        "threshold": float(threshold),
        "n_total": len(truth),
        "n_answered": n_answered,
        "n_abstained": len(truth) - n_answered,
        "coverage": n_answered / len(truth),
        "selective_accuracy": correct / n_answered if n_answered else None,
        "selective_error_rate": 1 - correct / n_answered if n_answered else None,
        "full_coverage_accuracy": sum(1 for index in range(len(truth))
                                      if truth[index] == predicted[index]) / len(truth),
        "abstained_would_have_been_correct": deferred_correct,
        "abstention_precision": (
            1 - deferred_correct / (len(truth) - n_answered) if len(truth) - n_answered else None
        ),
    }


def confidence_ranks_correctness(points: list[dict], minimum_coverage: float = 0.2) -> dict:
    """Does abstaining on low confidence actually improve accuracy on answered fields?

    Selective prediction only helps when confidence ranks correctness. If accuracy
    among answered fields does not rise as coverage falls, the confidence score is
    uninformative for that input distribution and abstention buys nothing. Saying so
    is the point: a coverage number alone would hide it.
    """
    usable = [point for point in points
              if point["selective_accuracy"] is not None and point["coverage"] >= minimum_coverage]
    full = next((point for point in points if point["coverage"] == 1.0), None)
    if not usable or full is None or full["selective_accuracy"] is None:
        return {"status": "not_assessable", "minimum_coverage": minimum_coverage,
                "detail": "No threshold retained enough coverage to compare against full coverage."}
    strictest = min(usable, key=lambda point: point["coverage"])
    gain = strictest["selective_accuracy"] - full["selective_accuracy"]
    if full["selective_accuracy"] >= 1.0:
        status, detail = "no_headroom", (
            "Full-coverage accuracy is already 1.0 on these rows, so abstention has nothing to "
            "improve. That ceiling is a property of the synthetic generator, not evidence about "
            "real fields.")
    elif gain > 0:
        status, detail = "improves", (
            "Accuracy among answered fields rises as coverage falls, so the confidence score "
            "ranks correctness for this input distribution.")
    elif gain == 0:
        status, detail = "flat", (
            "Accuracy among answered fields is unchanged as coverage falls, so abstaining on this "
            "confidence score neither helps nor hurts for this input distribution.")
    else:
        status, detail = "does_not_improve", (
            "Accuracy among answered fields falls as coverage falls, so the confidence score does "
            "not rank correctness for this input distribution and abstaining on it buys nothing "
            "here.")
    return {
        "status": status,
        "minimum_coverage": minimum_coverage,
        "full_coverage_accuracy": full["selective_accuracy"],
        "strictest_assessed_threshold": strictest["threshold"],
        "strictest_assessed_coverage": strictest["coverage"],
        "strictest_assessed_selective_accuracy": strictest["selective_accuracy"],
        "selective_accuracy_gain": gain,
        "detail": detail,
    }


def risk_coverage_curve(truth: list[str], predicted: list[str], probabilities: np.ndarray,
                        thresholds: list[float] | None = None) -> dict:
    """Selective accuracy across thresholds, with the trade-off stated explicitly.

    Reported thresholds are fixed, not selected to look good. Choosing a threshold
    on this same holdout would make its coverage and accuracy development
    estimates rather than a prediction for new fields.
    """
    grid = [0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] if thresholds is None else list(thresholds)
    if any(not 0 <= value <= 1 for value in grid):
        raise ValueError("every threshold must lie in [0, 1]")
    points = [selective_metrics(truth, predicted, probabilities, threshold)
              for threshold in sorted(set(grid))]
    return {
        "points": points,
        "confidence_diagnostic": confidence_ranks_correctness(points),
        "method": "Fixed top-label confidence thresholds on out-of-sample generator fields; "
                  "no threshold was selected using these same rows.",
        "limitations": [
            "Selective accuracy rises with a stricter threshold only because fewer fields are "
            "answered. Read it together with coverage, never alone.",
            "An abstention is a request for human review. It is not a statement that a culture "
            "is abnormal, contaminated or unusable.",
            "These rows come from the synthetic generator used for training, so the curve "
            "describes software behavior and not performance on real iPSC fields.",
            "A threshold chosen by inspecting this curve becomes a development choice; its "
            "coverage and accuracy would need an untouched cohort to be a prediction.",
        ],
    }
