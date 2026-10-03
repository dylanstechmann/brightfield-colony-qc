"""Multinomial logistic regression on morphology features.

Intentionally small. The point is a baseline a wet-lab collaborator can
retrain on their own annotated brightfield fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from colonyqc.features import FEATURE_SCHEMA_VERSION
from colonyqc.synthetic import LABELS

MODEL_SCHEMA_VERSION = 2


class SoftmaxQC:
    def __init__(self):
        self.classes = list(LABELS)
        self.mean = None
        self.std = None
        self.weights = None  # (d, k)
        self.bias = None
        self.training_provenance = {"source": "unknown_not_recorded"}
        self.training_domain = {"status": "unknown_not_recorded"}
        self.model_schema_version = MODEL_SCHEMA_VERSION
        self.feature_schema_version = FEATURE_SCHEMA_VERSION

    def fit(self, x: np.ndarray, y: list[str], lr: float = 0.35, epochs: int = 500, l2: float = 1e-3, seed: int = 0, provenance: dict | None = None, training_domain: dict | None = None):
        if x.ndim != 2 or len(x) != len(y) or not len(y):
            raise ValueError("training features and labels must be nonempty and aligned")
        if not np.isfinite(x).all():
            raise ValueError("training features must be finite")
        if any(label not in self.classes for label in y):
            raise ValueError("training labels must use the supported class names")
        self.training_provenance = dict(provenance or {"source": "unknown_not_recorded"})
        self.training_domain = dict(training_domain or {"status": "unknown_not_recorded"})
        class_index = {c: i for i, c in enumerate(self.classes)}
        y_idx = np.array([class_index[label] for label in y], dtype=np.int64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        z = self._standardize(x)
        n, d = z.shape
        k = len(self.classes)
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.01, size=(d, k))
        self.bias = np.zeros(k)
        y_hot = np.eye(k)[y_idx]
        for _ in range(epochs):
            logits = z @ self.weights + self.bias
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            proba = exp / exp.sum(axis=1, keepdims=True)
            grad = (proba - y_hot) / n
            self.weights -= lr * (z.T @ grad + l2 * self.weights)
            self.bias -= lr * grad.sum(axis=0)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self._standardize(x)
        logits = z @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> list[str]:
        proba = self.predict_proba(x)
        idx = proba.argmax(axis=1)
        return [self.classes[i] for i in idx]

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict:
        return {
            "classes": self.classes,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "training_provenance": self.training_provenance,
            "training_domain": self.training_domain,
            "model_schema_version": self.model_schema_version,
            "feature_schema_version": self.feature_schema_version,
            "training_data_status": self.training_data_status,
        }

    @property
    def training_data_status(self) -> str:
        source = self.training_provenance.get("source")
        if source == "synthetic_generator":
            return "synthetic"
        if source == "unknown_not_recorded":
            return "unknown"
        return "non_synthetic_unverified"

    @classmethod
    def from_dict(cls, payload: dict) -> "SoftmaxQC":
        model = cls()
        model.classes = list(payload["classes"])
        model.mean = np.array(payload["mean"], dtype=np.float64)
        model.std = np.array(payload["std"], dtype=np.float64)
        model.weights = np.array(payload["weights"], dtype=np.float64)
        model.bias = np.array(payload["bias"], dtype=np.float64)
        model.training_provenance = dict(
            payload.get("training_provenance", {"source": "unknown_not_recorded"})
        )
        model.training_domain = dict(payload.get("training_domain", {"status": "unknown_not_recorded"}))
        model.model_schema_version = int(payload.get("model_schema_version", 1))
        model.feature_schema_version = int(payload.get("feature_schema_version", 0))
        return model

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SoftmaxQC":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("empty or mismatched labels")
    hit = sum(a == b for a, b in zip(y_true, y_pred))
    return hit / len(y_true)


def majority_accuracy(y_true: list[str], y_train: list[str]) -> float:
    counts = {}
    for label in y_train:
        counts[label] = counts.get(label, 0) + 1
    majority = max(counts, key=counts.get)
    return sum(label == majority for label in y_true) / len(y_true)
