"""Binary image- and patient-level metrics with patient-cluster bootstrap CIs."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    auc = (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else None
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return {
            "n": int(len(labels)),
            "positive_n": int(labels.sum()),
            "negative_n": int(len(labels) - labels.sum()),
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "sensitivity": _safe_ratio(tp, tp + fn),
            "specificity": _safe_ratio(tn, tn + fp),
            "precision": float(
                precision_score(labels, predictions, zero_division=0)
            ),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "auroc": auc,
            "confusion_matrix": matrix.tolist(),
        }


def _patient_rows(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_indices: np.ndarray,
    patient_ids: list[str],
) -> list[dict]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for row, patient_index in enumerate(patient_indices):
        grouped[int(patient_index)].append(row)

    patients = []
    for patient_index, rows in grouped.items():
        patient_labels = np.unique(labels[rows])
        if len(patient_labels) != 1:
            raise ValueError("A patient has conflicting labels in one split")
        patients.append(
            {
                "patient_id": patient_ids[patient_index],
                "label": int(patient_labels[0]),
                "mean_probability": float(probabilities[rows].mean()),
                "n_images": len(rows),
            }
        )
    return sorted(patients, key=lambda row: row["patient_id"])


def _ci(values: list[float], lower: float = 2.5, upper: float = 97.5):
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(finite) < 20:
        return None
    return [
        float(np.percentile(finite, lower)),
        float(np.percentile(finite, upper)),
    ]


def _patient_bootstrap_ci(
    patients: list[dict], n_boot: int, seed: int, threshold: float
) -> dict:
    rng = np.random.default_rng(seed)
    accuracy_values: list[float] = []
    balanced_values: list[float] = []
    auc_values: list[float] = []

    for _ in range(n_boot):
        indices = rng.integers(0, len(patients), size=len(patients))
        sampled = [patients[index] for index in indices]
        labels = np.asarray([row["label"] for row in sampled], dtype=int)
        probabilities = np.asarray(
            [row["mean_probability"] for row in sampled], dtype=float
        )
        metrics = binary_metrics(labels, probabilities, threshold=threshold)
        accuracy_values.append(metrics["accuracy"])
        balanced_values.append(metrics["balanced_accuracy"])
        if metrics["auroc"] is not None:
            auc_values.append(metrics["auroc"])

    return {
        "accuracy": _ci(accuracy_values),
        "balanced_accuracy": _ci(balanced_values),
        "auroc": _ci(auc_values),
    }


def evaluate_predictions(
    labels: list[int],
    probabilities: list[float],
    patient_indices: list[int],
    patient_ids: list[str],
    threshold: float = 0.5,
    bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    labels_array = np.asarray(labels, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    patient_index_array = np.asarray(patient_indices, dtype=int)

    patients = _patient_rows(
        labels_array, probability_array, patient_index_array, patient_ids
    )
    patient_labels = np.asarray([row["label"] for row in patients], dtype=int)
    patient_probabilities = np.asarray(
        [row["mean_probability"] for row in patients], dtype=float
    )

    result = {
        "image_level": binary_metrics(
            labels_array, probability_array, threshold=threshold
        ),
        "patient_level": binary_metrics(
            patient_labels, patient_probabilities, threshold=threshold
        ),
        "patient_bootstrap_ci95": (
            _patient_bootstrap_ci(
                patients, n_boot=bootstrap, seed=seed, threshold=threshold
            )
            if bootstrap > 0
            else None
        ),
        "patients": patients,
    }
    for split_metrics in (result["image_level"], result["patient_level"]):
        for key, value in split_metrics.items():
            if isinstance(value, float) and math.isnan(value):
                split_metrics[key] = None
    return result


def select_operating_threshold(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
) -> dict:
    """Choose a threshold on validation data using patient-balanced accuracy."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates: list[dict] = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = binary_metrics(labels, probabilities, threshold=float(threshold))
        candidates.append(
            {
                "threshold": round(float(threshold), 2),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            row["balanced_accuracy"],
            row["sensitivity"] if row["sensitivity"] is not None else -1,
            -abs(row["threshold"] - 0.5),
        ),
    )
    return {
        "selected": best,
        "candidates": candidates,
        "rule": (
            "maximize patient-level balanced accuracy; break ties by sensitivity, "
            "then closeness to 0.5"
        ),
    }

