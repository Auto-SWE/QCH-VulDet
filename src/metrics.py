from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def compute_metrics_at_threshold(
    labels: np.ndarray,
    vuln_probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray(labels)
    vuln_probs = np.asarray(vuln_probs)
    preds = (vuln_probs >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "pr_auc": average_precision_score(labels, vuln_probs),
    }


def compute_binary_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    probs = softmax(logits, axis=1)[:, 1]
    return compute_metrics_at_threshold(labels, probs, threshold=0.5)


def find_best_threshold(labels, vuln_probs) -> dict[str, float]:
    labels = np.asarray(labels)
    vuln_probs = np.asarray(vuln_probs)
    precision, recall, thresholds = precision_recall_curve(labels, vuln_probs)

    if thresholds.size == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = precision[:-1]
    recall = recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(f1))

    return {
        "threshold": float(thresholds[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
    }
