from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, matthews_corrcoef,
                             precision_recall_curve,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score)

DEFAULT_THRESHOLD = 0.5
MIN_PRECISION_TARGETS = (0.05, 0.10, 0.20)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits into class probabilities using a numerically stable softmax."""
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)

    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def classify(probs: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Convert vulnerable-class probabilities into binary predictions."""
    return (probs >= threshold).astype(int)


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers, returning zero when the denominator is zero."""
    return numerator / denominator if denominator else 0.0


def mean_or_zero(values: np.ndarray) -> float:
    """Return the mean of an array, or zero when the array is empty."""
    return float(values.mean()) if values.size else 0.0


def percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    """Return a percentile value, or zero when the input array is empty."""
    values = np.asarray(values)

    if values.size == 0:
        return 0.0

    return float(np.percentile(values, percentile))


def metric_or_zero(metric_fn: Callable[..., float], *args: Any, **kwargs: Any) -> float:
    """Run a metric function, returning zero when sklearn cannot compute it."""
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return 0.0


def f1_from_precision_recall(
    precision: np.ndarray,
    recall: np.ndarray,
) -> np.ndarray:
    """Compute F1 scores from aligned precision and recall arrays."""
    denominator = precision + recall

    return np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def confusion_counts(labels: np.ndarray, preds: np.ndarray) -> dict[str, int]:
    """Count true positives, true negatives, false positives, and false negatives."""
    return {
        "tp": int(((preds == 1) & (labels == 1)).sum()),
        "tn": int(((preds == 0) & (labels == 0)).sum()),
        "fp": int(((preds == 1) & (labels == 0)).sum()),
        "fn": int(((preds == 0) & (labels == 1)).sum()),
    }


def compute_binary_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute all binary classification metrics expected by the trainer."""
    logits, labels = eval_pred

    labels = np.asarray(labels)
    vuln_probs = softmax_np(np.asarray(logits))[:, 1]
    preds = classify(vuln_probs)

    metrics = {}

    metrics.update(default_threshold_metrics(labels, preds, vuln_probs))
    metrics.update(probability_percentile_metrics(labels, vuln_probs))
    metrics.update(best_threshold_metrics(labels, vuln_probs))
    metrics.update(precision_recall_constraint_metrics(labels, vuln_probs))
    metrics.update(curve_metrics(labels, vuln_probs))

    return metrics


def default_threshold_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    vuln_probs: np.ndarray,
) -> dict[str, float]:
    """Compute metrics using the default 0.5 decision threshold."""
    counts = confusion_counts(labels, preds)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy_at_0_5": accuracy_score(labels, preds),
        "balanced_accuracy_at_0_5": balanced_accuracy_score(labels, preds),
        "precision_vulnerable_at_0_5": precision,
        "recall_vulnerable_at_0_5": recall,
        "f1_vulnerable_at_0_5": f1,
        "precision_safe_at_0_5": precision_score(
            labels,
            preds,
            pos_label=0,
            zero_division=0,
        ),
        "recall_safe_at_0_5": recall_score(
            labels,
            preds,
            pos_label=0,
            zero_division=0,
        ),
        "f1_safe_at_0_5": f1_score(
            labels,
            preds,
            pos_label=0,
            zero_division=0,
        ),
        "macro_f1_at_0_5": f1_score(labels, preds, average="macro"),
        "weighted_f1_at_0_5": f1_score(labels, preds, average="weighted"),
        "mcc_at_0_5": matthews_corrcoef(labels, preds),
        "brier_score": brier_score_loss(labels, vuln_probs),
        "predicted_positive_rate_at_0_5": mean_or_zero(preds),
        "confusion_tp_at_0_5": tp,
        "confusion_tn_at_0_5": tn,
        "confusion_fp_at_0_5": fp,
        "confusion_fn_at_0_5": fn,
        "false_positive_rate_at_0_5": safe_divide(fp, fp + tn),
        "false_negative_rate_at_0_5": safe_divide(fn, fn + tp),
    }


def probability_percentile_metrics(
    labels: np.ndarray,
    vuln_probs: np.ndarray,
) -> dict[str, float]:
    """Summarize vulnerable-class probabilities separately for safe and vulnerable labels."""
    safe_probs = vuln_probs[labels == 0]
    vulnerable_probs = vuln_probs[labels == 1]

    return {
        "safe_prob_p50": percentile_or_zero(safe_probs, 50),
        "safe_prob_p90": percentile_or_zero(safe_probs, 90),
        "safe_prob_p99": percentile_or_zero(safe_probs, 99),
        "vulnerable_prob_p50": percentile_or_zero(vulnerable_probs, 50),
        "vulnerable_prob_p90": percentile_or_zero(vulnerable_probs, 90),
        "vulnerable_prob_p99": percentile_or_zero(vulnerable_probs, 99),
    }


def best_threshold_metrics(
    labels: np.ndarray,
    vuln_probs: np.ndarray,
) -> dict[str, float]:
    """Compute metrics at the threshold that maximizes vulnerable-class F1."""
    best = find_best_threshold(labels, vuln_probs)
    best_preds = classify(vuln_probs, threshold=best["threshold"])

    return {
        "best_threshold": best["threshold"],
        "best_f1_vulnerable": best["f1"],
        "best_precision_vulnerable": best["precision"],
        "best_recall_vulnerable": best["recall"],
        "best_predicted_positive_rate": mean_or_zero(best_preds),
    }


def precision_recall_constraint_metrics(
    labels: np.ndarray,
    vuln_probs: np.ndarray,
) -> dict[str, float]:
    """Compute threshold metrics under minimum precision or recall constraints."""
    try:
        precision, recall, thresholds = precision_recall_curve(labels, vuln_probs)
    except ValueError:
        return default_precision_recall_constraint_metrics()

    recall_at_precision, threshold_at_precision = rate_at_constraint(
        precision,
        recall,
        thresholds,
        min_precision=0.2,
    )
    precision_at_recall, threshold_at_recall = rate_at_constraint(
        precision,
        recall,
        thresholds,
        min_recall=0.8,
    )

    metrics = {
        "recall_at_precision_0_2": recall_at_precision,
        "threshold_at_precision_0_2": threshold_at_precision,
        "precision_at_recall_0_8": precision_at_recall,
        "threshold_at_recall_0_8": threshold_at_recall,
    }

    for min_precision in MIN_PRECISION_TARGETS:
        best = best_f1_at_min_precision(
            precision,
            recall,
            thresholds,
            min_precision=min_precision,
        )

        suffix = str(min_precision).replace(".", "_")

        metrics.update(
            {
                f"best_f1_vulnerable_at_precision_{suffix}": best["f1"],
                f"best_threshold_at_precision_{suffix}": best["threshold"],
                f"best_recall_vulnerable_at_precision_{suffix}": best["recall"],
            }
        )

    return metrics


def default_precision_recall_constraint_metrics() -> dict[str, float]:
    """Return fallback values for constrained precision/recall metrics."""
    metrics = {
        "recall_at_precision_0_2": 0.0,
        "threshold_at_precision_0_2": DEFAULT_THRESHOLD,
        "precision_at_recall_0_8": 0.0,
        "threshold_at_recall_0_8": DEFAULT_THRESHOLD,
    }

    for min_precision in MIN_PRECISION_TARGETS:
        suffix = str(min_precision).replace(".", "_")

        metrics.update(
            {
                f"best_f1_vulnerable_at_precision_{suffix}": 0.0,
                f"best_threshold_at_precision_{suffix}": DEFAULT_THRESHOLD,
                f"best_recall_vulnerable_at_precision_{suffix}": 0.0,
            }
        )

    return metrics


def curve_metrics(labels: np.ndarray, vuln_probs: np.ndarray) -> dict[str, float]:
    """Compute threshold-independent probability metrics such as PR AUC and ROC AUC."""
    pred_probs = np.column_stack([1.0 - vuln_probs, vuln_probs])

    return {
        "pr_auc": metric_or_zero(average_precision_score, labels, vuln_probs),
        "roc_auc": metric_or_zero(roc_auc_score, labels, vuln_probs),
        "log_loss": metric_or_zero(log_loss, labels, pred_probs, labels=[0, 1]),
    }


def rate_at_constraint(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    *,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> tuple[float, float]:
    """Find the best achievable recall or precision under a minimum constraint."""
    if thresholds.size == 0:
        return 0.0, DEFAULT_THRESHOLD

    precision = precision[:-1]
    recall = recall[:-1]

    if min_precision is not None:
        valid = precision >= min_precision

        if not valid.any():
            return 0.0, DEFAULT_THRESHOLD

        best_idx = int(np.argmax(np.where(valid, recall, -1.0)))
        return float(recall[best_idx]), float(thresholds[best_idx])

    if min_recall is not None:
        valid = recall >= min_recall

        if not valid.any():
            return 0.0, DEFAULT_THRESHOLD

        best_idx = int(np.argmax(np.where(valid, precision, -1.0)))
        return float(precision[best_idx]), float(thresholds[best_idx])

    raise ValueError("Provide either min_precision or min_recall.")


def best_f1_at_min_precision(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    min_precision: float,
) -> dict[str, float]:
    """Find the best F1 score among thresholds that satisfy a minimum precision."""
    default = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "threshold": DEFAULT_THRESHOLD,
    }

    if thresholds.size == 0:
        return default

    precision = precision[:-1]
    recall = recall[:-1]

    valid = precision >= min_precision

    if not valid.any():
        return default

    f1 = f1_from_precision_recall(precision, recall)
    best_idx = int(np.argmax(np.where(valid, f1, -1.0)))

    return {
        "f1": float(f1[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "threshold": float(thresholds[best_idx]),
    }


def find_best_threshold(
    labels: np.ndarray,
    vuln_probs: np.ndarray,
) -> dict[str, float]:
    """Find the threshold that maximizes vulnerable-class F1."""
    labels = np.asarray(labels)
    vuln_probs = np.asarray(vuln_probs)

    default = {
        "threshold": DEFAULT_THRESHOLD,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    if labels.size == 0 or vuln_probs.size == 0:
        return default

    try:
        precision, recall, thresholds = precision_recall_curve(labels, vuln_probs)
    except ValueError:
        return default

    if thresholds.size == 0:
        return default

    precision = precision[:-1]
    recall = recall[:-1]

    f1 = f1_from_precision_recall(precision, recall)
    best_idx = int(np.argmax(f1))

    return {
        "threshold": float(thresholds[best_idx]),
        "f1": float(f1[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
    }
