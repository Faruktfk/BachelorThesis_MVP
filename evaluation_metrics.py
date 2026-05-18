"""Shared evaluation utilities for the ML debugging experiment.

Why this file exists:
- Baseline, XAI and Oracle repair should use exactly the same metrics.
- Accuracy alone is too coarse for small test splits.
- Log-loss and Brier score make probability-quality changes visible.
- Oracle-normalized repair impact helps interpret whether a workflow captures
  the theoretically possible repair effect.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


ORDERED_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "log_loss",
    "brier_score",
]

LOWER_IS_BETTER_SUFFIXES = (
    "log_loss",
    "brier_score",
)


def is_lower_better(metric_name: str) -> bool:
    """Return True if lower values mean better performance."""
    return metric_name.endswith(LOWER_IS_BETTER_SUFFIXES)


def metric_improvement(metric_name: str, delta: float) -> float:
    """Convert a before-after delta into an improvement value.

    Example:
    - accuracy delta +0.02 means improvement +0.02
    - log_loss delta -0.02 means improvement +0.02
    """
    if is_lower_better(metric_name):
        return -float(delta)
    return float(delta)


def safe_float(value: Any) -> float | None:
    """Convert a value to float, but return None for NaN/inf."""
    try:
        number = float(value)
    except Exception:
        return None

    if not np.isfinite(number):
        return None

    return number


def evaluate_classifier(
    model: RandomForestClassifier,
    features: pd.DataFrame,
    labels: pd.Series,
) -> Dict[str, float]:
    """Evaluate a fitted classifier with robust classification metrics.

    Metrics:
    - accuracy: share of correct hard predictions
    - balanced_accuracy: class-balanced accuracy
    - f1: harmonic mean of precision and recall
    - roc_auc: ranking/separation quality based on probabilities
    - log_loss: probability calibration/error; lower is better
    - brier_score: squared probability error; lower is better
    """
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    except Exception:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["log_loss"] = float(log_loss(labels, probabilities, labels=[0, 1]))
    except Exception:
        metrics["log_loss"] = float("nan")

    try:
        metrics["brier_score"] = float(brier_score_loss(labels, probabilities))
    except Exception:
        metrics["brier_score"] = float("nan")

    return metrics


def compute_split_metrics(
    model: RandomForestClassifier,
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
) -> Dict[str, float]:
    """Compute flat metrics for train, contaminated_eval and clean_holdout."""
    flat_metrics: Dict[str, float] = {}

    for split_name in ["train", "contaminated_eval", "clean_holdout"]:
        split_metrics = evaluate_classifier(
            model=model,
            features=splits[split_name]["features"],
            labels=splits[split_name]["labels"],
        )

        for metric_name, value in split_metrics.items():
            flat_metrics[f"{split_name}_{metric_name}"] = value

    return flat_metrics


def compute_fix_impact(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
) -> Dict[str, float]:
    """Compute after-before deltas for all available metrics."""
    return {
        metric_name: float(metrics_after.get(metric_name, 0.0)) - float(metrics_before.get(metric_name, 0.0))
        for metric_name in metrics_before.keys()
    }


def delta_as_improvement_label(metric_name: str, delta: float) -> str:
    """Human-readable direction label for metric deltas."""
    if abs(float(delta)) < 1e-12:
        return "no_change"

    improvement = metric_improvement(metric_name, delta)

    if improvement > 0:
        return "improved"

    return "worsened"


def format_delta_direction(metric_name: str, delta: float) -> str:
    """Format whether a delta is good or bad.

    For accuracy/F1/AUC:
    - positive delta is good

    For log_loss/Brier:
    - negative delta is good
    """
    if abs(float(delta)) < 1e-12:
        return "→ no change"

    if is_lower_better(metric_name):
        if delta < 0:
            return "↓ improved"
        return "↑ worsened"

    if delta > 0:
        return "↑ improved"

    return "↓ worsened"


def classify_oracle_repair_potential(
    oracle_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify whether the injected fault has measurable repair potential.

    Important:
    This does NOT evaluate Baseline or XAI directly.
    It only asks:

    If the true injected cause is fixed perfectly, does clean_holdout improve?

    This helps distinguish:
    - faults useful for localization experiments
    - faults useful for repair-impact experiments
    """
    if not oracle_result.get("oracle_fix_applied", False):
        return {
            "repair_effect_quality": "no_oracle_fix",
            "reason": "No oracle fix was applied.",
            "max_clean_holdout_improvement": 0.0,
        }

    fix_impact = oracle_result.get("fix_impact", {})

    clean_metric_names = [
        "clean_holdout_accuracy",
        "clean_holdout_balanced_accuracy",
        "clean_holdout_f1",
        "clean_holdout_roc_auc",
        "clean_holdout_log_loss",
        "clean_holdout_brier_score",
    ]

    improvements: Dict[str, float] = {}
    for metric_name in clean_metric_names:
        if metric_name in fix_impact:
            improvements[metric_name] = metric_improvement(metric_name, float(fix_impact[metric_name]))

    if not improvements:
        return {
            "repair_effect_quality": "unknown",
            "reason": "No clean_holdout metrics were available.",
            "max_clean_holdout_improvement": 0.0,
        }

    max_improvement = max(improvements.values())

    # Accuracy on this dataset changes in steps of about 0.0088.
    # Therefore, 0.008 is already one sample-level step.
    if (
        improvements.get("clean_holdout_accuracy", 0.0) >= 0.017
        or improvements.get("clean_holdout_f1", 0.0) >= 0.017
        or improvements.get("clean_holdout_log_loss", 0.0) >= 0.020
        or improvements.get("clean_holdout_brier_score", 0.0) >= 0.008
    ):
        quality = "repair_usable"
        reason = "Oracle repair creates a visible clean_holdout improvement."

    elif max_improvement >= 0.005:
        quality = "repair_weak"
        reason = "Oracle repair has only a small measurable effect."

    else:
        quality = "repair_too_weak"
        reason = "Even the perfect oracle repair barely improves clean_holdout."

    return {
        "repair_effect_quality": quality,
        "reason": reason,
        "max_clean_holdout_improvement": float(max_improvement),
        "clean_holdout_improvements": improvements,
    }


def compute_oracle_normalized_repair(
    workflow_result: Dict[str, Any],
    oracle_result: Dict[str, Any],
    min_oracle_effect: float = 0.005,
) -> Dict[str, Dict[str, float | None]]:
    """Compare workflow fix impact against oracle fix impact.

    Formula:

        normalized = workflow_improvement / oracle_improvement

    Interpretation:
    - 1.0 means workflow achieved the same improvement as oracle
    - 0.5 means workflow achieved about half of oracle potential
    - 0.0 means no useful repair improvement
    - negative means workflow moved in the wrong direction
    - None means oracle effect was too small to interpret
    """
    workflow_impact = workflow_result.get("fix_impact", {})
    oracle_impact = oracle_result.get("fix_impact", {})

    metrics_to_compare = [
        "clean_holdout_accuracy",
        "clean_holdout_balanced_accuracy",
        "clean_holdout_f1",
        "clean_holdout_roc_auc",
        "clean_holdout_log_loss",
        "clean_holdout_brier_score",
    ]

    normalized: Dict[str, Dict[str, float | None]] = {}

    for metric_name in metrics_to_compare:
        if metric_name not in workflow_impact or metric_name not in oracle_impact:
            continue

        workflow_delta = float(workflow_impact[metric_name])
        oracle_delta = float(oracle_impact[metric_name])

        workflow_improvement = metric_improvement(metric_name, workflow_delta)
        oracle_improvement = metric_improvement(metric_name, oracle_delta)

        if oracle_improvement <= min_oracle_effect:
            normalized_value = None
        else:
            normalized_value = float(workflow_improvement / oracle_improvement)

        normalized[metric_name] = {
            "workflow_delta": workflow_delta,
            "oracle_delta": oracle_delta,
            "workflow_improvement": workflow_improvement,
            "oracle_improvement": oracle_improvement,
            "oracle_normalized_value": normalized_value,
        }

    return normalized


def add_oracle_context_to_workflow_result(
    workflow_result: Dict[str, Any],
    oracle_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach oracle-normalized repair metrics to a Baseline/XAI result."""
    enriched = dict(workflow_result)

    repair_quality = classify_oracle_repair_potential(oracle_result)
    normalized_repair = compute_oracle_normalized_repair(workflow_result, oracle_result)

    enriched["oracle_repair_quality"] = repair_quality
    enriched["oracle_normalized_repair"] = normalized_repair

    # Add easy-to-use flat fields for CSV analysis.
    for metric_name, values in normalized_repair.items():
        normalized_value = values["oracle_normalized_value"]
        enriched[f"oracle_normalized_{metric_name}"] = normalized_value
        enriched[f"oracle_delta_{metric_name}"] = values["oracle_delta"]
        enriched[f"workflow_improvement_{metric_name}"] = values["workflow_improvement"]
        enriched[f"oracle_improvement_{metric_name}"] = values["oracle_improvement"]

    enriched["repair_effect_quality"] = repair_quality["repair_effect_quality"]
    enriched["repair_effect_reason"] = repair_quality["reason"]

    return enriched


def build_method_comparison_summary(
    baseline_result: Dict[str, Any],
    xai_result: Dict[str, Any],
    oracle_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact Baseline-vs-XAI comparison summary."""
    baseline_steps = baseline_result.get("steps_to_detect")
    xai_steps = xai_result.get("steps_to_detect")

    if isinstance(baseline_steps, (int, float)) and isinstance(xai_steps, (int, float)):
        if baseline_steps >= 0 and xai_steps >= 0:
            steps_saved_by_xai = float(baseline_steps) - float(xai_steps)
        else:
            steps_saved_by_xai = None
    else:
        steps_saved_by_xai = None

    baseline_runtime = float(baseline_result.get("runtime_sec", 0.0))
    xai_runtime = float(xai_result.get("runtime_sec", 0.0))

    runtime_ratio = None
    if baseline_runtime > 1e-12:
        runtime_ratio = xai_runtime / baseline_runtime

    comparison = {
        "baseline_steps_to_detect": baseline_steps,
        "xai_steps_to_detect": xai_steps,
        "steps_saved_by_xai": steps_saved_by_xai,
        "baseline_mrr": baseline_result.get("mrr"),
        "xai_mrr": xai_result.get("mrr"),
        "mrr_delta_xai_minus_baseline": float(xai_result.get("mrr", 0.0)) - float(baseline_result.get("mrr", 0.0)),
        "baseline_hit_at_10": baseline_result.get("hit_at_10"),
        "xai_hit_at_10": xai_result.get("hit_at_10"),
        "hit_at_10_delta_xai_minus_baseline": int(xai_result.get("hit_at_10", 0)) - int(baseline_result.get("hit_at_10", 0)),
        "baseline_precision_at_k": baseline_result.get("precision_at_k"),
        "xai_precision_at_k": xai_result.get("precision_at_k"),
        "precision_at_k_delta_xai_minus_baseline": None,
        "baseline_runtime_sec": baseline_runtime,
        "xai_runtime_sec": xai_runtime,
        "xai_runtime_overhead_sec": xai_runtime - baseline_runtime,
        "xai_runtime_ratio": runtime_ratio,
        "oracle_repair_quality": classify_oracle_repair_potential(oracle_result),
    }

    if baseline_result.get("precision_at_k") is not None and xai_result.get("precision_at_k") is not None:
        comparison["precision_at_k_delta_xai_minus_baseline"] = (
            float(xai_result.get("precision_at_k", 0.0))
            - float(baseline_result.get("precision_at_k", 0.0))
        )

    return comparison
