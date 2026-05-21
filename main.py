"""Main orchestration for fault-injection ML pipeline testing.

This module provides:
- dataset loading and three-way splitting
- fault injection via faults.py
- model training and evaluation
- baseline debugging
- SHAP/XAI debugging
- oracle repair evaluation
- CSV/JSONL-ready experiment output

Important methodological distinction:
- baseline_debugging and xai_debugging simulate the developer view.
- oracle repair and clean_holdout evaluation are experimentator-only post-hoc analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import faults
import baseline_debugging
import xai_debugging
from evaluation_metrics import (
    ORDERED_METRICS,
    add_oracle_context_to_workflow_result,
    build_method_comparison_summary,
    compute_fix_impact,
    compute_split_metrics as shared_compute_split_metrics,
    delta_as_improvement_label,
    evaluate_classifier,
    format_delta_direction,
)


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------
# Central configuration
# ---------------------
RANDOM_STATE = 42
TRAIN_SIZE = 0.60
CONTAMINATED_EVAL_SIZE = 0.20
CLEAN_HOLDOUT_SIZE = 0.20
N_ESTIMATORS = 200

# Fault selection (can be overridden via CLI)
FAULT_TYPE = "none"  # options: "none", "label_noise", "data_leakage", "spurious_correlation"

# Label noise config
LABEL_NOISE_RATE = 0.10
LABEL_NOISE_MODE = "random"  # options: "random", "hard"
PROBE_N_ESTIMATORS = 30

# Data leakage config
LEAKAGE_MODE = "indirect"  # options: "direct", "indirect"
LEAKAGE_STRENGTH = 0.80
INDIRECT_LEAKAGE_BINS = 10
INDIRECT_LEAKAGE_CONTAM_SMOOTHING = 0.75
INDIRECT_LEAKAGE_CLEAN_SMOOTHING = 18.0
INDIRECT_LEAKAGE_HOLDOUT_SHRINK = 0.96
INDIRECT_LEAKAGE_NOISE_STD = 0.05

# Subgroup-local indirect leakage
INDIRECT_LEAKAGE_ACTIVE_GROUPS = 1
INDIRECT_LEAKAGE_HOLDOUT_GROUP_SHIFT = 1
INDIRECT_LEAKAGE_OFFGROUP_SHRINK = 0.92
INDIRECT_LEAKAGE_OFFGROUP_NOISE_STD = 0.10
INDIRECT_LEAKAGE_HOLDOUT_ACTIVE_SCALE = 0.20

# Spurious correlation config
SPURIOUS_MODE = "broken"  # options: "broken", "inverted"
SPURIOUS_STRENGTH = 0.90
USE_GROUPS_FOR_SPURIOUS = True

# Subgroup-local spurious shortcut
SPURIOUS_ACTIVE_GROUPS = 1
SPURIOUS_HOLDOUT_GROUP_SHIFT = 1
SPURIOUS_OFFGROUP_SIGNAL_WEIGHT = 0.03
SPURIOUS_OFFGROUP_NOISE_STD = 0.45

BROKEN_ACTIVE_SCALE = 0.08
BROKEN_NOISE_STD = 0.38

INVERTED_GROUP_WEIGHT = 0.15
INVERTED_SIGNAL_WEIGHT = 0.16
INVERTED_NOISE_STD = 0.40

# XAI config
SHAP_BACKGROUND_SIZE = 120
LABEL_NOISE_CV_FOLDS = 5
XAI_LABEL_BASELINE_WEIGHT = 0.80
XAI_LABEL_PROFILE_WEIGHT = 0.20
XAI_LABEL_CANDIDATE_MULTIPLIER = 3.0
XAI_FEATURE_FOCUS_FRACTION = 0.20

# Sanity checks
ENABLE_FAULT_SANITY_CHECKS = True


# ---------------------
# Basic setup
# ---------------------
def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer dataset as tabular features and binary labels."""
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data.copy(), dataset.target.copy()


def split_dataset(features: pd.DataFrame, labels: pd.Series) -> Dict[str, Dict[str, pd.DataFrame | pd.Series]]:
    """Create a three-way split: train, contaminated_eval, and clean_holdout."""
    remaining_features, clean_holdout_features, remaining_labels, clean_holdout_labels = train_test_split(
        features,
        labels,
        test_size=CLEAN_HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    contaminated_eval_ratio = CONTAMINATED_EVAL_SIZE / (1.0 - CLEAN_HOLDOUT_SIZE)
    train_features, contaminated_eval_features, train_labels, contaminated_eval_labels = train_test_split(
        remaining_features,
        remaining_labels,
        test_size=contaminated_eval_ratio,
        random_state=RANDOM_STATE,
        stratify=remaining_labels,
    )

    return {
        "train": {"features": train_features, "labels": train_labels},
        "contaminated_eval": {"features": contaminated_eval_features, "labels": contaminated_eval_labels},
        "clean_holdout": {"features": clean_holdout_features, "labels": clean_holdout_labels},
    }


def train_model(train_features: pd.DataFrame, train_labels: pd.Series) -> RandomForestClassifier:
    """Train a RandomForestClassifier on the given training split."""
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    model.fit(train_features, train_labels)
    return model


def evaluate_model(model: RandomForestClassifier, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
    """Evaluate a fitted classifier with the shared metric set.

    Besides accuracy, F1 and ROC-AUC, this now also reports:
    - balanced_accuracy
    - log_loss
    - brier_score
    """
    return evaluate_classifier(model, features, labels)


def compute_split_metrics(
    model: RandomForestClassifier,
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
) -> Dict[str, float]:
    """Compute flat metrics for train, contaminated_eval, and clean_holdout."""
    return shared_compute_split_metrics(model, splits)


def _print_metric_block(title: str, metrics: Dict[str, float]) -> None:
    print(title)

    for metric_name in ORDERED_METRICS:
        if metric_name in metrics:
            value = metrics[metric_name]
            if value is None or not np.isfinite(float(value)):
                formatted = "n/a"
            else:
                formatted = f"{float(value):.4f}"
            print(f"  {metric_name}: {formatted}")


# ---------------------
# Fault quality checks
# ---------------------
def _quality_label_noise(
    train_metrics: Dict[str, float],
    contaminated_eval_metrics: Dict[str, float],
    clean_holdout_metrics: Dict[str, float],
    fault_metadata: Dict[str, Any],
) -> Tuple[str, str]:
    train_gap = train_metrics["accuracy"] - clean_holdout_metrics["accuracy"]
    eval_gap = train_metrics["accuracy"] - contaminated_eval_metrics["accuracy"]
    changed_rate = float(fault_metadata["changed_count"]) / max(1, int(fault_metadata.get("train_count", 1)))
    mean_margin = float(fault_metadata.get("mean_changed_margin", 0.0))

    if eval_gap < 0.03:
        return "too_weak", "Training accuracy barely changes under label noise."
    if train_gap > 0.22 and contaminated_eval_metrics["accuracy"] < 0.80:
        return "too_strong", "Noise degrades the model too much for a stable comparison."
    if fault_metadata["noise_mode"] == "hard" and mean_margin > 0.80:
        return "too_trivial", "Hard-noise targets are not concentrated enough on uncertain samples."
    if changed_rate < 0.05:
        return "too_weak", "Too few labels were changed to matter materially."
    return "usable", "Label noise is strong enough to affect training but still leaves signal."


def _quality_leakage(
    contaminated_eval_metrics: Dict[str, float],
    clean_holdout_metrics: Dict[str, float],
    contaminated_stats: Dict[str, float],
    clean_stats: Dict[str, float],
    leakage_mode: str,
) -> Tuple[str, str]:
    eval_gain = contaminated_eval_metrics["accuracy"] - clean_holdout_metrics["accuracy"]
    contaminated_corr = abs(contaminated_stats["correlation"])
    clean_corr = abs(clean_stats["correlation"])
    corr_drop = contaminated_corr - clean_corr

    if leakage_mode == "direct":
        if contaminated_corr > 0.90 or contaminated_eval_metrics["accuracy"] > 0.98:
            return "too_trivial", "Direct leakage is nearly a label copy; too obvious for debugging."
        return "too_trivial", "Direct leakage is too straightforward to be interesting for multi-method comparison."

    if contaminated_corr < 0.15 or eval_gain < 0.010:
        return "too_weak", "Indirect leakage signal is too weak to separate offline and clean views."
    if contaminated_corr > 0.88 or eval_gain > 0.12:
        return "too_strong", "Indirect leakage is too dominant; leaves little room for debugging insights."
    if corr_drop < 0.08:
        return "too_weak", "Indirect leakage behaves too similarly on contaminated_eval and clean_holdout."
    return "usable", "Indirect leakage creates a meaningful offline/clean gap without being globally obvious."


def _quality_spurious(
    train_metrics: Dict[str, float],
    contaminated_eval_metrics: Dict[str, float],
    clean_holdout_metrics: Dict[str, float],
    train_stats: Dict[str, float],
    contaminated_stats: Dict[str, float],
    clean_stats: Dict[str, float],
    spurious_mode: str,
) -> Tuple[str, str]:
    train_corr = abs(train_stats["correlation"])
    contaminated_corr = abs(contaminated_stats["correlation"])
    holdout_corr = abs(clean_stats["correlation"])
    holdout_gap = train_metrics["accuracy"] - clean_holdout_metrics["accuracy"]
    holdout_negative_corr = clean_stats["correlation"]

    if train_corr < 0.15 or contaminated_corr < 0.15:
        return "too_weak", "Shortcut feature is not attractive enough in the offline world."

    if spurious_mode == "broken":
        if holdout_corr > 0.12:
            return "too_weak", "Holdout shortcut is not broken enough; residual correlation remains."
        if holdout_gap < 0.04:
            return "too_weak", "The subgroup shortcut does not hurt holdout enough."
        return "usable", "Shortcut is subgroup-local offline and breaks meaningfully on holdout."

    if spurious_mode == "inverted":
        if holdout_negative_corr > -0.08:
            return "too_weak", "Inversion in holdout is too weak."
        if holdout_gap > 0.24 or clean_holdout_metrics["accuracy"] < 0.70:
            return "too_strong", "Inversion effect is still too brutal; holdout fails too badly."
        return "usable", "Inversion is visible but still realistic for debugging."

    return "usable", "Spurious correlation shows a realistic train/holdout shift."


def assess_fault_quality(
    fault_type: str,
    train_metrics: Dict[str, float],
    contaminated_eval_metrics: Dict[str, float],
    clean_holdout_metrics: Dict[str, float],
    fault_metadata: Dict[str, Any],
    diagnostics: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    if fault_type == "label_noise":
        quality, reason = _quality_label_noise(
            train_metrics,
            contaminated_eval_metrics,
            clean_holdout_metrics,
            fault_metadata,
        )
    elif fault_type == "data_leakage":
        quality, reason = _quality_leakage(
            contaminated_eval_metrics,
            clean_holdout_metrics,
            diagnostics["contaminated_eval"],
            diagnostics["clean_holdout"],
            fault_metadata.get("leakage_mode", "direct"),
        )
    elif fault_type == "spurious_correlation":
        quality, reason = _quality_spurious(
            train_metrics,
            contaminated_eval_metrics,
            clean_holdout_metrics,
            diagnostics["train"],
            diagnostics["contaminated_eval"],
            diagnostics["clean_holdout"],
            fault_metadata.get("spurious_mode", "broken"),
        )
    else:
        quality, reason = "usable", "No fault injected."

    return {"quality": quality, "reason": reason}


def print_fault_diagnostics(
    fault_type: str,
    fault_metadata: Dict[str, Any],
    train_metrics: Dict[str, float],
    contaminated_eval_metrics: Dict[str, float],
    clean_holdout_metrics: Dict[str, float],
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
) -> None:
    print("\n--- Fault Sanity Check ---")
    print(f"Fault type: {fault_type}")
    print(f"Train -> contaminated_eval accuracy gap: {train_metrics['accuracy'] - contaminated_eval_metrics['accuracy']:.4f}")
    print(f"Train -> clean_holdout accuracy gap: {train_metrics['accuracy'] - clean_holdout_metrics['accuracy']:.4f}")
    print(f"contaminated_eval -> clean_holdout accuracy gap: {contaminated_eval_metrics['accuracy'] - clean_holdout_metrics['accuracy']:.4f}")

    diagnostics: Dict[str, Dict[str, float]] = {}

    if fault_type == "label_noise":
        print(f"Label noise mode: {fault_metadata.get('noise_mode')}")
        print(f"Changed labels: {fault_metadata.get('changed_count')}")
        if "mean_changed_margin" in fault_metadata:
            print(f"Mean margin of changed samples: {fault_metadata['mean_changed_margin']:.4f} (lower means harder / more uncertain)")
        quality_report = assess_fault_quality(
            fault_type,
            train_metrics,
            contaminated_eval_metrics,
            clean_holdout_metrics,
            fault_metadata,
            diagnostics,
        )
        print(f"Fault quality: {quality_report['quality']} - {quality_report['reason']}")

    elif fault_type == "data_leakage":
        feature_name = fault_metadata.get("leakage_feature_name")
        if feature_name:
            for split_name in ("contaminated_eval", "clean_holdout"):
                feature = splits[split_name]["features"][feature_name]
                labels = splits[split_name]["labels"]
                stats = faults._feature_proxy(feature, labels)
                diagnostics[split_name] = stats
                print(
                    f"{split_name} leakage stats - corr: {stats['correlation']:.4f}, "
                    f"mi: {stats['mutual_information']:.4f}, proxy: {stats['mean_gap']:.4f}"
                )
            quality_report = assess_fault_quality(
                fault_type,
                train_metrics,
                contaminated_eval_metrics,
                clean_holdout_metrics,
                fault_metadata,
                diagnostics,
            )
            print(f"Leakage assessment: {quality_report['quality']} - {quality_report['reason']}")
            print(f"Fault quality: {quality_report['quality']} - {quality_report['reason']}")

    elif fault_type == "spurious_correlation":
        feature_name = fault_metadata.get("feature_name")
        if feature_name:
            for split_name in ("train", "contaminated_eval", "clean_holdout"):
                feature = splits[split_name]["features"][feature_name]
                labels = splits[split_name]["labels"]
                stats = faults._feature_proxy(feature, labels)
                diagnostics[split_name] = stats
                print(
                    f"{split_name} spurious stats - corr: {stats['correlation']:.4f}, "
                    f"mi: {stats['mutual_information']:.4f}, proxy: {stats['mean_gap']:.4f}"
                )
            print(f"Holdout behavior: {fault_metadata.get('holdout_behavior')}")
            quality_report = assess_fault_quality(
                fault_type,
                train_metrics,
                contaminated_eval_metrics,
                clean_holdout_metrics,
                fault_metadata,
                diagnostics,
            )
            print(f"Fault quality: {quality_report['quality']} - {quality_report['reason']}")

    print("--- End Sanity Check ---\n")


# ---------------------
# Fault injection
# ---------------------
def build_fault_config() -> Dict[str, Any]:
    return {
        "RANDOM_STATE": RANDOM_STATE,
        "LABEL_NOISE_RATE": LABEL_NOISE_RATE,
        "LABEL_NOISE_MODE": LABEL_NOISE_MODE,
        "PROBE_N_ESTIMATORS": PROBE_N_ESTIMATORS,
        "LEAKAGE_MODE": LEAKAGE_MODE,
        "LEAKAGE_STRENGTH": LEAKAGE_STRENGTH,
        "INDIRECT_LEAKAGE_BINS": INDIRECT_LEAKAGE_BINS,
        "INDIRECT_LEAKAGE_CONTAM_SMOOTHING": INDIRECT_LEAKAGE_CONTAM_SMOOTHING,
        "INDIRECT_LEAKAGE_CLEAN_SMOOTHING": INDIRECT_LEAKAGE_CLEAN_SMOOTHING,
        "INDIRECT_LEAKAGE_HOLDOUT_SHRINK": INDIRECT_LEAKAGE_HOLDOUT_SHRINK,
        "INDIRECT_LEAKAGE_NOISE_STD": INDIRECT_LEAKAGE_NOISE_STD,
        "INDIRECT_LEAKAGE_ACTIVE_GROUPS": INDIRECT_LEAKAGE_ACTIVE_GROUPS,
        "INDIRECT_LEAKAGE_HOLDOUT_GROUP_SHIFT": INDIRECT_LEAKAGE_HOLDOUT_GROUP_SHIFT,
        "INDIRECT_LEAKAGE_OFFGROUP_SHRINK": INDIRECT_LEAKAGE_OFFGROUP_SHRINK,
        "INDIRECT_LEAKAGE_OFFGROUP_NOISE_STD": INDIRECT_LEAKAGE_OFFGROUP_NOISE_STD,
        "INDIRECT_LEAKAGE_HOLDOUT_ACTIVE_SCALE": INDIRECT_LEAKAGE_HOLDOUT_ACTIVE_SCALE,
        "SPURIOUS_MODE": SPURIOUS_MODE,
        "SPURIOUS_STRENGTH": SPURIOUS_STRENGTH,
        "USE_GROUPS_FOR_SPURIOUS": USE_GROUPS_FOR_SPURIOUS,
        "SPURIOUS_ACTIVE_GROUPS": SPURIOUS_ACTIVE_GROUPS,
        "SPURIOUS_HOLDOUT_GROUP_SHIFT": SPURIOUS_HOLDOUT_GROUP_SHIFT,
        "SPURIOUS_OFFGROUP_SIGNAL_WEIGHT": SPURIOUS_OFFGROUP_SIGNAL_WEIGHT,
        "SPURIOUS_OFFGROUP_NOISE_STD": SPURIOUS_OFFGROUP_NOISE_STD,
        "BROKEN_ACTIVE_SCALE": BROKEN_ACTIVE_SCALE,
        "BROKEN_NOISE_STD": BROKEN_NOISE_STD,
        "INVERTED_GROUP_WEIGHT": INVERTED_GROUP_WEIGHT,
        "INVERTED_SIGNAL_WEIGHT": INVERTED_SIGNAL_WEIGHT,
        "INVERTED_NOISE_STD": INVERTED_NOISE_STD,
    }


def apply_fault_injection(
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
) -> Tuple[Dict[str, Dict[str, pd.DataFrame | pd.Series]], Dict[str, Any]]:
    """Apply fault injection using faults.py with config dictionary."""
    config = build_fault_config()

    if FAULT_TYPE == "label_noise":
        noisy_train_labels, metadata = faults._inject_label_noise(
            splits["train"]["features"],
            splits["train"]["labels"],
            config,
        )
        injected_splits = {
            "train": {"features": splits["train"]["features"], "labels": noisy_train_labels},
            "contaminated_eval": splits["contaminated_eval"],
            "clean_holdout": splits["clean_holdout"],
        }
        return injected_splits, metadata

    if FAULT_TYPE == "data_leakage":
        return faults._inject_data_leakage(
            splits["train"],
            splits["contaminated_eval"],
            splits["clean_holdout"],
            config,
        )

    if FAULT_TYPE == "spurious_correlation":
        return faults._inject_spurious_correlation(
            splits["train"],
            splits["contaminated_eval"],
            splits["clean_holdout"],
            config,
        )

    metadata = {
        "fault_type": "none",
        "injected": False,
        "contaminated_splits": [],
        "contaminated_eval_contaminated": False,
        "clean_holdout_contaminated": False,
    }
    return splits, metadata


# ---------------------
# New: true detection metrics
# ---------------------
def _get_true_feature_name(fault_type: str, fault_metadata: Dict[str, Any]) -> str | None:
    if fault_type == "data_leakage":
        return fault_metadata.get("leakage_feature_name")
    if fault_type == "spurious_correlation":
        return fault_metadata.get("feature_name")
    return None


def _compute_feature_detection_metrics(
    result: Dict[str, Any],
    true_feature: str | None,
) -> Dict[str, Any]:
    suspect_features = result.get("suspect_features", []) or []

    if true_feature is None:
        rank = -1
    else:
        try:
            rank = suspect_features.index(true_feature) + 1
        except ValueError:
            rank = -1

    if rank > 0:
        mrr = 1.0 / float(rank)
        hit_at_1 = 1 if rank <= 1 else 0
        hit_at_3 = 1 if rank <= 3 else 0
        hit_at_5 = 1 if rank <= 5 else 0
        hit_at_10 = 1 if rank <= 10 else 0
        steps_to_detect = rank
    else:
        mrr = 0.0
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0
        hit_at_10 = 0
        steps_to_detect = -1

    return {
        "true_feature": true_feature,
        "rank_true_feature": rank,
        "steps_to_detect": steps_to_detect,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "mrr": mrr,
    }


def _compute_label_noise_detection_metrics(
    result: Dict[str, Any],
    fault_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    suspect_indices = [int(idx) for idx in result.get("suspect_indices", [])]
    true_indices = set(int(idx) for idx in fault_metadata.get("changed_indices", []))

    if not true_indices:
        return {
            "first_true_label_rank": -1,
            "steps_to_detect": -1,
            "hit_at_1": 0,
            "hit_at_3": 0,
            "hit_at_5": 0,
            "hit_at_10": 0,
            "mrr": 0.0,
        }

    first_rank = -1
    for position, idx in enumerate(suspect_indices, start=1):
        if idx in true_indices:
            first_rank = position
            break

    if first_rank > 0:
        mrr = 1.0 / float(first_rank)
    else:
        mrr = 0.0

    return {
        "first_true_label_rank": first_rank,
        "steps_to_detect": first_rank,
        "hit_at_1": 1 if 0 < first_rank <= 1 else 0,
        "hit_at_3": 1 if 0 < first_rank <= 3 else 0,
        "hit_at_5": 1 if 0 < first_rank <= 5 else 0,
        "hit_at_10": 1 if 0 < first_rank <= 10 else 0,
        "mrr": mrr,
    }


def enrich_debugging_result(
    result: Dict[str, Any],
    fault_type: str,
    fault_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Replace placeholder steps_to_detect with real detection metrics.

    Feature faults:
    - steps_to_detect = rank of true fault feature

    Label noise:
    - steps_to_detect = first rank at which a truly corrupted label appears
    - precision@k and recall@k still measure broader top-k quality
    """
    enriched = dict(result)

    if fault_type in ("data_leakage", "spurious_correlation"):
        true_feature = _get_true_feature_name(fault_type, fault_metadata)
        enriched.update(_compute_feature_detection_metrics(enriched, true_feature))

    elif fault_type == "label_noise":
        enriched.update(_compute_label_noise_detection_metrics(enriched, fault_metadata))

    else:
        enriched["steps_to_detect"] = 0
        enriched["hit_at_1"] = 0
        enriched["hit_at_3"] = 0
        enriched["hit_at_5"] = 0
        enriched["hit_at_10"] = 0
        enriched["mrr"] = 0.0

    return enriched


# ---------------------
# New: oracle repair
# ---------------------
def _drop_feature_from_splits(
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    feature_name: str,
) -> Dict[str, Dict[str, pd.DataFrame | pd.Series]]:
    fixed_splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]] = {}

    for split_name, split_data in splits.items():
        features = split_data["features"]
        labels = split_data["labels"]

        if feature_name in features.columns:
            fixed_features = features.drop(columns=[feature_name])
        else:
            fixed_features = features.copy()

        fixed_splits[split_name] = {
            "features": fixed_features,
            "labels": labels,
        }

    return fixed_splits


def _apply_oracle_label_fix(
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    fault_metadata: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, pd.DataFrame | pd.Series]], List[int]]:
    fixed_labels = splits["train"]["labels"].copy()
    corrected_indices: List[int] = []

    original_labels = {
        int(idx): int(label)
        for idx, label in fault_metadata.get("original_labels_by_index", {}).items()
    }

    for idx, original_label in original_labels.items():
        if idx in fixed_labels.index:
            fixed_labels.loc[idx] = original_label
            corrected_indices.append(idx)

    fixed_splits = {
        "train": {
            "features": splits["train"]["features"],
            "labels": fixed_labels,
        },
        "contaminated_eval": splits["contaminated_eval"],
        "clean_holdout": splits["clean_holdout"],
    }

    return fixed_splits, corrected_indices


def run_oracle_repair(
    model: RandomForestClassifier,
    fault_type: str,
    fault_metadata: Dict[str, Any],
    injected_splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
) -> Dict[str, Any]:
    """Experimentator-only oracle repair.

    This is NOT a developer workflow.
    It answers: what would happen if the true injected root cause were fixed perfectly?

    Use this to separate:
    - localization quality
    - workflow repair quality
    - theoretical repair potential
    """
    start_time = time.time()
    metrics_before = compute_split_metrics(model, injected_splits)

    if fault_type == "none":
        metrics_after = metrics_before
        fix_impact = {key: 0.0 for key in metrics_before.keys()}
        return {
            "workflow": "oracle_true_fix",
            "fault_type": fault_type,
            "oracle_target": None,
            "oracle_fix_applied": False,
            "oracle_fix_count": 0,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "fix_impact": fix_impact,
            "steps_to_detect": 0,
            "retrains": 0,
            "runtime_sec": float(time.time() - start_time),
        }

    if fault_type == "label_noise":
        fixed_splits, corrected_indices = _apply_oracle_label_fix(injected_splits, fault_metadata)
        retrained_model = train_model(fixed_splits["train"]["features"], fixed_splits["train"]["labels"])
        metrics_after = compute_split_metrics(retrained_model, fixed_splits)
        oracle_target = "all_true_noisy_labels"
        oracle_fix_count = len(corrected_indices)
        oracle_fix_applied = oracle_fix_count > 0

    elif fault_type in ("data_leakage", "spurious_correlation"):
        true_feature = _get_true_feature_name(fault_type, fault_metadata)
        if true_feature is None:
            metrics_after = metrics_before
            oracle_target = None
            oracle_fix_count = 0
            oracle_fix_applied = False
        else:
            fixed_splits = _drop_feature_from_splits(injected_splits, true_feature)
            retrained_model = train_model(fixed_splits["train"]["features"], fixed_splits["train"]["labels"])
            metrics_after = compute_split_metrics(retrained_model, fixed_splits)
            oracle_target = true_feature
            oracle_fix_count = 1
            oracle_fix_applied = True

    else:
        metrics_after = metrics_before
        oracle_target = None
        oracle_fix_count = 0
        oracle_fix_applied = False

    fix_impact = compute_fix_impact(metrics_before, metrics_after)

    return {
        "workflow": "oracle_true_fix",
        "fault_type": fault_type,
        "oracle_target": oracle_target,
        "oracle_fix_applied": oracle_fix_applied,
        "oracle_fix_count": oracle_fix_count,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "fix_impact": fix_impact,
        "steps_to_detect": 0,
        "retrains": 1 if oracle_fix_applied else 0,
        "runtime_sec": float(time.time() - start_time),
    }


# ---------------------
# Printing
# ---------------------
def _format_metric_value(value: float | None) -> str:
    """Format metric values safely for printing."""
    if value is None:
        return "n/a"

    try:
        number = float(value)
    except Exception:
        return "n/a"

    if not np.isfinite(number):
        return "n/a"

    return f"{number:.4f}"


def _print_split_metrics_from_flat_dict(
    metrics: Dict[str, float],
    split_name: str,
) -> None:
    """Print all metrics for one split from a flat metric dictionary."""
    pieces = []

    for metric_name in ORDERED_METRICS:
        full_name = f"{split_name}_{metric_name}"
        if full_name in metrics:
            pieces.append(f"{metric_name}={_format_metric_value(metrics[full_name])}")

    print(f"  {split_name}: " + ", ".join(pieces))


def _print_fix_impact(fix_impact: Dict[str, float]) -> None:
    """Print fix impact with correct improvement direction.

    Important:
    - For accuracy/F1/AUC: higher is better.
    - For log_loss/Brier: lower is better.
    """
    for metric_name, delta in fix_impact.items():
        direction = format_delta_direction(metric_name, float(delta))
        label = delta_as_improvement_label(metric_name, float(delta))
        print(f"  {metric_name}: {float(delta):+.4f} ({direction}, {label})")


def _print_oracle_normalized_repair(result: Dict[str, Any]) -> None:
    """Print oracle-normalized repair information if available."""
    normalized = result.get("oracle_normalized_repair", {})
    if not normalized:
        return

    print("\nOracle-normalized repair impact:")
    print("  Interpretation: 1.0 = same improvement as perfect oracle fix; 0.0 = no useful repair effect.")

    for metric_name, values in normalized.items():
        normalized_value = values.get("oracle_normalized_value")
        workflow_improvement = values.get("workflow_improvement")
        oracle_improvement = values.get("oracle_improvement")

        if normalized_value is None:
            normalized_text = "n/a (oracle effect too small)"
        else:
            normalized_text = f"{float(normalized_value):.4f}"

        print(
            f"  {metric_name}: normalized={normalized_text}, "
            f"workflow_improvement={_format_metric_value(workflow_improvement)}, "
            f"oracle_improvement={_format_metric_value(oracle_improvement)}"
        )

    if "repair_effect_quality" in result:
        print(f"  repair_effect_quality: {result['repair_effect_quality']}")
        print(f"  repair_effect_reason: {result.get('repair_effect_reason', 'n/a')}")


def _print_debugging_result(title: str, result: Dict[str, Any], fault_type: str) -> None:
    print(f"\n{'*' * 70}")
    print(title)
    print(f"{'*' * 70}")

    print(f"\nDetection steps: {result.get('steps_to_detect')}")
    print(f"MRR: {_format_metric_value(result.get('mrr', 0.0))}")
    print(
        f"Hit@1: {result.get('hit_at_1', 0)} | "
        f"Hit@3: {result.get('hit_at_3', 0)} | "
        f"Hit@5: {result.get('hit_at_5', 0)} | "
        f"Hit@10: {result.get('hit_at_10', 0)}"
    )
    print(f"Retrains: {result['retrains']}")
    print(f"Runtime: {result['runtime_sec']:.3f} seconds")

    if fault_type == "label_noise":
        print("\nLabel Noise Detection:")
        print(f"  Precision@k: {_format_metric_value(result.get('precision_at_k', 0.0))}")
        print(f"  Recall@k: {_format_metric_value(result.get('recall_at_k', 0.0))}")
        print(f"  First true label rank: {result.get('first_true_label_rank', -1)}")
        print(f"  Top suspect indices: {result.get('suspect_indices', [])[:5]}")

    elif fault_type == "data_leakage":
        print("\nData Leakage Detection:")
        print(f"  True feature: {result.get('true_feature')}")
        print(f"  Rank of true feature: {result.get('rank_true_feature', -1)}")
        print(f"  Top candidate feature: {result.get('top_candidate_feature')}")
        print(f"  Top-5 suspect features: {result.get('suspect_features', [])[:5]}")

    elif fault_type == "spurious_correlation":
        print("\nSpurious Correlation Detection:")
        print(f"  True feature: {result.get('true_feature')}")
        print(f"  Rank of true feature: {result.get('rank_true_feature', -1)}")
        print(f"  Top candidate feature: {result.get('top_candidate_feature')}")
        print(f"  Top-5 suspect features: {result.get('suspect_features', [])[:5]}")

    print("\nMetrics before fix:")
    for split_name in ["train", "contaminated_eval", "clean_holdout"]:
        _print_split_metrics_from_flat_dict(result["metrics_before"], split_name)

    print("\nMetrics after fix:")
    for split_name in ["train", "contaminated_eval", "clean_holdout"]:
        _print_split_metrics_from_flat_dict(result["metrics_after"], split_name)

    print("\nFix impact (delta):")
    _print_fix_impact(result["fix_impact"])

    _print_oracle_normalized_repair(result)


def _print_oracle_result(result: Dict[str, Any]) -> None:
    print(f"\n{'*' * 70}")
    print("ORACLE REPAIR (EXPERIMENTATOR ONLY)")
    print(f"{'*' * 70}")

    print(f"\nOracle target: {result.get('oracle_target')}")
    print(f"Oracle fix applied: {result.get('oracle_fix_applied')}")
    print(f"Oracle fix count: {result.get('oracle_fix_count')}")
    print(f"Runtime: {result['runtime_sec']:.3f} seconds")

    print("\nOracle fix impact (delta):")
    _print_fix_impact(result["fix_impact"])


def _print_method_comparison_summary(summary: Dict[str, Any]) -> None:
    """Print compact Baseline-vs-XAI comparison summary."""
    print(f"\n{'*' * 70}")
    print("BASELINE VS. XAI SUMMARY")
    print(f"{'*' * 70}")

    print("\nLocalization:")
    print(f"  Baseline steps_to_detect: {summary.get('baseline_steps_to_detect')}")
    print(f"  XAI steps_to_detect: {summary.get('xai_steps_to_detect')}")
    print(f"  Steps saved by XAI: {summary.get('steps_saved_by_xai')}")
    print(f"  Baseline MRR: {_format_metric_value(summary.get('baseline_mrr'))}")
    print(f"  XAI MRR: {_format_metric_value(summary.get('xai_mrr'))}")
    print(f"  MRR delta XAI - Baseline: {_format_metric_value(summary.get('mrr_delta_xai_minus_baseline'))}")
    print(f"  Hit@10 delta XAI - Baseline: {summary.get('hit_at_10_delta_xai_minus_baseline')}")

    if summary.get("baseline_precision_at_k") is not None:
        print("\nLabel-noise top-k quality:")
        print(f"  Baseline Precision@k: {_format_metric_value(summary.get('baseline_precision_at_k'))}")
        print(f"  XAI Precision@k: {_format_metric_value(summary.get('xai_precision_at_k'))}")
        print(
            "  Precision@k delta XAI - Baseline: "
            f"{_format_metric_value(summary.get('precision_at_k_delta_xai_minus_baseline'))}"
        )

    print("\nRuntime:")
    print(f"  Baseline runtime: {_format_metric_value(summary.get('baseline_runtime_sec'))} sec")
    print(f"  XAI runtime: {_format_metric_value(summary.get('xai_runtime_sec'))} sec")
    print(f"  XAI overhead: {_format_metric_value(summary.get('xai_runtime_overhead_sec'))} sec")
    print(f"  XAI runtime ratio: {_format_metric_value(summary.get('xai_runtime_ratio'))}")

    oracle_quality = summary.get("oracle_repair_quality", {})
    print("\nOracle repair potential:")
    print(f"  Quality: {oracle_quality.get('repair_effect_quality')}")
    print(f"  Reason: {oracle_quality.get('reason')}")
    print(f"  Max clean-holdout improvement: {_format_metric_value(oracle_quality.get('max_clean_holdout_improvement'))}")


# ---------------------
# CSV / JSONL output
# ---------------------
def _current_mode_label() -> str:
    if FAULT_TYPE == "label_noise":
        return LABEL_NOISE_MODE
    if FAULT_TYPE == "data_leakage":
        return LEAKAGE_MODE
    if FAULT_TYPE == "spurious_correlation":
        return SPURIOUS_MODE
    return "none"


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return str(value)


def _row_from_result(
    workflow_result: Dict[str, Any],
    seed: int,
    fault_type: str,
    fault_mode: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "seed": seed,
        "fault_type": fault_type,
        "fault_mode": fault_mode,
        "workflow": workflow_result.get("workflow"),
        "runtime_sec": workflow_result.get("runtime_sec"),
        "retrains": workflow_result.get("retrains"),
        "steps_to_detect": workflow_result.get("steps_to_detect"),
        "mrr": workflow_result.get("mrr"),
        "hit_at_1": workflow_result.get("hit_at_1"),
        "hit_at_3": workflow_result.get("hit_at_3"),
        "hit_at_5": workflow_result.get("hit_at_5"),
        "hit_at_10": workflow_result.get("hit_at_10"),
        "precision_at_k": workflow_result.get("precision_at_k"),
        "recall_at_k": workflow_result.get("recall_at_k"),
        "first_true_label_rank": workflow_result.get("first_true_label_rank"),
        "rank_true_feature": workflow_result.get("rank_true_feature"),
        "true_feature": workflow_result.get("true_feature"),
        "top_candidate_feature": workflow_result.get("top_candidate_feature"),
        "oracle_target": workflow_result.get("oracle_target"),
        "oracle_fix_applied": workflow_result.get("oracle_fix_applied"),
        "oracle_fix_count": workflow_result.get("oracle_fix_count"),
        "repair_effect_quality": workflow_result.get("repair_effect_quality"),
        "repair_effect_reason": workflow_result.get("repair_effect_reason"),
        "oracle_normalized_clean_holdout_accuracy": workflow_result.get("oracle_normalized_clean_holdout_accuracy"),
        "oracle_normalized_clean_holdout_balanced_accuracy": workflow_result.get("oracle_normalized_clean_holdout_balanced_accuracy"),
        "oracle_normalized_clean_holdout_f1": workflow_result.get("oracle_normalized_clean_holdout_f1"),
        "oracle_normalized_clean_holdout_roc_auc": workflow_result.get("oracle_normalized_clean_holdout_roc_auc"),
        "oracle_normalized_clean_holdout_log_loss": workflow_result.get("oracle_normalized_clean_holdout_log_loss"),
        "oracle_normalized_clean_holdout_brier_score": workflow_result.get("oracle_normalized_clean_holdout_brier_score"),
    }

    for metric_name, value in workflow_result.get("metrics_before", {}).items():
        row[f"before_{metric_name}"] = value

    for metric_name, value in workflow_result.get("metrics_after", {}).items():
        row[f"after_{metric_name}"] = value

    for metric_name, value in workflow_result.get("fix_impact", {}).items():
        row[f"delta_{metric_name}"] = value

    suspect_features = workflow_result.get("suspect_features") or []
    suspect_indices = workflow_result.get("suspect_indices") or []

    row["top5_suspect_features"] = json.dumps(suspect_features[:5], ensure_ascii=False)
    row["top5_suspect_indices"] = json.dumps(suspect_indices[:5], ensure_ascii=False)

    return row


def build_output_rows(
    seed: int,
    fault_type: str,
    fault_mode: str,
    baseline_result: Dict[str, Any],
    xai_result: Dict[str, Any],
    oracle_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        _row_from_result(baseline_result, seed, fault_type, fault_mode),
        _row_from_result(xai_result, seed, fault_type, fault_mode),
        _row_from_result(oracle_result, seed, fault_type, fault_mode),
    ]


def append_rows_to_csv(rows: List[Dict[str, Any]], output_csv: str | None) -> None:
    if not output_csv:
        return

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=write_header, index=False, encoding="utf-8")


def append_result_to_jsonl(result: Dict[str, Any], output_jsonl: str | None) -> None:
    if not output_jsonl:
        return

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False, default=_safe_json_value))
        file.write("\n")


# ---------------------
# Main
# ---------------------
def run_single_experiment(
    output_csv: str | None = None,
    output_jsonl: str | None = None,
) -> Dict[str, Any]:
    """Run one experiment configuration and optionally append CSV/JSONL outputs."""
    features, labels = load_dataset()
    splits = split_dataset(features, labels)
    injected_splits, fault_metadata = apply_fault_injection(splits)

    model = train_model(
        injected_splits["train"]["features"],
        injected_splits["train"]["labels"],
    )

    train_metrics = evaluate_model(
        model,
        injected_splits["train"]["features"],
        injected_splits["train"]["labels"],
    )
    contaminated_eval_metrics = evaluate_model(
        model,
        injected_splits["contaminated_eval"]["features"],
        injected_splits["contaminated_eval"]["labels"],
    )
    clean_holdout_metrics = evaluate_model(
        model,
        injected_splits["clean_holdout"]["features"],
        injected_splits["clean_holdout"]["labels"],
    )

    print(f"\n{'*' * 70}")
    print("FAULT INJECTION & INITIAL TRAINING")
    print(f"{'*' * 70}")
    print(f"Active fault type: {FAULT_TYPE}")
    print(f"Active fault mode: {_current_mode_label()}")
    print(f"Seed: {RANDOM_STATE}")
    print("\nFault metadata:")
    print(json.dumps(fault_metadata, indent=2, ensure_ascii=False))

    _print_metric_block("\nTrain metrics:", train_metrics)
    _print_metric_block("Contaminated eval metrics:", contaminated_eval_metrics)
    _print_metric_block("Clean holdout metrics:", clean_holdout_metrics)

    print("\nAccuracy gap summary:")
    print(f"  train -> contaminated_eval: {train_metrics['accuracy'] - contaminated_eval_metrics['accuracy']:.4f}")
    print(f"  train -> clean_holdout: {train_metrics['accuracy'] - clean_holdout_metrics['accuracy']:.4f}")
    print(f"  contaminated_eval -> clean_holdout: {contaminated_eval_metrics['accuracy'] - clean_holdout_metrics['accuracy']:.4f}")

    if ENABLE_FAULT_SANITY_CHECKS:
        print_fault_diagnostics(
            FAULT_TYPE,
            fault_metadata,
            train_metrics,
            contaminated_eval_metrics,
            clean_holdout_metrics,
            injected_splits,
        )

    baseline_config = {
        "RANDOM_STATE": RANDOM_STATE,
        "N_ESTIMATORS": N_ESTIMATORS,
        "LABEL_NOISE_CV_FOLDS": LABEL_NOISE_CV_FOLDS,
    }

    xai_config = {
        "RANDOM_STATE": RANDOM_STATE,
        "N_ESTIMATORS": N_ESTIMATORS,
        "LABEL_NOISE_CV_FOLDS": LABEL_NOISE_CV_FOLDS,
        "SHAP_BACKGROUND_SIZE": SHAP_BACKGROUND_SIZE,
        "XAI_LABEL_BASELINE_WEIGHT": XAI_LABEL_BASELINE_WEIGHT,
        "XAI_LABEL_PROFILE_WEIGHT": XAI_LABEL_PROFILE_WEIGHT,
        "XAI_LABEL_CANDIDATE_MULTIPLIER": XAI_LABEL_CANDIDATE_MULTIPLIER,
        "XAI_FEATURE_FOCUS_FRACTION": XAI_FEATURE_FOCUS_FRACTION,
    }

    baseline_result_raw = baseline_debugging.run_baseline_debugging(
        model=model,
        fault_type=FAULT_TYPE,
        fault_metadata=fault_metadata,
        injected_splits=injected_splits,
        config=baseline_config,
    )
    baseline_result = enrich_debugging_result(
        baseline_result_raw,
        FAULT_TYPE,
        fault_metadata,
    )

    xai_result_raw = xai_debugging.run_xai_debugging(
        model=model,
        fault_type=FAULT_TYPE,
        fault_metadata=fault_metadata,
        injected_splits=injected_splits,
        config=xai_config,
    )
    xai_result = enrich_debugging_result(
        xai_result_raw,
        FAULT_TYPE,
        fault_metadata,
    )

    oracle_result = run_oracle_repair(
        model=model,
        fault_type=FAULT_TYPE,
        fault_metadata=fault_metadata,
        injected_splits=injected_splits,
    )

    # Add oracle-normalized repair interpretation after oracle is available.
    baseline_result = add_oracle_context_to_workflow_result(
        baseline_result,
        oracle_result,
    )
    xai_result = add_oracle_context_to_workflow_result(
        xai_result,
        oracle_result,
    )

    method_comparison_summary = build_method_comparison_summary(
        baseline_result=baseline_result,
        xai_result=xai_result,
        oracle_result=oracle_result,
    )

    _print_debugging_result("BASELINE DEBUGGING (NO XAI)", baseline_result, FAULT_TYPE)
    _print_debugging_result("XAI DEBUGGING (SHAP)", xai_result, FAULT_TYPE)
    _print_oracle_result(oracle_result)
    _print_method_comparison_summary(method_comparison_summary)

    output_rows = build_output_rows(
        seed=RANDOM_STATE,
        fault_type=FAULT_TYPE,
        fault_mode=_current_mode_label(),
        baseline_result=baseline_result,
        xai_result=xai_result,
        oracle_result=oracle_result,
    )

    append_rows_to_csv(output_rows, output_csv)

    full_result = {
        "seed": RANDOM_STATE,
        "fault_type": FAULT_TYPE,
        "fault_mode": _current_mode_label(),
        "fault_metadata": fault_metadata,
        "initial_metrics": {
            "train": train_metrics,
            "contaminated_eval": contaminated_eval_metrics,
            "clean_holdout": clean_holdout_metrics,
        },
        "baseline_debugging_result": baseline_result,
        "xai_debugging_result": xai_result,
        "oracle_repair_result": oracle_result,
        "method_comparison_summary": method_comparison_summary,
        "csv_rows": output_rows,
    }

    append_result_to_jsonl(full_result, output_jsonl)

    return full_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fault-injection experiment")
    parser.add_argument(
        "--fault",
        default=FAULT_TYPE,
        help="Fault type: none,label_noise,data_leakage,spurious_correlation",
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=(0, 1),
        default=0,
        help="Mode selector: 0 or 1 mapped per fault",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for split, injection, and model training",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to append flat experiment rows, e.g. results/experiments.csv",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional path to append full nested experiment result, e.g. results/experiments.jsonl",
    )

    args = parser.parse_args()

    fault = args.fault
    mode = int(args.mode)

    if fault == "label_noise":
        LABEL_NOISE_MODE = "random" if mode == 0 else "hard"
    elif fault == "data_leakage":
        LEAKAGE_MODE = "direct" if mode == 0 else "indirect"
    elif fault == "spurious_correlation":
        SPURIOUS_MODE = "broken" if mode == 0 else "inverted"

    FAULT_TYPE = fault
    RANDOM_STATE = int(args.seed)

    print(f"Running with FAULT_TYPE={FAULT_TYPE}, mode={mode}, seed={RANDOM_STATE}")

    run_single_experiment(
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
    )