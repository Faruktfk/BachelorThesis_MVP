"""Main orchestration for fault-injection ML pipeline testing.

This module provides experiment setup, training, evaluation, and diagnostics.
Fault-injection logic is delegated to faults.py module.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")


import argparse
import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import faults
import baseline_debugging


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
INDIRECT_LEAKAGE_BINS = 8
INDIRECT_LEAKAGE_CONTAM_SMOOTHING = 0.75
INDIRECT_LEAKAGE_CLEAN_SMOOTHING = 18.0
INDIRECT_LEAKAGE_HOLDOUT_SHRINK = 0.95
INDIRECT_LEAKAGE_NOISE_STD = 0.06

# Spurious correlation config
SPURIOUS_MODE = "broken"  # options: "broken", "inverted"
SPURIOUS_STRENGTH = 0.90
USE_GROUPS_FOR_SPURIOUS = True
INVERTED_GROUP_WEIGHT = 0.35
INVERTED_SIGNAL_WEIGHT = 0.20
INVERTED_NOISE_STD = 0.42

# Sanity checks
ENABLE_FAULT_SANITY_CHECKS = True


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
    """Train a RandomForestClassifier on the injected training split."""
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(train_features, train_labels)
    return model


def evaluate_model(model: RandomForestClassifier, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
    """Compute accuracy, F1, and ROC-AUC for a fitted classifier."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _print_metric_block(title: str, metrics: Dict[str, float]) -> None:
    print(title)
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  f1: {metrics['f1']:.4f}")
    print(f"  roc_auc: {metrics['roc_auc']:.4f}")


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

    if contaminated_corr < 0.30 or eval_gain < 0.020:
        return "too_weak", "Indirect leakage signal is too weak to separate offline and clean views."
    if contaminated_corr > 0.92 or eval_gain > 0.14:
        return "too_strong", "Indirect leakage is too dominant; leaves little room for debugging insights."
    if corr_drop < 0.15:
        return "too_weak", "Indirect leakage behaves too similarly on contaminated_eval and clean_holdout."
    return "usable", "Indirect leakage creates a clear offline/clean gap without being trivial."


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

    if train_corr < 0.35 or contaminated_corr < 0.35:
        return "too_weak", "Shortcut feature is not attractive enough in the training view."

    if spurious_mode == "broken":
        if holdout_corr > 0.15:
            return "too_weak", "Holdout shortcut is not truly broken; residual correlation remains."
        if holdout_gap < 0.08:
            return "too_weak", "Distribution shift from shortcut breaking is too subtle."
        return "usable", "Shortcut breaks cleanly in holdout; domain shift is realistic."

    if spurious_mode == "inverted":
        if holdout_negative_corr > -0.15:
            return "too_weak", "Inversion in holdout is too weak."
        if holdout_gap > 0.30 or clean_holdout_metrics["accuracy"] < 0.68:
            return "too_strong", "Inversion effect is still too brutal; holdout fails too badly."
        return "usable", "Inversion is clear but not catastrophic; realistic debugging scenario."

    return "usable", "Spurious correlation shows realistic train/holdout shift."


def assess_fault_quality(
    fault_type: str,
    train_metrics: Dict[str, float],
    contaminated_eval_metrics: Dict[str, float],
    clean_holdout_metrics: Dict[str, float],
    fault_metadata: Dict[str, Any],
    diagnostics: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    if fault_type == "label_noise":
        quality, reason = _quality_label_noise(train_metrics, contaminated_eval_metrics, clean_holdout_metrics, fault_metadata)
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


def apply_fault_injection(
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
) -> Tuple[Dict[str, Dict[str, pd.DataFrame | pd.Series]], Dict[str, Any]]:
    """Apply fault injection using faults module with config dictionary."""
    # Build config dict from global variables for faults module
    config = {
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
        "SPURIOUS_MODE": SPURIOUS_MODE,
        "SPURIOUS_STRENGTH": SPURIOUS_STRENGTH,
        "USE_GROUPS_FOR_SPURIOUS": USE_GROUPS_FOR_SPURIOUS,
        "INVERTED_GROUP_WEIGHT": INVERTED_GROUP_WEIGHT,
        "INVERTED_SIGNAL_WEIGHT": INVERTED_SIGNAL_WEIGHT,
        "INVERTED_NOISE_STD": INVERTED_NOISE_STD,
    }

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


def main() -> Dict[str, Any]:
    """Run the full reproducible training and fault-injection workflow with baseline debugging."""
    features, labels = load_dataset()
    splits = split_dataset(features, labels)
    injected_splits, fault_metadata = apply_fault_injection(splits)

    model = train_model(injected_splits["train"]["features"], injected_splits["train"]["labels"])

    train_metrics = evaluate_model(model, injected_splits["train"]["features"], injected_splits["train"]["labels"])
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

    print(f"\n{'*'*70}")
    print("FAULT INJECTION & INITIAL TRAINING")
    print(f"{'*'*70}")
    print(f"Active fault type: {FAULT_TYPE}")
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

    # Build config dict for baseline debugging
    baseline_config = {
        "RANDOM_STATE": RANDOM_STATE,
        "N_ESTIMATORS": N_ESTIMATORS,
    }

    # Run baseline debugging
    print(f"\n{'*'*70}")
    print("BASELINE DEBUGGING (NO XAI)")
    print(f"{'*'*70}")
    baseline_result = baseline_debugging.run_baseline_debugging(
        model=model,
        fault_type=FAULT_TYPE,
        fault_metadata=fault_metadata,
        injected_splits=injected_splits,
        config=baseline_config,
    )

    # Format baseline results
    print(f"\nDetection steps: {baseline_result['steps_to_detect']}")
    print(f"Retrains: {baseline_result['retrains']}")
    print(f"Runtime: {baseline_result['runtime_sec']:.3f} seconds")

    if FAULT_TYPE == "label_noise":
        print(f"\nLabel Noise Detection:")
        print(f"  Precision@k: {baseline_result.get('precision_at_k', 0.0):.4f}")
        print(f"  Recall@k: {baseline_result.get('recall_at_k', 0.0):.4f}")
        print(f"  Top suspect indices: {baseline_result.get('suspect_indices', [])[:5]}")

    elif FAULT_TYPE == "data_leakage":
        print(f"\nData Leakage Detection:")
        print(f"  Rank of true feature: {baseline_result.get('rank_true_feature', -1)}")
        print(f"  Top candidate feature: {baseline_result.get('top_candidate_feature')}")
        print(f"  Top-5 suspect features: {baseline_result.get('suspect_features', [])[:5]}")

    elif FAULT_TYPE == "spurious_correlation":
        print(f"\nSpurious Correlation Detection:")
        print(f"  Rank of true feature: {baseline_result.get('rank_true_feature', -1)}")
        print(f"  Top candidate feature: {baseline_result.get('top_candidate_feature')}")
        print(f"  Top-5 suspect features: {baseline_result.get('suspect_features', [])[:5]}")

    print(f"\nMetrics before fix:")
    for split_name in ["train", "contaminated_eval", "clean_holdout"]:
        acc = baseline_result["metrics_before"].get(f"{split_name}_accuracy", 0.0)
        f1 = baseline_result["metrics_before"].get(f"{split_name}_f1", 0.0)
        auc = baseline_result["metrics_before"].get(f"{split_name}_roc_auc", 0.0)
        print(f"  {split_name}: accuracy={acc:.4f}, f1={f1:.4f}, roc_auc={auc:.4f}")

    print(f"\nMetrics after fix:")
    for split_name in ["train", "contaminated_eval", "clean_holdout"]:
        acc = baseline_result["metrics_after"].get(f"{split_name}_accuracy", 0.0)
        f1 = baseline_result["metrics_after"].get(f"{split_name}_f1", 0.0)
        auc = baseline_result["metrics_after"].get(f"{split_name}_roc_auc", 0.0)
        print(f"  {split_name}: accuracy={acc:.4f}, f1={f1:.4f}, roc_auc={auc:.4f}")

    print(f"\nFix impact (delta):")
    for metric_name, delta in baseline_result["fix_impact"].items():
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  {metric_name}: {delta:+.4f} {direction}")

    return {
        "fault_type": FAULT_TYPE,
        "fault_metadata": fault_metadata,
        "initial_metrics": {
            "train": train_metrics,
            "contaminated_eval": contaminated_eval_metrics,
            "clean_holdout": clean_holdout_metrics,
        },
        "baseline_debugging_result": baseline_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fault-injection experiment")
    parser.add_argument("--fault", default=FAULT_TYPE, help="Fault type: none,label_noise,data_leakage,spurious_correlation")
    parser.add_argument("--mode", type=int, choices=(0, 1), default=0, help="Mode selector: 0 or 1 (mapped per fault)")
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
    print(f"Running with FAULT_TYPE={FAULT_TYPE}, mode={mode}")
    main()
