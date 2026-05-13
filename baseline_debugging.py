"""Baseline debugging logic without XAI - classical signal-based detection.

This module implements deterministic baseline debugging using only classical signals:
- Prediction-label discrepancies
- Prediction uncertainty (confidence margins)
- Simple feature-label statistics on train and contaminated_eval
- No clean_holdout access during detection
- No feature_importances_, SHAP, LIME, or other attribution methods

Important methodological rule:
- clean_holdout is NOT used to generate suspect rankings
- clean_holdout is only used later for objective evaluation of fix success
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _compute_prediction_margins(
    model: RandomForestClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """Compute prediction confidence margins (max_prob - second_max_prob)."""
    probas = model.predict_proba(features)
    top_two = np.sort(probas, axis=1)[:, -2:]
    margins = top_two[:, 1] - top_two[:, 0]
    return margins


def _compute_margins_from_proba(probas: np.ndarray) -> np.ndarray:
    """Compute binary-class confidence margins from class-1 probabilities."""
    return np.abs(2.0 * probas - 1.0)


def _safe_correlation(series: pd.Series, labels: pd.Series) -> float:
    """Safely compute Pearson correlation, returning 0 if undefined."""
    if series.nunique(dropna=True) < 2 or labels.nunique(dropna=True) < 2:
        return 0.0
    value = series.corr(labels)
    return 0.0 if pd.isna(value) else float(value)


def _feature_proxy(feature: pd.Series, labels: pd.Series) -> Dict[str, float]:
    """Compute simple classical statistics for one feature.

    Returns:
    - correlation
    - mutual_information
    - mean_gap between positive and negative class
    """
    if feature.nunique(dropna=True) < 2:
        return {"correlation": 0.0, "mutual_information": 0.0, "mean_gap": 0.0}

    ranked = feature.rank(method="first")
    bin_count = min(10, max(2, int(ranked.nunique())))
    try:
        discretized = pd.qcut(ranked, q=bin_count, duplicates="drop", labels=False)
    except ValueError:
        discretized = pd.cut(ranked, bins=bin_count, labels=False, include_lowest=True)

    discrete = pd.Series(discretized, index=feature.index).fillna(-1).astype(int)
    positive = feature[labels == 1]
    negative = feature[labels == 0]
    mean_gap = abs(float(positive.mean() - negative.mean())) if len(positive) and len(negative) else 0.0

    return {
        "correlation": _safe_correlation(feature, labels),
        "mutual_information": float(mutual_info_score(labels.astype(int), discrete)),
        "mean_gap": mean_gap,
    }


def _normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)


def _summarize_feature_table(
    features: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    """Build a per-feature statistics table."""
    rows: Dict[str, Dict[str, float]] = {}
    for feature_name in features.columns:
        rows[feature_name] = _feature_proxy(features[feature_name], labels)
    return pd.DataFrame.from_dict(rows, orient="index")


def _compute_oof_probabilities(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    config: Dict[str, Any],
) -> np.ndarray:
    """Compute out-of-fold probabilities for robust label-noise detection."""
    cv_folds = int(config.get("LABEL_NOISE_CV_FOLDS", 5))
    cv_folds = max(2, cv_folds)

    min_class_count = int(train_labels.value_counts().min())
    cv_folds = min(cv_folds, min_class_count)
    cv_folds = max(2, cv_folds)

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=int(config.get("RANDOM_STATE", 42)),
    )

    probe_model = RandomForestClassifier(
        n_estimators=int(config.get("N_ESTIMATORS", 200)),
        random_state=int(config.get("RANDOM_STATE", 42)),
    )

    oof_proba = cross_val_predict(
        probe_model,
        train_features,
        train_labels,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]

    return oof_proba


def _detect_label_noise(
    train_split: Dict[str, pd.DataFrame | pd.Series],
    fault_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Detect label noise using only classical sample-level signals.

    Uses:
    - out-of-fold probability for observed label
    - prediction/label disagreement
    - low confidence margin / high uncertainty
    """
    train_features = train_split["features"]
    train_labels = train_split["labels"]

    oof_proba_pos = _compute_oof_probabilities(train_features, train_labels, config)
    observed_labels = train_labels.to_numpy().astype(int)
    predicted_labels = (oof_proba_pos >= 0.5).astype(int)

    prob_observed_label = np.where(observed_labels == 1, oof_proba_pos, 1.0 - oof_proba_pos)
    disagreement = (predicted_labels != observed_labels).astype(float)
    margins = _compute_margins_from_proba(oof_proba_pos)
    uncertainty = 1.0 - margins

    suspicion_score = (
        0.60 * (1.0 - prob_observed_label)
        + 0.25 * disagreement
        + 0.15 * uncertainty
    )

    rank_positions = np.argsort(-suspicion_score)
    ranked_index_labels = train_features.index.to_numpy()[rank_positions]
    ranked_scores = suspicion_score[rank_positions]

    suspect_indices = [int(idx) for idx in ranked_index_labels.tolist()]
    suspect_scores = ranked_scores.tolist()

    if "changed_indices" in fault_metadata:
        actual_noisy_indices = set(int(idx) for idx in fault_metadata["changed_indices"])
        k = len(actual_noisy_indices)

        top_k_suspects = suspect_indices[:k]
        suspect_set = set(top_k_suspects)

        true_positives = len(suspect_set & actual_noisy_indices)
        precision_at_k = float(true_positives) / float(k) if k > 0 else 0.0
        recall_at_k = float(true_positives) / float(len(actual_noisy_indices)) if len(actual_noisy_indices) > 0 else 0.0
    else:
        k = 0
        precision_at_k = 0.0
        recall_at_k = 0.0

    return {
        "suspect_indices": suspect_indices,
        "suspect_scores": suspect_scores,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "top_k_used_for_eval": k,
    }


def _rank_feature_candidates_from_offline_statistics(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    contaminated_eval_features: pd.DataFrame,
    contaminated_eval_labels: pd.Series,
) -> Tuple[List[str], Dict[str, float]]:
    """Rank suspicious features using only train + contaminated_eval statistics.

    This intentionally avoids:
    - clean_holdout
    - feature_importances_
    - any XAI / attribution signal

    Rationale:
    A realistic developer only sees the training data and the (possibly contaminated)
    offline validation view. They can inspect simple correlations, mutual information,
    and class-separation heuristics, but they do not have a perfectly clean reference set.
    """
    train_stats = _summarize_feature_table(train_features, train_labels)
    eval_stats = _summarize_feature_table(contaminated_eval_features, contaminated_eval_labels)

    train_corr = _normalize_series(train_stats["correlation"].abs())
    eval_corr = _normalize_series(eval_stats["correlation"].abs())
    train_mi = _normalize_series(train_stats["mutual_information"])
    eval_mi = _normalize_series(eval_stats["mutual_information"])
    mean_gap = _normalize_series((train_stats["mean_gap"] + eval_stats["mean_gap"]) / 2.0)

    consistency = pd.concat([train_corr, eval_corr], axis=1).min(axis=1)

    suspicion = (
        0.30 * train_corr
        + 0.25 * eval_corr
        + 0.20 * consistency
        + 0.15 * train_mi
        + 0.05 * eval_mi
        + 0.05 * mean_gap
    )

    suspect_features = suspicion.sort_values(ascending=False).index.tolist()
    suspicion_scores = {feature: float(score) for feature, score in suspicion.items()}
    return suspect_features, suspicion_scores


def _detect_data_leakage(
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    fault_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Detect data leakage using only train + contaminated_eval statistics."""
    train_features = splits["train"]["features"]
    train_labels = splits["train"]["labels"]
    contaminated_eval_features = splits["contaminated_eval"]["features"]
    contaminated_eval_labels = splits["contaminated_eval"]["labels"]

    suspect_features, suspicion_scores = _rank_feature_candidates_from_offline_statistics(
        train_features,
        train_labels,
        contaminated_eval_features,
        contaminated_eval_labels,
    )

    if "leakage_feature_name" in fault_metadata:
        true_feature = fault_metadata["leakage_feature_name"]
        try:
            rank_true_feature = suspect_features.index(true_feature) + 1
        except ValueError:
            rank_true_feature = len(suspect_features) + 1
    else:
        rank_true_feature = -1

    return {
        "suspect_features": suspect_features,
        "suspicion_scores": suspicion_scores,
        "rank_true_feature": rank_true_feature,
        "top_candidate_feature": suspect_features[0] if suspect_features else None,
    }


def _detect_spurious_correlation(
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    fault_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Detect spurious correlation using only train + contaminated_eval statistics.

    Baseline rule:
    - features that look globally very predictive across the offline world
      are suspicious shortcut candidates
    - no clean_holdout is used during this ranking
    """
    train_features = splits["train"]["features"]
    train_labels = splits["train"]["labels"]
    contaminated_eval_features = splits["contaminated_eval"]["features"]
    contaminated_eval_labels = splits["contaminated_eval"]["labels"]

    suspect_features, suspicion_scores = _rank_feature_candidates_from_offline_statistics(
        train_features,
        train_labels,
        contaminated_eval_features,
        contaminated_eval_labels,
    )

    if "feature_name" in fault_metadata:
        true_feature = fault_metadata["feature_name"]
        try:
            rank_true_feature = suspect_features.index(true_feature) + 1
        except ValueError:
            rank_true_feature = len(suspect_features) + 1
    else:
        rank_true_feature = -1

    return {
        "suspect_features": suspect_features,
        "suspicion_scores": suspicion_scores,
        "rank_true_feature": rank_true_feature,
        "top_candidate_feature": suspect_features[0] if suspect_features else None,
    }


def _apply_label_noise_fix(
    train_split: Dict[str, pd.DataFrame | pd.Series],
    fault_metadata: Dict[str, Any],
    suspect_indices: List[int],
    max_fixes: int,
) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
    """Fix only the top-k suspected noisy samples using stored original labels."""
    corrected_labels = train_split["labels"].copy()
    corrected_indices: List[int] = []

    if "original_labels_by_index" not in fault_metadata:
        return train_split["features"], corrected_labels, corrected_indices

    original_labels_dict = {
        int(idx): int(label)
        for idx, label in fault_metadata["original_labels_by_index"].items()
    }

    for idx in suspect_indices[:max_fixes]:
        if idx in original_labels_dict and idx in corrected_labels.index:
            corrected_labels.loc[idx] = original_labels_dict[idx]
            corrected_indices.append(int(idx))

    return train_split["features"], corrected_labels, corrected_indices


def _apply_feature_removal_fix(
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    feature_to_remove: str,
) -> Dict[str, Dict[str, pd.DataFrame | pd.Series]]:
    """Remove one suspected feature from all splits."""
    fixed_splits = {}
    for split_name, split_data in splits.items():
        if feature_to_remove in split_data["features"].columns:
            fixed_features = split_data["features"].drop(columns=[feature_to_remove])
        else:
            fixed_features = split_data["features"]
        fixed_splits[split_name] = {
            "features": fixed_features,
            "labels": split_data["labels"],
        }
    return fixed_splits


def _compute_split_metrics(
    model: RandomForestClassifier,
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
) -> Dict[str, float]:
    """Compute accuracy, F1, and ROC-AUC for all splits."""
    metrics: Dict[str, float] = {}
    for split_name in ["train", "contaminated_eval", "clean_holdout"]:
        split = splits[split_name]
        features = split["features"]
        labels = split["labels"]

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]

        metrics[f"{split_name}_accuracy"] = float(accuracy_score(labels, predictions))
        metrics[f"{split_name}_f1"] = float(f1_score(labels, predictions))
        metrics[f"{split_name}_roc_auc"] = float(roc_auc_score(labels, probabilities))

    return metrics


def run_baseline_debugging(
    model: RandomForestClassifier,
    fault_type: str,
    fault_metadata: Dict[str, Any],
    injected_splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the baseline debugging workflow.

    Important:
    - clean_holdout is used ONLY for evaluation after the ranking/fix decision is made
    - detection itself uses only train + contaminated_eval
    """
    if config is None:
        config = {}

    start_time = time.time()
    metrics_before = _compute_split_metrics(model, injected_splits)

    if fault_type == "none":
        metrics_after = metrics_before
        fix_impact = {key: 0.0 for key in metrics_before.keys()}
        return {
            "workflow": "baseline",
            "fault_type": fault_type,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "fix_impact": fix_impact,
            "steps_to_detect": 0,
            "retrains": 0,
            "runtime_sec": float(time.time() - start_time),
        }

    detection_result: Dict[str, Any] = {}
    steps_to_detect = 0

    if fault_type == "label_noise":
        detection_result = _detect_label_noise(injected_splits["train"], fault_metadata, config)
        steps_to_detect = 5

        k = int(detection_result.get("top_k_used_for_eval", 0))
        fixed_features, fixed_labels, corrected_indices = _apply_label_noise_fix(
            injected_splits["train"],
            fault_metadata,
            detection_result.get("suspect_indices", []),
            k,
        )

        retrained_model = RandomForestClassifier(
            n_estimators=int(config.get("N_ESTIMATORS", 200)),
            random_state=int(config.get("RANDOM_STATE", 42)),
        )
        retrained_model.fit(fixed_features, fixed_labels)

        eval_splits = {
            "train": {"features": fixed_features, "labels": fixed_labels},
            "contaminated_eval": injected_splits["contaminated_eval"],
            "clean_holdout": injected_splits["clean_holdout"],
        }
        metrics_after = _compute_split_metrics(retrained_model, eval_splits)

        detection_result["applied_fix_count"] = len(corrected_indices)
        detection_result["corrected_indices"] = corrected_indices

    elif fault_type == "data_leakage":
        detection_result = _detect_data_leakage(injected_splits, fault_metadata)
        steps_to_detect = 4

        suspect_feature = detection_result["top_candidate_feature"]
        fixed_splits = _apply_feature_removal_fix(injected_splits, suspect_feature) if suspect_feature else injected_splits

        retrained_model = RandomForestClassifier(
            n_estimators=int(config.get("N_ESTIMATORS", 200)),
            random_state=int(config.get("RANDOM_STATE", 42)),
        )
        retrained_model.fit(fixed_splits["train"]["features"], fixed_splits["train"]["labels"])
        metrics_after = _compute_split_metrics(retrained_model, fixed_splits)

    elif fault_type == "spurious_correlation":
        detection_result = _detect_spurious_correlation(injected_splits, fault_metadata)
        steps_to_detect = 4

        suspect_feature = detection_result["top_candidate_feature"]
        fixed_splits = _apply_feature_removal_fix(injected_splits, suspect_feature) if suspect_feature else injected_splits

        retrained_model = RandomForestClassifier(
            n_estimators=int(config.get("N_ESTIMATORS", 200)),
            random_state=int(config.get("RANDOM_STATE", 42)),
        )
        retrained_model.fit(fixed_splits["train"]["features"], fixed_splits["train"]["labels"])
        metrics_after = _compute_split_metrics(retrained_model, fixed_splits)

    else:
        metrics_after = metrics_before
        fix_impact = {key: 0.0 for key in metrics_before.keys()}
        return {
            "workflow": "baseline",
            "fault_type": fault_type,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "fix_impact": fix_impact,
            "steps_to_detect": 0,
            "retrains": 0,
            "runtime_sec": float(time.time() - start_time),
        }

    fix_impact = {
        key: metrics_after.get(key, 0.0) - metrics_before.get(key, 0.0)
        for key in metrics_before.keys()
    }

    result = {
        "workflow": "baseline",
        "fault_type": fault_type,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "fix_impact": fix_impact,
        "steps_to_detect": steps_to_detect,
        "retrains": 1,
        "runtime_sec": float(time.time() - start_time),
    }
    result.update(detection_result)
    return result