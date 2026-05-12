"""Baseline debugging logic without XAI - classical signal-based detection.

This module implements deterministic baseline debugging using only classical signals:
- Prediction-label discrepancies
- Prediction uncertainty (confidence margins)
- Feature correlations and importance
- Split-wise behavior comparison (contaminated_eval vs clean_holdout)

No explainability methods (SHAP, LIME, etc.) are used here.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _compute_prediction_margins(
    model: RandomForestClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """Compute prediction confidence margins (max_prob - second_max_prob)."""
    probas = model.predict_proba(features)
    top_two = np.sort(probas, axis=1)[:, -2:]
    margins = top_two[:, 1] - top_two[:, 0]
    return margins


def _safe_correlation(series: pd.Series, labels: pd.Series) -> float:
    """Safely compute Pearson correlation, returning 0 if undefined."""
    if series.nunique(dropna=True) < 2 or labels.nunique(dropna=True) < 2:
        return 0.0
    value = series.corr(labels)
    return 0.0 if pd.isna(value) else float(value)


def _detect_label_noise(
    model: RandomForestClassifier,
    train_split: Dict[str, pd.DataFrame | pd.Series],
    fault_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect label noise using classical signals.
    
    Strategy:
    1. Compute model predictions on training data
    2. Identify prediction-label discrepancies
    3. Weight discrepancies by prediction margin (uncertainty)
    4. Rank training samples by suspicion score
    5. Compare with actual noisy labels to compute precision/recall@k
    
    Returns dict with suspect_indices, scores, precision_at_k, recall_at_k.
    """
    train_features = train_split["features"]
    train_labels = train_split["labels"]

    # Step 1: Get predictions and margins
    predictions = model.predict(train_features).astype(int)
    margins = _compute_prediction_margins(model, train_features)

    # Step 2: Compute discrepancy score
    # Wrong predictions weighted by uncertainty are more suspicious
    is_wrong = (predictions != train_labels.values.astype(int)).astype(float)
    discrepancy_score = is_wrong * (1.0 - margins)

    # Step 3: Rank samples by suspicion (higher = more suspicious)
    suspect_indices = np.argsort(-discrepancy_score)
    suspect_scores = discrepancy_score[suspect_indices]

    # Step 4: Evaluate against ground truth
    if "changed_indices" in fault_metadata:
        actual_noisy_indices = set(fault_metadata["changed_indices"])
        # Select top-k suspects (k = number of actual noisy labels)
        k = len(actual_noisy_indices)
        suspect_set = set(suspect_indices[:k])

        # Precision @ k: fraction of top-k that are actually noisy
        true_positives = len(suspect_set & actual_noisy_indices)
        precision_at_k = float(true_positives) / float(k) if k > 0 else 0.0
        # Recall @ k: fraction of actual noisy labels in top-k
        recall_at_k = float(true_positives) / float(len(actual_noisy_indices)) if len(actual_noisy_indices) > 0 else 0.0
    else:
        precision_at_k = 0.0
        recall_at_k = 0.0

    return {
        "suspect_indices": suspect_indices.tolist(),
        "suspect_scores": suspect_scores.tolist(),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
    }


def _detect_data_leakage(
    model: RandomForestClassifier,
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    fault_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect data leakage using classical signals.
    
    Strategy:
    1. Compute feature importances from trained model
    2. Compute feature correlations in contaminated_eval and clean_holdout
    3. Identify features with high instability (correlated in contaminated, weak in clean)
    4. Rank features by combined importance + instability score
    5. Find rank of true leakage feature
    
    Returns dict with suspect_features, suspicion_scores, rank_true_feature, top_candidate.
    """
    contaminated_eval_features = splits["contaminated_eval"]["features"]
    contaminated_eval_labels = splits["contaminated_eval"]["labels"]
    clean_holdout_features = splits["clean_holdout"]["features"]
    clean_holdout_labels = splits["clean_holdout"]["labels"]

    # Step 1: Extract feature importances
    feature_importance = pd.Series(model.feature_importances_, index=contaminated_eval_features.columns)

    # Step 2: Compute correlations in each view
    contaminated_corr = contaminated_eval_features.apply(
        lambda col: abs(_safe_correlation(col, contaminated_eval_labels))
    )
    clean_corr = clean_holdout_features.apply(
        lambda col: abs(_safe_correlation(col, clean_holdout_labels))
    )

    # Step 3: Compute instability metric
    # High correlation in contaminated but weak in clean = suspicious
    instability = contaminated_corr - clean_corr

    # Step 4: Combine signals
    # Prioritize instability (distribution shift) over importance
    suspicion = 0.65 * instability + 0.35 * feature_importance

    # Step 5: Sort features by suspicion
    suspect_features = suspicion.sort_values(ascending=False).index.tolist()

    # Step 6: Find rank of true leakage feature
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
        "suspicion_scores": suspicion.to_dict(),
        "rank_true_feature": rank_true_feature,
        "top_candidate_feature": suspect_features[0] if suspect_features else None,
    }


def _detect_spurious_correlation(
    model: RandomForestClassifier,
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    fault_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect spurious correlation using classical signals.
    
    Strategy:
    1. Compute feature importances from model (trained on contaminated data)
    2. Compute feature correlations in train, contaminated_eval, and clean_holdout views
    3. Identify features that are important in train/contaminated but unstable in clean_holdout
    4. Rank by instability metric: (max_train_corr - clean_corr) * importance
    5. Find rank of true spurious feature
    
    Returns dict with suspect_features, suspicion_scores, rank_true_feature, top_candidate.
    """
    train_features = splits["train"]["features"]
    train_labels = splits["train"]["labels"]
    contaminated_eval_features = splits["contaminated_eval"]["features"]
    contaminated_eval_labels = splits["contaminated_eval"]["labels"]
    clean_holdout_features = splits["clean_holdout"]["features"]
    clean_holdout_labels = splits["clean_holdout"]["labels"]

    # Step 1: Extract feature importances
    feature_importance = pd.Series(model.feature_importances_, index=train_features.columns)

    # Step 2: Compute correlations in each view
    train_corr = train_features.apply(lambda col: abs(_safe_correlation(col, train_labels)))
    contaminated_corr = contaminated_eval_features.apply(
        lambda col: abs(_safe_correlation(col, contaminated_eval_labels))
    )
    clean_corr = clean_holdout_features.apply(
        lambda col: abs(_safe_correlation(col, clean_holdout_labels))
    )

    # Step 3: Compute instability metric
    # Features that are correlated in train/contaminated but weak in clean are suspicious
    max_offline_corr = pd.concat([train_corr, contaminated_corr], axis=1).max(axis=1)
    instability = max_offline_corr - clean_corr

    # Step 4: Combine signals
    # Prioritize instability (domain shift indicator) over raw importance
    suspicion = 0.70 * instability + 0.30 * feature_importance

    # Step 5: Sort features by suspicion
    suspect_features = suspicion.sort_values(ascending=False).index.tolist()

    # Step 6: Find rank of true spurious feature
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
        "suspicion_scores": suspicion.to_dict(),
        "rank_true_feature": rank_true_feature,
        "top_candidate_feature": suspect_features[0] if suspect_features else None,
    }


def _apply_label_noise_fix(
    train_split: Dict[str, pd.DataFrame | pd.Series],
    fault_metadata: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fix label noise by correcting suspected samples using original labels.
    
    Uses all samples marked as changed in fault_metadata and reverts them
    to their original labels.
    """
    corrected_labels = train_split["labels"].copy()

    if "original_labels_by_index" in fault_metadata:
        original_labels_dict = fault_metadata["original_labels_by_index"]
        for str_idx, original_label in original_labels_dict.items():
            idx = int(str_idx)
            # Use .loc to access by index label (not position)
            if idx in corrected_labels.index:
                corrected_labels.loc[idx] = original_label

    return train_split["features"], corrected_labels


def _apply_feature_removal_fix(
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    feature_to_remove: str,
) -> Dict[str, Dict[str, pd.DataFrame | pd.Series]]:
    """Remove a suspected feature from all splits."""
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
    metrics = {}
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
    """
    Run baseline debugging workflow.
    
    Steps:
    1. Compute metrics before fix
    2. Perform fault-specific detection (label noise / leakage / spurious)
    3. Apply fix (correct labels / remove feature)
    4. Retrain model on fixed data
    5. Compute metrics after fix
    6. Return standardized results
    
    Args:
        model: Trained RandomForestClassifier
        fault_type: "label_noise", "data_leakage", "spurious_correlation", or "none"
        fault_metadata: Metadata dict from fault injection
        injected_splits: Dict of train/contaminated_eval/clean_holdout splits with injected faults
        config: Optional config dict with RANDOM_STATE, N_ESTIMATORS, etc.
    
    Returns:
        Dict with metrics_before, metrics_after, detection results, fix_impact,
        steps_to_detect, retrains, runtime_sec, and fault-specific fields.
    """
    if config is None:
        config = {}

    start_time = time.time()

    # Step 1: Metrics before fix
    metrics_before = _compute_split_metrics(model, injected_splits)

    # Step 2: No fault case
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

    # Step 2: Fault-specific detection
    detection_result = {}
    steps_to_detect = 0

    if fault_type == "label_noise":
        # 4 steps: feature/prediction loading, margin computation, discrepancy scoring, ranking
        detection_result = _detect_label_noise(model, injected_splits["train"], fault_metadata, config)
        steps_to_detect = 4

        # Step 3: Apply fix (correct labels)
        fixed_features, fixed_labels = _apply_label_noise_fix(injected_splits["train"], fault_metadata)

        # Step 4: Retrain model
        retrained_model = RandomForestClassifier(
            n_estimators=config.get("N_ESTIMATORS", 200),
            random_state=config.get("RANDOM_STATE", 42),
        )
        retrained_model.fit(fixed_features, fixed_labels)

        # Step 5: Compute metrics after (on original splits with corrected labels)
        eval_splits = {
            "train": {"features": fixed_features, "labels": fixed_labels},
            "contaminated_eval": injected_splits["contaminated_eval"],
            "clean_holdout": injected_splits["clean_holdout"],
        }
        metrics_after = _compute_split_metrics(retrained_model, eval_splits)

    elif fault_type == "data_leakage":
        # 4 steps: feature correlation, importance, instability analysis, ranking
        detection_result = _detect_data_leakage(model, injected_splits, fault_metadata, config)
        steps_to_detect = 4

        # Step 3: Apply fix (remove leakage feature)
        suspect_feature = detection_result["top_candidate_feature"]
        if suspect_feature:
            fixed_splits = _apply_feature_removal_fix(injected_splits, suspect_feature)
        else:
            fixed_splits = injected_splits

        # Step 4: Retrain model
        retrained_model = RandomForestClassifier(
            n_estimators=config.get("N_ESTIMATORS", 200),
            random_state=config.get("RANDOM_STATE", 42),
        )
        retrained_model.fit(fixed_splits["train"]["features"], fixed_splits["train"]["labels"])

        # Step 5: Compute metrics after
        metrics_after = _compute_split_metrics(retrained_model, fixed_splits)

    elif fault_type == "spurious_correlation":
        # 4 steps: feature importance, correlations, instability detection, ranking
        detection_result = _detect_spurious_correlation(model, injected_splits, fault_metadata, config)
        steps_to_detect = 4

        # Step 3: Apply fix (remove spurious feature)
        suspect_feature = detection_result["top_candidate_feature"]
        if suspect_feature:
            fixed_splits = _apply_feature_removal_fix(injected_splits, suspect_feature)
        else:
            fixed_splits = injected_splits

        # Step 4: Retrain model
        retrained_model = RandomForestClassifier(
            n_estimators=config.get("N_ESTIMATORS", 200),
            random_state=config.get("RANDOM_STATE", 42),
        )
        retrained_model.fit(fixed_splits["train"]["features"], fixed_splits["train"]["labels"])

        # Step 5: Compute metrics after
        metrics_after = _compute_split_metrics(retrained_model, fixed_splits)

    else:
        # Unknown fault type
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

    # Step 6: Compute fix impact
    fix_impact = {
        key: metrics_after.get(key, 0.0) - metrics_before.get(key, 0.0)
        for key in metrics_before.keys()
    }

    # Build standardized result
    result = {
        "workflow": "baseline",
        "fault_type": fault_type,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "fix_impact": fix_impact,
        "steps_to_detect": steps_to_detect,
        "retrains": 1,  # One retrain after fix
        "runtime_sec": float(time.time() - start_time),
    }

    # Add fault-specific detection results
    result.update(detection_result)

    return result
