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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
    # For binary classification, the second probability is 1-p.
    # Margin = |p - (1-p)| = |2p - 1|
    return np.abs(2.0 * probas - 1.0)


def _safe_correlation(series: pd.Series, labels: pd.Series) -> float:
    """Safely compute Pearson correlation, returning 0 if undefined."""
    if series.nunique(dropna=True) < 2 or labels.nunique(dropna=True) < 2:
        return 0.0
    value = series.corr(labels)
    return 0.0 if pd.isna(value) else float(value)


def _compute_oof_probabilities(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    config: Dict[str, Any],
) -> np.ndarray:
    """Compute out-of-fold probabilities for robust label-noise detection.

    Using in-sample predictions on a flexible RandomForest is misleading because the model
    can memorize noisy labels. Out-of-fold probabilities are a much fairer classical signal.
    """
    cv_folds = int(config.get("LABEL_NOISE_CV_FOLDS", 5))
    cv_folds = max(2, cv_folds)

    # Make sure we do not ask for more folds than the smallest class supports.
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

    # cross_val_predict keeps the original row order of train_features.
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
    model: RandomForestClassifier,
    train_split: Dict[str, pd.DataFrame | pd.Series],
    fault_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Detect label noise using classical non-XAI signals.

    Strategy:
    1. Compute out-of-fold probabilities on training data
    2. Build a suspicion score from:
       - low probability assigned to the observed label
       - prediction/label disagreement
       - low confidence margin / high uncertainty
    3. Rank training samples by suspicion score
    4. Compare the top-k ranked samples with the actually flipped labels

    Returns dict with suspect_indices, scores, precision_at_k, recall_at_k.
    """
    train_features = train_split["features"]
    train_labels = train_split["labels"]

    # Step 1: robust classical signal = OOF predictions instead of in-sample predictions
    oof_proba_pos = _compute_oof_probabilities(train_features, train_labels, config)
    observed_labels = train_labels.to_numpy().astype(int)
    predicted_labels = (oof_proba_pos >= 0.5).astype(int)

    # Step 2: compute sample-wise classical suspicion signals
    prob_observed_label = np.where(observed_labels == 1, oof_proba_pos, 1.0 - oof_proba_pos)
    disagreement = (predicted_labels != observed_labels).astype(float)
    margins = _compute_margins_from_proba(oof_proba_pos)
    uncertainty = 1.0 - margins

    # Higher = more suspicious.
    # - low prob for observed label is the strongest signal
    # - outright disagreement is a strong bonus
    # - uncertainty adds a softer secondary signal
    suspicion_score = (
        0.60 * (1.0 - prob_observed_label)
        + 0.25 * disagreement
        + 0.15 * uncertainty
    )

    # Step 3: rank samples by suspicion score, but keep REAL dataframe indices
    rank_positions = np.argsort(-suspicion_score)
    ranked_index_labels = train_features.index.to_numpy()[rank_positions]
    ranked_scores = suspicion_score[rank_positions]

    suspect_indices = [int(idx) for idx in ranked_index_labels.tolist()]
    suspect_scores = ranked_scores.tolist()

    # Step 4: evaluate against injected ground truth
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
    """
    contaminated_eval_features = splits["contaminated_eval"]["features"]
    contaminated_eval_labels = splits["contaminated_eval"]["labels"]
    clean_holdout_features = splits["clean_holdout"]["features"]
    clean_holdout_labels = splits["clean_holdout"]["labels"]

    feature_importance = pd.Series(model.feature_importances_, index=contaminated_eval_features.columns)

    contaminated_corr = contaminated_eval_features.apply(
        lambda col: abs(_safe_correlation(col, contaminated_eval_labels))
    )
    clean_corr = clean_holdout_features.apply(
        lambda col: abs(_safe_correlation(col, clean_holdout_labels))
    )

    instability = contaminated_corr - clean_corr
    suspicion = 0.65 * instability + 0.35 * feature_importance
    suspect_features = suspicion.sort_values(ascending=False).index.tolist()

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
    1. Compute feature importances from model
    2. Compute feature correlations in train, contaminated_eval, and clean_holdout
    3. Identify features important offline but unstable in clean_holdout
    4. Rank by combined instability + importance
    5. Find rank of true spurious feature
    """
    train_features = splits["train"]["features"]
    train_labels = splits["train"]["labels"]
    contaminated_eval_features = splits["contaminated_eval"]["features"]
    contaminated_eval_labels = splits["contaminated_eval"]["labels"]
    clean_holdout_features = splits["clean_holdout"]["features"]
    clean_holdout_labels = splits["clean_holdout"]["labels"]

    feature_importance = pd.Series(model.feature_importances_, index=train_features.columns)

    train_corr = train_features.apply(lambda col: abs(_safe_correlation(col, train_labels)))
    contaminated_corr = contaminated_eval_features.apply(
        lambda col: abs(_safe_correlation(col, contaminated_eval_labels))
    )
    clean_corr = clean_holdout_features.apply(
        lambda col: abs(_safe_correlation(col, clean_holdout_labels))
    )

    max_offline_corr = pd.concat([train_corr, contaminated_corr], axis=1).max(axis=1)
    instability = max_offline_corr - clean_corr
    suspicion = 0.70 * instability + 0.30 * feature_importance
    suspect_features = suspicion.sort_values(ascending=False).index.tolist()

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
    suspect_indices: List[int],
    max_fixes: int,
) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
    """
    Fix label noise by correcting ONLY the top-k suspected samples using original labels.

    This is the key bugfix:
    - previously, all changed labels were reverted regardless of the detector output
    - now, only the top-k suspects are corrected
    """
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
    2. Perform fault-specific detection
    3. Apply fix
    4. Retrain model on fixed data
    5. Compute metrics after fix
    6. Return standardized results
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
        detection_result = _detect_label_noise(model, injected_splits["train"], fault_metadata, config)
        # OOF prediction + score construction + ranking + localization
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
        detection_result = _detect_data_leakage(model, injected_splits, fault_metadata, config)
        steps_to_detect = 4

        suspect_feature = detection_result["top_candidate_feature"]
        if suspect_feature:
            fixed_splits = _apply_feature_removal_fix(injected_splits, suspect_feature)
        else:
            fixed_splits = injected_splits

        retrained_model = RandomForestClassifier(
            n_estimators=int(config.get("N_ESTIMATORS", 200)),
            random_state=int(config.get("RANDOM_STATE", 42)),
        )
        retrained_model.fit(fixed_splits["train"]["features"], fixed_splits["train"]["labels"])
        metrics_after = _compute_split_metrics(retrained_model, fixed_splits)

    elif fault_type == "spurious_correlation":
        detection_result = _detect_spurious_correlation(model, injected_splits, fault_metadata, config)
        steps_to_detect = 4

        suspect_feature = detection_result["top_candidate_feature"]
        if suspect_feature:
            fixed_splits = _apply_feature_removal_fix(injected_splits, suspect_feature)
        else:
            fixed_splits = injected_splits

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