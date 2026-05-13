"""SHAP-based debugging logic for feature faults and label noise.

This module implements an XAI debugging workflow that is intentionally comparable to
baseline_debugging.py, but uses SHAP values to build suspect rankings.

Methodological rules:
- Detection uses only train + contaminated_eval
- clean_holdout is used only for objective evaluation after the ranking/fix decision
- For feature faults, ranking is subgroup-sensitive:
  it emphasizes tail SHAP behavior and concentration instead of only global mean importance
- For label noise, ranking combines OOF classical suspicion with SHAP profile mismatch
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _compute_margins_from_proba(proba: np.ndarray) -> np.ndarray:
    """Convert positive-class probabilities to symmetric margins in [0,1]."""
    return np.abs(2.0 * np.asarray(proba) - 1.0)


def _normalize_series(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)


def _compute_split_metrics(
    model: RandomForestClassifier,
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
) -> Dict[str, float]:
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


def _compute_oof_probabilities(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    config: Dict[str, Any],
) -> np.ndarray:
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


def _build_background_sample(
    train_features: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    background_size = int(config.get("SHAP_BACKGROUND_SIZE", 120))
    background_size = max(20, min(background_size, len(train_features)))
    return train_features.sample(
        n=background_size,
        random_state=int(config.get("RANDOM_STATE", 42)),
        replace=False,
    )


def _build_tree_explainer(
    model: RandomForestClassifier,
    background: pd.DataFrame,
):
    """Build a SHAP explainer with a probability-scale attempt first, raw fallback second."""
    try:
        return shap.TreeExplainer(
            model,
            data=background,
            model_output="probability",
            feature_perturbation="interventional",
        )
    except Exception:
        return shap.TreeExplainer(model)


def _extract_positive_class_shap_values(
    explainer,
    features: pd.DataFrame,
) -> np.ndarray:
    """Extract SHAP values for the positive class robustly across SHAP versions."""
    raw = None

    try:
        explanation = explainer(features, check_additivity=False)
        raw = getattr(explanation, "values", explanation)
    except Exception:
        pass

    if raw is None:
        try:
            raw = explainer.shap_values(features, check_additivity=False)
        except TypeError:
            raw = explainer.shap_values(features)

    if isinstance(raw, list):
        values = raw[1] if len(raw) > 1 else raw[0]
        return np.asarray(values, dtype=float)

    values = np.asarray(raw, dtype=float)

    if values.ndim == 2:
        return values

    if values.ndim == 3:
        # Common shapes:
        # (n_samples, n_features, n_outputs)
        # (n_outputs, n_samples, n_features)
        if values.shape[-1] == 2:
            return values[:, :, 1]
        if values.shape[0] == 2:
            return values[1]
        if values.shape[-1] == 1:
            return np.squeeze(values, axis=-1)

    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def _safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _row_normalize_abs_profiles(abs_profiles: np.ndarray) -> np.ndarray:
    row_sums = abs_profiles.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 1e-12] = 1.0
    return abs_profiles / row_sums


def _feature_tail_statistics(
    abs_shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Compute subgroup-sensitive SHAP summary statistics per feature.

    The goal is to surface features that:
    - are not globally dominant,
    - but have strong local bursts in a subset of samples.
    """
    n_samples, n_features = abs_shap_values.shape
    top_count = max(1, int(np.ceil(0.10 * n_samples)))
    topk = min(3, n_features)

    topk_idx = np.argpartition(abs_shap_values, kth=n_features - topk, axis=1)[:, -topk:]
    topk_counts = np.zeros(n_features, dtype=float)
    for row in topk_idx:
        topk_counts[row] += 1.0
    topk_rate = topk_counts / float(n_samples)

    rows: Dict[str, Dict[str, float]] = {}
    for j, feature_name in enumerate(feature_names):
        values = abs_shap_values[:, j]
        mean_abs = float(np.mean(values))
        q90 = float(np.quantile(values, 0.90))
        q95 = float(np.quantile(values, 0.95))
        top_sorted = np.sort(values)
        top_share = float(np.sum(top_sorted[-top_count:]) / (np.sum(values) + 1e-12))
        peak_ratio = float(q95 / (mean_abs + 1e-12))
        rows[feature_name] = {
            "mean_abs_shap": mean_abs,
            "q90_abs_shap": q90,
            "q95_abs_shap": q95,
            "top10_share": top_share,
            "peak_ratio": peak_ratio,
            "top3_rate": float(topk_rate[j]),
        }

    stats = pd.DataFrame.from_dict(rows, orient="index")

    score = (
        0.40 * _normalize_series(stats["peak_ratio"])
        + 0.25 * _normalize_series(stats["top10_share"])
        + 0.20 * _normalize_series(stats["q95_abs_shap"])
        + 0.10 * _normalize_series(stats["top3_rate"])
        + 0.05 * _normalize_series(stats["mean_abs_shap"])
    )
    stats["xai_suspicion"] = score
    return stats.sort_values("xai_suspicion", ascending=False)


def _detect_feature_fault_with_shap(
    model: RandomForestClassifier,
    train_features: pd.DataFrame,
    contaminated_eval_features: pd.DataFrame,
    fault_metadata: Dict[str, Any],
    true_feature_key: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """SHAP ranking for feature-based faults using subgroup-sensitive tail behavior."""
    background = _build_background_sample(train_features, config)
    explainer = _build_tree_explainer(model, background)
    shap_values = _extract_positive_class_shap_values(explainer, contaminated_eval_features)
    abs_shap = np.abs(shap_values)

    stats = _feature_tail_statistics(abs_shap, list(contaminated_eval_features.columns))
    suspect_features = stats.index.tolist()
    suspicion_scores = {feature: float(stats.loc[feature, "xai_suspicion"]) for feature in suspect_features}

    true_feature = fault_metadata.get(true_feature_key)
    if true_feature is not None:
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
        "feature_tail_table": stats.head(10).to_dict(orient="index"),
    }


def _build_clean_eval_shap_prototypes(
    model: RandomForestClassifier,
    train_features: pd.DataFrame,
    contaminated_eval_features: pd.DataFrame,
    contaminated_eval_labels: pd.Series,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build class-specific SHAP profile prototypes from contaminated_eval.

    Rationale:
    - for label noise, train labels are corrupted
    - contaminated_eval labels are clean in this setup
    - therefore contaminated_eval is a fair offline reference set for 'normal' explanation profiles
    """
    background = _build_background_sample(train_features, config)
    explainer = _build_tree_explainer(model, background)
    shap_eval = _extract_positive_class_shap_values(explainer, contaminated_eval_features)
    abs_eval = np.abs(shap_eval)
    eval_profiles = _row_normalize_abs_profiles(abs_eval)

    eval_proba = model.predict_proba(contaminated_eval_features)[:, 1]
    eval_pred = (eval_proba >= 0.5).astype(int)
    eval_margin = np.abs(2.0 * eval_proba - 1.0)

    prototypes = []
    for label in [0, 1]:
        mask = (contaminated_eval_labels.to_numpy().astype(int) == label)
        confident_agree_mask = mask & (eval_pred == label) & (eval_margin >= np.quantile(eval_margin, 0.50))

        if confident_agree_mask.sum() >= 3:
            proto = eval_profiles[confident_agree_mask].mean(axis=0)
        elif mask.sum() >= 1:
            proto = eval_profiles[mask].mean(axis=0)
        else:
            proto = eval_profiles.mean(axis=0)

        norm = np.linalg.norm(proto)
        if norm > 1e-12:
            proto = proto / norm
        prototypes.append(proto)

    return prototypes[0], prototypes[1]


def _detect_label_noise_with_shap(
    model: RandomForestClassifier,
    train_split: Dict[str, pd.DataFrame | pd.Series],
    contaminated_eval_split: Dict[str, pd.DataFrame | pd.Series],
    fault_metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Detect label noise using OOF baseline signals + SHAP profile mismatch."""
    train_features = train_split["features"]
    train_labels = train_split["labels"]
    contaminated_eval_features = contaminated_eval_split["features"]
    contaminated_eval_labels = contaminated_eval_split["labels"]

    # Classical baseline part (same philosophy as baseline arm)
    oof_proba_pos = _compute_oof_probabilities(train_features, train_labels, config)
    observed_labels = train_labels.to_numpy().astype(int)
    predicted_labels = (oof_proba_pos >= 0.5).astype(int)

    prob_observed_label = np.where(observed_labels == 1, oof_proba_pos, 1.0 - oof_proba_pos)
    disagreement = (predicted_labels != observed_labels).astype(float)
    margins = _compute_margins_from_proba(oof_proba_pos)
    uncertainty = 1.0 - margins

    baseline_core = (
        0.60 * (1.0 - prob_observed_label)
        + 0.25 * disagreement
        + 0.15 * uncertainty
    )

    # SHAP profile mismatch against clean contaminated_eval prototypes
    background = _build_background_sample(train_features, config)
    explainer = _build_tree_explainer(model, background)

    shap_train = _extract_positive_class_shap_values(explainer, train_features)
    abs_train = np.abs(shap_train)
    train_profiles = _row_normalize_abs_profiles(abs_train)

    proto_label0, proto_label1 = _build_clean_eval_shap_prototypes(
        model,
        train_features,
        contaminated_eval_features,
        contaminated_eval_labels,
        config,
    )

    profile_mismatch = np.zeros(len(train_profiles), dtype=float)
    for i, label in enumerate(observed_labels):
        same_proto = proto_label1 if label == 1 else proto_label0
        other_proto = proto_label0 if label == 1 else proto_label1

        sim_same = _safe_cosine_similarity(train_profiles[i], same_proto)
        sim_other = _safe_cosine_similarity(train_profiles[i], other_proto)
        profile_mismatch[i] = max(0.0, sim_other - sim_same)

    baseline_core_norm = _normalize_series(pd.Series(baseline_core)).to_numpy()
    profile_mismatch_norm = _normalize_series(pd.Series(profile_mismatch)).to_numpy()

    final_suspicion = (
        float(config.get("XAI_LABEL_BASELINE_WEIGHT", 0.65)) * baseline_core_norm
        + float(config.get("XAI_LABEL_PROFILE_WEIGHT", 0.35)) * profile_mismatch_norm
    )

    rank_positions = np.argsort(-final_suspicion)
    ranked_index_labels = train_features.index.to_numpy()[rank_positions]
    ranked_scores = final_suspicion[rank_positions]

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


def _apply_label_noise_fix(
    train_split: Dict[str, pd.DataFrame | pd.Series],
    fault_metadata: Dict[str, Any],
    suspect_indices: List[int],
    max_fixes: int,
) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
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


def run_xai_debugging(
    model: RandomForestClassifier,
    fault_type: str,
    fault_metadata: Dict[str, Any],
    injected_splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the SHAP-based debugging workflow."""
    if config is None:
        config = {}

    start_time = time.time()
    metrics_before = _compute_split_metrics(model, injected_splits)

    if fault_type == "none":
        metrics_after = metrics_before
        fix_impact = {key: 0.0 for key in metrics_before.keys()}
        return {
            "workflow": "xai_shap",
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
        detection_result = _detect_label_noise_with_shap(
            model,
            injected_splits["train"],
            injected_splits["contaminated_eval"],
            fault_metadata,
            config,
        )
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
        detection_result = _detect_feature_fault_with_shap(
            model,
            injected_splits["train"]["features"],
            injected_splits["contaminated_eval"]["features"],
            fault_metadata,
            true_feature_key="leakage_feature_name",
            config=config,
        )
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
        detection_result = _detect_feature_fault_with_shap(
            model,
            injected_splits["train"]["features"],
            injected_splits["contaminated_eval"]["features"],
            fault_metadata,
            true_feature_key="feature_name",
            config=config,
        )
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
            "workflow": "xai_shap",
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
        "workflow": "xai_shap",
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
