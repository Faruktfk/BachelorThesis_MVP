"""SHAP-based debugging logic for feature faults and label noise.

This module implements an XAI debugging workflow that is intentionally comparable to
baseline_debugging.py, but uses SHAP values to build suspect rankings.

Methodological rules:
- Detection uses only train + contaminated_eval
- clean_holdout is used only for objective evaluation after the ranking/fix decision
- For feature faults, ranking is hybrid:
  it combines global SHAP strength with local spike / concentration behavior
- For label noise, SHAP acts as a reranker on top of a strong classical OOF baseline
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

from evaluation_metrics import compute_split_metrics as shared_compute_split_metrics


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
    """Compute shared metrics for all splits.

    This delegates metric computation to evaluation_metrics.py so that
    Baseline, XAI and Oracle repair are evaluated identically.
    """
    return shared_compute_split_metrics(model, splits)


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
    """Build a SHAP explainer with probability-scale attempt first, raw fallback second."""
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


def _compute_feature_focus_slice(
    model: RandomForestClassifier,
    offline_features: pd.DataFrame,
    abs_shap_values: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a focus slice for feature-fault ranking.

    We want a subset of samples where the model appears to rely strongly on a small number
    of features. This helps both:
    - direct leakage (globally dominant feature)
    - subgroup-local faults (local spikes in a subset)
    """
    probabilities = model.predict_proba(offline_features)[:, 1]
    confidence = np.abs(2.0 * probabilities - 1.0)

    total_abs = abs_shap_values.sum(axis=1)
    total_abs_safe = total_abs.copy()
    total_abs_safe[total_abs_safe <= 1e-12] = 1.0

    share = abs_shap_values / total_abs_safe[:, None]
    dominance = share.max(axis=1)

    risk = (
        0.45 * _normalize_series(pd.Series(dominance)).to_numpy()
        + 0.30 * _normalize_series(pd.Series(total_abs)).to_numpy()
        + 0.25 * _normalize_series(pd.Series(confidence)).to_numpy()
    )

    focus_fraction = float(config.get("XAI_FEATURE_FOCUS_FRACTION", 0.20))
    focus_fraction = min(max(focus_fraction, 0.05), 0.50)
    focus_n = max(10, int(np.ceil(focus_fraction * len(offline_features))))
    focus_idx = np.argsort(-risk)[:focus_n]

    return focus_idx, share


def _feature_hybrid_statistics(
    model: RandomForestClassifier,
    offline_features: pd.DataFrame,
    abs_shap_values: np.ndarray,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Compute hybrid SHAP feature statistics.

    This score intentionally combines:
    - global signal (for direct leakage or broadly important shortcuts)
    - local spikes and concentration (for subgroup-local faults)
    """
    n_samples, n_features = abs_shap_values.shape
    feature_names = list(offline_features.columns)

    focus_idx, share = _compute_feature_focus_slice(model, offline_features, abs_shap_values, config)

    topk = min(3, n_features)
    topk_idx = np.argpartition(abs_shap_values, kth=n_features - topk, axis=1)[:, -topk:]
    winner_idx = np.argmax(abs_shap_values, axis=1)

    topk_counts = np.zeros(n_features, dtype=float)
    winner_counts = np.zeros(n_features, dtype=float)

    for row in topk_idx:
        topk_counts[row] += 1.0
    for idx in winner_idx:
        winner_counts[idx] += 1.0

    topk_rate = topk_counts / float(n_samples)
    winner_rate = winner_counts / float(n_samples)

    focus_topk_idx = topk_idx[focus_idx]
    focus_winner_idx = winner_idx[focus_idx]

    focus_topk_counts = np.zeros(n_features, dtype=float)
    focus_winner_counts = np.zeros(n_features, dtype=float)

    for row in focus_topk_idx:
        focus_topk_counts[row] += 1.0
    for idx in focus_winner_idx:
        focus_winner_counts[idx] += 1.0

    focus_topk_rate = focus_topk_counts / float(len(focus_idx))
    focus_winner_rate = focus_winner_counts / float(len(focus_idx))

    top_count = max(1, int(np.ceil(0.10 * n_samples)))

    rows: Dict[str, Dict[str, float]] = {}
    for j, feature_name in enumerate(feature_names):
        values = abs_shap_values[:, j]
        focus_values = abs_shap_values[focus_idx, j]
        shares = share[:, j]
        focus_shares = share[focus_idx, j]

        mean_abs = float(np.mean(values))
        q90_abs = float(np.quantile(values, 0.90))
        q95_abs = float(np.quantile(values, 0.95))
        top_sorted = np.sort(values)
        top10_share = float(np.sum(top_sorted[-top_count:]) / (np.sum(values) + 1e-12))

        focus_mean_abs = float(np.mean(focus_values))
        focus_mean_share = float(np.mean(focus_shares))
        focus_q90_abs = float(np.quantile(focus_values, 0.90))

        rows[feature_name] = {
            "mean_abs_shap": mean_abs,
            "q90_abs_shap": q90_abs,
            "q95_abs_shap": q95_abs,
            "top10_share": top10_share,
            "top3_rate": float(topk_rate[j]),
            "winner_rate": float(winner_rate[j]),
            "focus_mean_abs_shap": focus_mean_abs,
            "focus_q90_abs_shap": focus_q90_abs,
            "focus_mean_share": focus_mean_share,
            "focus_top3_rate": float(focus_topk_rate[j]),
            "focus_winner_rate": float(focus_winner_rate[j]),
        }

    stats = pd.DataFrame.from_dict(rows, orient="index")

    score = (
        0.20 * _normalize_series(stats["mean_abs_shap"])
        + 0.12 * _normalize_series(stats["q95_abs_shap"])
        + 0.08 * _normalize_series(stats["top10_share"])
        + 0.10 * _normalize_series(stats["top3_rate"])
        + 0.10 * _normalize_series(stats["winner_rate"])
        + 0.16 * _normalize_series(stats["focus_mean_abs_shap"])
        + 0.08 * _normalize_series(stats["focus_q90_abs_shap"])
        + 0.10 * _normalize_series(stats["focus_mean_share"])
        + 0.03 * _normalize_series(stats["focus_top3_rate"])
        + 0.03 * _normalize_series(stats["focus_winner_rate"])
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
    """SHAP ranking for feature-based faults using a hybrid global+local score.

    We intentionally use the whole offline world:
    - train
    - contaminated_eval

    because the developer is allowed to inspect both, while clean_holdout remains hidden.
    """
    offline_features = pd.concat([train_features, contaminated_eval_features], axis=0)

    background = _build_background_sample(train_features, config)
    explainer = _build_tree_explainer(model, background)
    shap_values = _extract_positive_class_shap_values(explainer, offline_features)
    abs_shap = np.abs(shap_values)

    stats = _feature_hybrid_statistics(model, offline_features, abs_shap, config)
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
    explainer,
    contaminated_eval_features: pd.DataFrame,
    contaminated_eval_labels: pd.Series,
    contaminated_eval_pred_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build class-specific SHAP profile prototypes from contaminated_eval."""
    shap_eval = _extract_positive_class_shap_values(explainer, contaminated_eval_features)
    abs_eval = np.abs(shap_eval)
    eval_profiles = _row_normalize_abs_profiles(abs_eval)

    eval_pred = (contaminated_eval_pred_proba >= 0.5).astype(int)
    eval_margin = np.abs(2.0 * contaminated_eval_pred_proba - 1.0)

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
    """Detect label noise using a two-stage strategy.

    Stage 1:
    - classical OOF ranking over all train samples

    Stage 2:
    - SHAP only reranks the strongest baseline candidates
    - this keeps the strong baseline behavior and uses SHAP as support, not replacement
    """
    train_features = train_split["features"]
    train_labels = train_split["labels"]
    contaminated_eval_features = contaminated_eval_split["features"]
    contaminated_eval_labels = contaminated_eval_split["labels"]

    # ---------- Stage 1: strong classical baseline ranking ----------
    oof_proba_pos = _compute_oof_probabilities(train_features, train_labels, config)
    observed_labels = train_labels.to_numpy().astype(int)
    predicted_labels = (oof_proba_pos >= 0.5).astype(int)

    prob_observed_label = np.where(observed_labels == 1, oof_proba_pos, 1.0 - oof_proba_pos)
    disagreement = (predicted_labels != observed_labels).astype(float)
    margins = np.abs(2.0 * oof_proba_pos - 1.0)
    uncertainty = 1.0 - margins

    baseline_core = (
        0.60 * (1.0 - prob_observed_label)
        + 0.25 * disagreement
        + 0.15 * uncertainty
    )

    baseline_rank_positions = np.argsort(-baseline_core)
    baseline_ranked_indices = train_features.index.to_numpy()[baseline_rank_positions]

    if "changed_indices" in fault_metadata:
        actual_noisy_indices = set(int(idx) for idx in fault_metadata["changed_indices"])
        k = len(actual_noisy_indices)
    else:
        actual_noisy_indices = set()
        k = 0

    rerank_multiplier = float(config.get("XAI_LABEL_CANDIDATE_MULTIPLIER", 3.0))
    candidate_pool_size = max(40, int(np.ceil(max(1, k) * rerank_multiplier)))
    candidate_pool_size = min(candidate_pool_size, len(train_features))

    candidate_positions = baseline_rank_positions[:candidate_pool_size]
    candidate_index_labels = train_features.index.to_numpy()[candidate_positions]
    candidate_features = train_features.loc[candidate_index_labels]

    # ---------- Stage 2: SHAP reranking on candidate pool ----------
    background = _build_background_sample(train_features, config)
    explainer = _build_tree_explainer(model, background)

    contaminated_eval_pred_proba = model.predict_proba(contaminated_eval_features)[:, 1]
    proto_label0, proto_label1 = _build_clean_eval_shap_prototypes(
        explainer,
        contaminated_eval_features,
        contaminated_eval_labels,
        contaminated_eval_pred_proba,
    )

    shap_candidates = _extract_positive_class_shap_values(explainer, candidate_features)
    abs_candidates = np.abs(shap_candidates)
    candidate_profiles = _row_normalize_abs_profiles(abs_candidates)

    candidate_labels = train_labels.loc[candidate_index_labels].to_numpy().astype(int)
    profile_mismatch = np.zeros(len(candidate_index_labels), dtype=float)

    for i, label in enumerate(candidate_labels):
        same_proto = proto_label1 if label == 1 else proto_label0
        other_proto = proto_label0 if label == 1 else proto_label1

        sim_same = _safe_cosine_similarity(candidate_profiles[i], same_proto)
        sim_other = _safe_cosine_similarity(candidate_profiles[i], other_proto)
        profile_mismatch[i] = max(0.0, sim_other - sim_same)

    candidate_baseline = baseline_core[candidate_positions]
    candidate_baseline_norm = _normalize_series(pd.Series(candidate_baseline)).to_numpy()
    profile_mismatch_norm = _normalize_series(pd.Series(profile_mismatch)).to_numpy()

    baseline_weight = float(config.get("XAI_LABEL_BASELINE_WEIGHT", 0.80))
    profile_weight = float(config.get("XAI_LABEL_PROFILE_WEIGHT", 0.20))

    candidate_combined = (
        baseline_weight * candidate_baseline_norm
        + profile_weight * profile_mismatch_norm
    )

    candidate_order = np.argsort(-candidate_combined)
    reranked_candidate_indices = candidate_index_labels[candidate_order]

    remaining_positions = baseline_rank_positions[candidate_pool_size:]
    remaining_indices = train_features.index.to_numpy()[remaining_positions]

    suspect_indices = [int(idx) for idx in reranked_candidate_indices.tolist()] + [int(idx) for idx in remaining_indices.tolist()]
    suspect_scores = candidate_combined[candidate_order].tolist() + baseline_core[remaining_positions].tolist()

    if k > 0:
        top_k_suspects = suspect_indices[:k]
        suspect_set = set(top_k_suspects)

        true_positives = len(suspect_set & actual_noisy_indices)
        precision_at_k = float(true_positives) / float(k) if k > 0 else 0.0
        recall_at_k = float(true_positives) / float(len(actual_noisy_indices)) if len(actual_noisy_indices) > 0 else 0.0
    else:
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