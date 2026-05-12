from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.model_selection import train_test_split


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

# Spurious correlation config
SPURIOUS_MODE = "broken"  # options: "broken", "inverted"
SPURIOUS_STRENGTH = 0.90
USE_GROUPS_FOR_SPURIOUS = True

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


def _probe_uncertainty(train_features: pd.DataFrame, train_labels: pd.Series) -> pd.Series:
    """Estimate uncertainty with a small probe model for hard label noise."""
    probe = RandomForestClassifier(n_estimators=PROBE_N_ESTIMATORS, random_state=RANDOM_STATE)
    probe.fit(train_features, train_labels)
    probabilities = probe.predict_proba(train_features)
    top_two = np.sort(probabilities, axis=1)[:, -2:]
    margins = top_two[:, 1] - top_two[:, 0]
    return pd.Series(margins, index=train_features.index)


def _safe_corr(feature: pd.Series, labels: pd.Series) -> float:
    if feature.nunique(dropna=True) < 2 or labels.nunique(dropna=True) < 2:
        return 0.0
    value = feature.corr(labels)
    return 0.0 if pd.isna(value) else float(value)


def _feature_proxy(feature: pd.Series, labels: pd.Series) -> Dict[str, float]:
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
        "correlation": _safe_corr(feature, labels),
        "mutual_information": float(mutual_info_score(labels.astype(int), discrete)),
        "mean_gap": mean_gap,
    }


def _normalize_series(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)


def _quantile_edges(series: pd.Series, bins: int = 5) -> np.ndarray:
    values = series.dropna().to_numpy()
    if values.size == 0:
        return np.array([0.0, 1.0])

    unique_values = np.unique(values)
    if unique_values.size == 1:
        center = float(unique_values[0])
        return np.array([center - 1e-6, center + 1e-6])

    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        minimum = float(values.min())
        maximum = float(values.max())
        if np.isclose(minimum, maximum):
            return np.array([minimum - 1e-6, maximum + 1e-6])
        edges = np.linspace(minimum, maximum, min(6, unique_values.size + 1))

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_series(series: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.cut(series, bins=edges, include_lowest=True, labels=False).fillna(-1).astype(int)


def _cluster_keys(frame: pd.DataFrame, feature_a: str, feature_b: str, edges_a: np.ndarray, edges_b: np.ndarray) -> pd.Series:
    bins_a = _bin_series(frame[feature_a], edges_a)
    bins_b = _bin_series(frame[feature_b], edges_b)
    return bins_a.astype(str) + "_" + bins_b.astype(str)


def _smoothed_target_encoding(keys: pd.Series, labels: pd.Series, smoothing: float = 5.0) -> Dict[str, float]:
    global_mean = float(labels.mean())
    group_stats = labels.groupby(keys).agg(["mean", "count"])
    encoded: Dict[str, float] = {}
    for key, row in group_stats.iterrows():
        encoded[key] = float((row["mean"] * row["count"] + smoothing * global_mean) / (row["count"] + smoothing))
    encoded["__global__"] = global_mean
    return encoded


def _apply_lookup(keys: pd.Series, lookup: Dict[str, float]) -> pd.Series:
    fallback = lookup["__global__"]
    return keys.map(lambda key: lookup.get(key, fallback)).astype(float)


def _inject_label_noise(train_features: pd.DataFrame, train_labels: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """Corrupt only training labels. Hard mode flips the most uncertain samples."""
    rng = np.random.default_rng(RANDOM_STATE)
    noisy_labels = train_labels.copy()
    noise_count = max(1, int(round(len(noisy_labels) * LABEL_NOISE_RATE)))

    if LABEL_NOISE_MODE == "hard":
        uncertainty = _probe_uncertainty(train_features, noisy_labels)
        changed_indices = uncertainty.nsmallest(noise_count).index
        changed_uncertainty = {int(index): float(uncertainty.loc[index]) for index in changed_indices}
    else:
        changed_indices = pd.Index(rng.choice(noisy_labels.index.to_numpy(), size=noise_count, replace=False))
        changed_uncertainty = {}

    original_labels = {int(index): int(noisy_labels.loc[index]) for index in changed_indices}
    noisy_labels.loc[changed_indices] = 1 - noisy_labels.loc[changed_indices]

    metadata: Dict[str, Any] = {
        "fault_type": "label_noise",
        "injected": True,
        "noise_mode": LABEL_NOISE_MODE,
        "noise_rate": LABEL_NOISE_RATE,
        "train_count": int(len(train_labels)),
        "changed_count": int(len(changed_indices)),
        "changed_indices": [int(index) for index in changed_indices],
        "original_labels_by_index": original_labels,
        "contaminated_splits": ["train"],
        "contaminated_eval_contaminated": False,
        "clean_holdout_contaminated": False,
        "injection_description": "Training labels are corrupted; evaluation labels stay clean.",
    }

    if changed_uncertainty:
        metadata["changed_uncertainty_scores"] = changed_uncertainty
        metadata["mean_changed_uncertainty"] = float(np.mean(list(changed_uncertainty.values())))

    return noisy_labels, metadata


def _direct_leakage_values(labels: pd.Series, strength: float, noisy_baseline: float | None = None) -> pd.Series:
    rng = np.random.default_rng(RANDOM_STATE)
    baseline = float(labels.mean()) if noisy_baseline is None else float(noisy_baseline)
    signal = labels.astype(float).to_numpy(copy=True)
    noise = rng.normal(loc=baseline, scale=max(0.08, (1.0 - strength) * 0.35), size=len(labels))
    values = strength * signal + (1.0 - strength) * noise
    return pd.Series(np.clip(values, 0.0, 1.0), index=labels.index)


def _indirect_leakage_values(
    source_frame: pd.DataFrame,
    labels: pd.Series,
    feature_a: str,
    feature_b: str,
    clean_reference_labels: pd.Series,
    clean_reference_frame: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Create indirect leakage via multi-feature cluster target encoding.
    
    More realistic than direct: simulates a pipeline mistake where global target statistics
    leak into feature engineering. Uses combined bins of multiple features with minimal smoothing
    to create a strong but not trivial leakage signal.
    """
    combined_edges_a = _quantile_edges(source_frame[feature_a], bins=5)
    combined_edges_b = _quantile_edges(source_frame[feature_b], bins=5)

    combined_keys = _cluster_keys(source_frame, feature_a, feature_b, combined_edges_a, combined_edges_b)
    # Minimal smoothing (1.5) for contaminated view to make leakage stronger without being trivial
    leaky_lookup = _smoothed_target_encoding(combined_keys, labels, smoothing=1.5)

    clean_keys = _cluster_keys(clean_reference_frame, feature_a, feature_b, combined_edges_a, combined_edges_b)
    # Higher smoothing (10.0) for clean view to remove spurious correlation
    clean_lookup = _smoothed_target_encoding(clean_keys, clean_reference_labels, smoothing=10.0)

    leaky_values = _apply_lookup(combined_keys, leaky_lookup)
    clean_values = _apply_lookup(clean_keys, clean_lookup)

    metadata = {
        "construction": f"cross-binned target encoding over {feature_a} and {feature_b} with minimal smoothing",
        "source_variables": [feature_a, feature_b],
        "statistic": "smoothed_cluster_target_mean",
        "leakage_strength_rationale": "Minimal smoothing (1.5) on contaminated view creates strong information leak; high smoothing (10.0) on clean view breaks the spurious signal.",
    }
    return leaky_values, clean_values, metadata


def _inject_data_leakage(
    train_split: Dict[str, pd.DataFrame | pd.Series],
    contaminated_eval_split: Dict[str, pd.DataFrame | pd.Series],
    clean_holdout_split: Dict[str, pd.DataFrame | pd.Series],
) -> Tuple[Dict[str, pd.DataFrame | pd.Series], Dict[str, Any]]:
    train_features = train_split["features"].copy()
    contaminated_eval_features = contaminated_eval_split["features"].copy()
    clean_holdout_features = clean_holdout_split["features"].copy()

    train_labels = train_split["labels"]
    contaminated_eval_labels = contaminated_eval_split["labels"]
    clean_holdout_labels = clean_holdout_split["labels"]

    metadata: Dict[str, Any] = {
        "fault_type": "data_leakage",
        "injected": True,
        "leakage_mode": LEAKAGE_MODE,
        "contaminated_splits": ["train", "contaminated_eval"],
        "contaminated_eval_contaminated": True,
        "clean_holdout_contaminated": False,
    }

    if LEAKAGE_MODE == "direct":
        leakage_feature_name = "leakage_direct_signal"
        train_leakage = _direct_leakage_values(train_labels, LEAKAGE_STRENGTH)
        contaminated_eval_leakage = _direct_leakage_values(contaminated_eval_labels, LEAKAGE_STRENGTH)
        clean_holdout_leakage = pd.Series(
            np.random.default_rng(RANDOM_STATE).normal(
                loc=float(train_labels.mean()),
                scale=0.18,
                size=len(clean_holdout_labels),
            ).clip(0.0, 1.0),
            index=clean_holdout_labels.index,
        )

        metadata.update(
            {
                "leakage_feature_name": leakage_feature_name,
                "construction": "noisy label copy for train and contaminated_eval; independent noise for clean_holdout",
                "orig_variables": ["label"],
                "statistic": "noisy_label_signal",
            }
        )
    else:
        combined_source = pd.concat([train_features, contaminated_eval_features], axis=0)
        numeric = combined_source.select_dtypes(include=[np.number])
        candidate_features = numeric.var(axis=0).sort_values(ascending=False).head(2).index.tolist()
        if len(candidate_features) < 2:
            candidate_features = list(numeric.columns[:2])

        source_a, source_b = candidate_features[0], candidate_features[1]
        leaky_values, clean_values, construction_meta = _indirect_leakage_values(
            source_frame=combined_source,
            labels=pd.concat([train_labels, contaminated_eval_labels], axis=0),
            feature_a=source_a,
            feature_b=source_b,
            clean_reference_labels=train_labels,
            clean_reference_frame=train_features,
        )

        leakage_feature_name = "leakage_indirect_signal"
        train_leakage = leaky_values.loc[train_features.index]
        contaminated_eval_leakage = leaky_values.loc[contaminated_eval_features.index]

        holdout_keys = _cluster_keys(
            clean_holdout_features,
            source_a,
            source_b,
            _quantile_edges(combined_source[source_a], bins=5),
            _quantile_edges(combined_source[source_b], bins=5),
        )
        clean_lookup = _smoothed_target_encoding(
            _cluster_keys(train_features, source_a, source_b, _quantile_edges(combined_source[source_a], bins=5), _quantile_edges(combined_source[source_b], bins=5)),
            train_labels,
            smoothing=8.0,
        )
        clean_holdout_leakage = _apply_lookup(holdout_keys, clean_lookup)

        metadata.update(
            {
                "leakage_feature_name": leakage_feature_name,
                "construction": construction_meta["construction"],
                "orig_variables": construction_meta["source_variables"],
                "statistic": construction_meta["statistic"],
            }
        )

    train_features[leakage_feature_name] = train_leakage
    contaminated_eval_features[leakage_feature_name] = contaminated_eval_leakage
    clean_holdout_features[leakage_feature_name] = clean_holdout_leakage

    metadata.update(
        {
            "contaminated_eval_contaminated": True,
            "clean_holdout_contaminated": False,
            "contamination_targets": {
                "train": True,
                "contaminated_eval": True,
                "clean_holdout": False,
            },
        }
    )

    return (
        {
            "train": {"features": train_features, "labels": train_labels},
            "contaminated_eval": {"features": contaminated_eval_features, "labels": contaminated_eval_labels},
            "clean_holdout": {"features": clean_holdout_features, "labels": clean_holdout_labels},
        },
        metadata,
    )


def _inject_spurious_correlation(
    train_split: Dict[str, pd.DataFrame | pd.Series],
    contaminated_eval_split: Dict[str, pd.DataFrame | pd.Series],
    clean_holdout_split: Dict[str, pd.DataFrame | pd.Series],
) -> Tuple[Dict[str, pd.DataFrame | pd.Series], Dict[str, Any]]:
    train_features = train_split["features"].copy()
    contaminated_eval_features = contaminated_eval_split["features"].copy()
    clean_holdout_features = clean_holdout_split["features"].copy()

    train_labels = train_split["labels"]
    contaminated_eval_labels = contaminated_eval_split["labels"]
    clean_holdout_labels = clean_holdout_split["labels"]

    rng = np.random.default_rng(RANDOM_STATE)
    source_frame = pd.concat([train_features, contaminated_eval_features], axis=0)
    numeric = source_frame.select_dtypes(include=[np.number])
    source_feature = numeric.var(axis=0).sort_values(ascending=False).index[0]
    group_edges = _quantile_edges(source_frame[source_feature], bins=4) if USE_GROUPS_FOR_SPURIOUS else np.array([-np.inf, np.inf])

    def build_groups(frame: pd.DataFrame) -> pd.Series:
        if USE_GROUPS_FOR_SPURIOUS:
            return _bin_series(frame[source_feature], group_edges)
        return pd.Series(0, index=frame.index)

    train_groups = build_groups(train_features)
    contaminated_eval_groups = build_groups(contaminated_eval_features)
    clean_holdout_groups = build_groups(clean_holdout_features)

    unique_groups = sorted(pd.Index(train_groups.unique()).tolist())
    group_biases = {group: float(bias) for group, bias in zip(unique_groups, np.linspace(-0.6, 0.6, num=len(unique_groups))) }

    def make_shortcut(labels: pd.Series, groups: pd.Series, mode: str) -> pd.Series:
        signal = 2.0 * labels.astype(float) - 1.0
        group_component = groups.map(group_biases).fillna(0.0).astype(float)
        if mode == "broken":
            # Truly broken: independent noise uncorrelated with label or groups. Correlation -> ~0.
            shortcut = rng.normal(0.0, 0.35, size=len(labels))
        elif mode == "inverted":
            # Inverted but not catastrophic: partial inversion with higher noise to soften the effect.
            inversion_strength = 0.55 * SPURIOUS_STRENGTH  # Reduced from full strength
            shortcut = 0.6 * group_component - inversion_strength * signal + rng.normal(0.0, 0.25, size=len(labels))
        else:
            # Train mode: strong signal, label-aligned with group structure.
            shortcut = 0.85 * group_component + SPURIOUS_STRENGTH * signal + rng.normal(0.0, 0.18, size=len(labels))
        return pd.Series(shortcut, index=labels.index)

    train_shortcut = make_shortcut(train_labels, train_groups, "train")
    contaminated_eval_shortcut = make_shortcut(contaminated_eval_labels, contaminated_eval_groups, "train")
    clean_holdout_shortcut = make_shortcut(clean_holdout_labels, clean_holdout_groups, SPURIOUS_MODE)

    feature_name = "domain_shortcut_signal"
    train_features[feature_name] = train_shortcut
    contaminated_eval_features[feature_name] = contaminated_eval_shortcut
    clean_holdout_features[feature_name] = clean_holdout_shortcut

    train_stats = _feature_proxy(train_shortcut, train_labels)
    contaminated_eval_stats = _feature_proxy(contaminated_eval_shortcut, contaminated_eval_labels)
    clean_holdout_stats = _feature_proxy(clean_holdout_shortcut, clean_holdout_labels)

    metadata = {
        "fault_type": "spurious_correlation",
        "injected": True,
        "spurious_mode": SPURIOUS_MODE,
        "feature_name": feature_name,
        "contaminated_splits": ["train", "contaminated_eval"],
        "contaminated_eval_contaminated": True,
        "clean_holdout_contaminated": False,
        "domain_logic": {
            "group_variable": source_feature if USE_GROUPS_FOR_SPURIOUS else "synthetic_constant_group",
            "group_definition": "quantile_bins over train+contaminated_eval for shortcut groups",
            "group_biases": group_biases,
        },
        "holdout_behavior": "correlation breaks in clean_holdout" if SPURIOUS_MODE == "broken" else "correlation flips in clean_holdout",
        "train_correlation_strength": float(train_stats["correlation"]),
        "contaminated_eval_correlation_strength": float(contaminated_eval_stats["correlation"]),
        "clean_holdout_correlation_strength": float(clean_holdout_stats["correlation"]),
        "train_mutual_information": float(train_stats["mutual_information"]),
        "contaminated_eval_mutual_information": float(contaminated_eval_stats["mutual_information"]),
        "clean_holdout_mutual_information": float(clean_holdout_stats["mutual_information"]),
    }

    return (
        {
            "train": {"features": train_features, "labels": train_labels},
            "contaminated_eval": {"features": contaminated_eval_features, "labels": contaminated_eval_labels},
            "clean_holdout": {"features": clean_holdout_features, "labels": clean_holdout_labels},
        },
        metadata,
    )


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
    mean_uncertainty = float(fault_metadata.get("mean_changed_uncertainty", 0.0))

    if eval_gap < 0.03:
        return "too_weak", "Training accuracy barely changes under label noise."
    if train_gap > 0.22 and contaminated_eval_metrics["accuracy"] < 0.80:
        return "too_strong", "Noise degrades the model too much for a stable comparison."
    if fault_metadata["noise_mode"] == "hard" and mean_uncertainty > 0.35:
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
    corr = abs(contaminated_stats["correlation"])
    corr_drop = abs(contaminated_stats["correlation"]) - abs(clean_stats["correlation"])

    if leakage_mode == "direct":
        if corr > 0.90 or contaminated_eval_metrics["accuracy"] > 0.98:
            return "too_trivial", "Direct leakage is nearly a label copy; too obvious for debugging."
        return "too_trivial", "Direct leakage is too straightforward to be interesting for multi-method comparison."
    
    # Indirect mode criteria
    if corr < 0.25 or eval_gain < 0.05:
        return "too_weak", "Indirect leakage signal is too weak to separate offline and clean views."
    if corr > 0.92 or eval_gain > 0.20:
        return "too_strong", "Indirect leakage is too dominant; leaves little room for debugging insights."
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
    holdout_negative_corr = clean_stats["correlation"]  # signed, not absolute

    if train_corr < 0.35 or contaminated_corr < 0.35:
        return "too_weak", "Shortcut feature is not attractive enough in the training view."
    
    if spurious_mode == "broken":
        # Truly broken: holdout correlation should be near 0
        if holdout_corr > 0.15:
            return "too_weak", "Holdout shortcut is not truly broken; residual correlation remains."
        if holdout_gap < 0.08:
            return "too_weak", "Distribution shift from shortcut breaking is too subtle."
        return "usable", "Shortcut breaks cleanly in holdout; domain shift is realistic."
    
    if spurious_mode == "inverted":
        # Inverted but not catastrophic: should see negative correlation but not total collapse
        if holdout_negative_corr > -0.15:
            return "too_weak", "Inversion in holdout is too weak."
        if holdout_gap > 0.25 or clean_holdout_metrics["accuracy"] < 0.65:
            return "too_strong", "Inversion effect is too brutal; model fails too badly."
        return "usable", "Inversion is clear but recoverable; realistic debugging scenario."
    
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
        if "mean_changed_uncertainty" in fault_metadata:
            print(f"Mean uncertainty of changed samples: {fault_metadata['mean_changed_uncertainty']:.4f}")
    elif fault_type == "data_leakage":
        feature_name = fault_metadata.get("leakage_feature_name")
        if feature_name:
            for split_name in ("contaminated_eval", "clean_holdout"):
                feature = splits[split_name]["features"][feature_name]
                labels = splits[split_name]["labels"]
                stats = _feature_proxy(feature, labels)
                diagnostics[split_name] = stats
                print(
                    f"{split_name} leakage stats - corr: {stats['correlation']:.4f}, "
                    f"mi: {stats['mutual_information']:.4f}, proxy: {stats['mean_gap']:.4f}"
                )
            quality, reason = _quality_leakage(
                contaminated_eval_metrics,
                clean_holdout_metrics,
                diagnostics["contaminated_eval"],
                diagnostics["clean_holdout"],
                fault_metadata.get("leakage_mode", "direct"),
            )
            print(f"Leakage assessment: {quality} - {reason}")
    elif fault_type == "spurious_correlation":
        feature_name = fault_metadata.get("feature_name")
        if feature_name:
            for split_name in ("train", "contaminated_eval", "clean_holdout"):
                feature = splits[split_name]["features"][feature_name]
                labels = splits[split_name]["labels"]
                stats = _feature_proxy(feature, labels)
                diagnostics[split_name] = stats
                print(
                    f"{split_name} spurious stats - corr: {stats['correlation']:.4f}, "
                    f"mi: {stats['mutual_information']:.4f}, proxy: {stats['mean_gap']:.4f}"
                )
            print(f"Holdout behavior: {fault_metadata.get('holdout_behavior')}")

    if diagnostics:
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
    if FAULT_TYPE == "label_noise":
        noisy_train_labels, metadata = _inject_label_noise(splits["train"]["features"], splits["train"]["labels"])
        injected_splits = {
            "train": {"features": splits["train"]["features"], "labels": noisy_train_labels},
            "contaminated_eval": splits["contaminated_eval"],
            "clean_holdout": splits["clean_holdout"],
        }
        return injected_splits, metadata

    if FAULT_TYPE == "data_leakage":
        return _inject_data_leakage(splits["train"], splits["contaminated_eval"], splits["clean_holdout"])

    if FAULT_TYPE == "spurious_correlation":
        return _inject_spurious_correlation(splits["train"], splits["contaminated_eval"], splits["clean_holdout"])

    metadata = {
        "fault_type": "none",
        "injected": False,
        "contaminated_splits": [],
        "contaminated_eval_contaminated": False,
        "clean_holdout_contaminated": False,
    }
    return splits, metadata


def main() -> Dict[str, Any]:
    """Run the full reproducible training and fault-injection workflow."""
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

    print(f"Active fault type: {FAULT_TYPE}")
    print("Fault metadata:")
    print(json.dumps(fault_metadata, indent=2, ensure_ascii=False))
    _print_metric_block("Train metrics:", train_metrics)
    _print_metric_block("Contaminated eval metrics:", contaminated_eval_metrics)
    _print_metric_block("Clean holdout metrics:", clean_holdout_metrics)

    print("Gap summary:")
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

    return {
        "fault_type": FAULT_TYPE,
        "fault_metadata": fault_metadata,
        "train_metrics": train_metrics,
        "contaminated_eval_metrics": contaminated_eval_metrics,
        "clean_holdout_metrics": clean_holdout_metrics,
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
