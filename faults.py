"""Fault injection functions and utilities for ML pipeline testing.

This module provides all fault-injection logic and supporting helper functions.
It accepts a config dictionary to avoid global dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.cluster import mutual_info_score


def _probe_uncertainty(train_features: pd.DataFrame, train_labels: pd.Series, config: Dict[str, Any]) -> pd.Series:
    """Estimate uncertainty with a small probe model for hard label noise.

    Returns probability margins: smaller margin means more uncertain.
    """
    probe = RandomForestClassifier(
        n_estimators=config["PROBE_N_ESTIMATORS"],
        random_state=config["RANDOM_STATE"]
    )
    probe.fit(train_features, train_labels)
    probabilities = probe.predict_proba(train_features)
    top_two = np.sort(probabilities, axis=1)[:, -2:]
    margins = top_two[:, 1] - top_two[:, 0]
    return pd.Series(margins, index=train_features.index)


def _safe_corr(feature: pd.Series, labels: pd.Series) -> float:
    """Safely compute Pearson correlation, returning 0 if undefined."""
    if feature.nunique(dropna=True) < 2 or labels.nunique(dropna=True) < 2:
        return 0.0
    value = feature.corr(labels)
    return 0.0 if pd.isna(value) else float(value)


def _feature_proxy(feature: pd.Series, labels: pd.Series) -> Dict[str, float]:
    """Compute correlation, mutual information, and mean gap between positive/negative classes."""
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


def _quantile_edges(series: pd.Series, bins: int = 5) -> np.ndarray:
    """Compute quantile-based bin edges for a series."""
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
    """Bin a series using provided edges."""
    return pd.cut(series, bins=edges, include_lowest=True, labels=False).fillna(-1).astype(int)


def _smoothed_target_encoding(keys: pd.Series, labels: pd.Series, smoothing: float = 5.0) -> Dict[str, float]:
    """Compute smoothed target encoding: (count*mean + smoothing*global_mean) / (count + smoothing)."""
    global_mean = float(labels.mean())
    group_stats = labels.groupby(keys).agg(["mean", "count"])
    encoded: Dict[str, float] = {}
    for key, row in group_stats.iterrows():
        encoded[key] = float((row["mean"] * row["count"] + smoothing * global_mean) / (row["count"] + smoothing))
    encoded["__global__"] = global_mean
    return encoded


def _apply_lookup(keys: pd.Series, lookup: Dict[str, float]) -> pd.Series:
    """Apply a lookup dictionary to keys, falling back to global mean."""
    fallback = lookup["__global__"]
    return keys.map(lambda key: lookup.get(key, fallback)).astype(float)


def _select_indirect_leakage_features(
    frame: pd.DataFrame,
    labels: pd.Series,
    n_features: int = 3
) -> list[str]:
    """Choose medium-correlation features so indirect leakage is not just a relabeled strong feature."""
    numeric = frame.select_dtypes(include=[np.number])
    corr_scores = numeric.apply(lambda col: abs(_safe_corr(col, labels)), axis=0)
    var_scores = numeric.var(axis=0)

    low_q = float(corr_scores.quantile(0.25))
    high_q = float(corr_scores.quantile(0.65))
    candidate_mask = (corr_scores >= low_q) & (corr_scores <= high_q)
    candidates = corr_scores.index[candidate_mask].tolist()
    if len(candidates) < n_features:
        median_corr = float(corr_scores.median())
        ordered = sorted(corr_scores.index.tolist(), key=lambda f: abs(corr_scores[f] - median_corr))
        candidates = ordered[: max(n_features, 5)]

    ranked = sorted(candidates, key=lambda f: (-var_scores[f], corr_scores[f]))
    chosen = ranked[:n_features]
    if len(chosen) < n_features:
        fallback = [c for c in numeric.columns if c not in chosen]
        chosen.extend(fallback[: n_features - len(chosen)])
    return chosen[:n_features]


def _build_multi_feature_keys(frame: pd.DataFrame, feature_names: list[str], edges_map: Dict[str, np.ndarray]) -> pd.Series:
    """Build composite keys from multiple binned features."""
    parts = []
    for feature_name in feature_names:
        parts.append(_bin_series(frame[feature_name], edges_map[feature_name]).astype(str))
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key


def _build_indirect_leakage_signal(
    train_features: pd.DataFrame,
    contaminated_eval_features: pd.DataFrame,
    clean_holdout_features: pd.DataFrame,
    train_labels: pd.Series,
    contaminated_eval_labels: pd.Series,
    config: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """Create a stronger but still non-trivial indirect leakage signal.

    Strategy:
    - choose medium-correlation raw features instead of the strongest label correlates
    - build a higher-cardinality multi-feature bucket key
    - encode the target on train+contaminated_eval (leaky offline world)
    - encode the clean holdout only with train statistics, heavy smoothing and shrinkage

    This makes contaminated_eval optimistic while keeping clean_holdout notably weaker,
    without degenerating into a direct label copy.
    """
    rng = np.random.default_rng(config["RANDOM_STATE"])
    contaminated_source = pd.concat([train_features, contaminated_eval_features], axis=0)
    contaminated_labels = pd.concat([train_labels, contaminated_eval_labels], axis=0)

    feature_names = _select_indirect_leakage_features(contaminated_source, contaminated_labels, n_features=3)
    edges_map = {
        feature_name: _quantile_edges(contaminated_source[feature_name], bins=config["INDIRECT_LEAKAGE_BINS"])
        for feature_name in feature_names
    }

    contaminated_keys = _build_multi_feature_keys(contaminated_source, feature_names, edges_map)
    train_keys = _build_multi_feature_keys(train_features, feature_names, edges_map)
    holdout_keys = _build_multi_feature_keys(clean_holdout_features, feature_names, edges_map)

    contaminated_lookup = _smoothed_target_encoding(
        contaminated_keys,
        contaminated_labels,
        smoothing=config["INDIRECT_LEAKAGE_CONTAM_SMOOTHING"],
    )
    clean_lookup = _smoothed_target_encoding(
        train_keys,
        train_labels,
        smoothing=config["INDIRECT_LEAKAGE_CLEAN_SMOOTHING"],
    )

    contaminated_signal = _apply_lookup(contaminated_keys, contaminated_lookup)
    contaminated_signal = pd.Series(
        np.clip(
            contaminated_signal.to_numpy() + rng.normal(0.0, config["INDIRECT_LEAKAGE_NOISE_STD"], size=len(contaminated_signal)),
            0.0,
            1.0,
        ),
        index=contaminated_signal.index,
    )

    global_mean = float(train_labels.mean())
    raw_holdout_signal = _apply_lookup(holdout_keys, clean_lookup)
    holdout_signal = pd.Series(
        np.clip(
            (1.0 - config["INDIRECT_LEAKAGE_HOLDOUT_SHRINK"]) * raw_holdout_signal.to_numpy()
            + config["INDIRECT_LEAKAGE_HOLDOUT_SHRINK"] * global_mean
            + rng.normal(0.0, config["INDIRECT_LEAKAGE_NOISE_STD"], size=len(raw_holdout_signal)),
            0.0,
            1.0,
        ),
        index=raw_holdout_signal.index,
    )

    train_signal = contaminated_signal.loc[train_features.index]
    eval_signal = contaminated_signal.loc[contaminated_eval_features.index]

    metadata = {
        "construction": "high-cardinality multi-feature target encoding with contaminated offline lookup and train-only clean fallback",
        "orig_variables": feature_names,
        "statistic": "multi_bucket_target_encoding",
        "leakage_strength_rationale": (
            "Contaminated view uses train+offline labels with low smoothing; clean holdout uses train-only encoding "
            "with heavy shrinkage to the global mean so offline optimism appears without a direct label copy."
        ),
    }
    return train_signal, eval_signal, holdout_signal, metadata


def _inject_label_noise(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    config: Dict[str, Any],
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Corrupt only training labels. Hard mode flips the most uncertain samples."""
    rng = np.random.default_rng(config["RANDOM_STATE"])
    noisy_labels = train_labels.copy()
    noise_count = max(1, int(round(len(noisy_labels) * config["LABEL_NOISE_RATE"])))

    if config["LABEL_NOISE_MODE"] == "hard":
        margins = _probe_uncertainty(train_features, noisy_labels, config)
        changed_indices = margins.nsmallest(noise_count).index
        changed_margins = {int(index): float(margins.loc[index]) for index in changed_indices}
    else:
        changed_indices = pd.Index(rng.choice(noisy_labels.index.to_numpy(), size=noise_count, replace=False))
        changed_margins = {}

    original_labels = {int(index): int(noisy_labels.loc[index]) for index in changed_indices}
    noisy_labels.loc[changed_indices] = 1 - noisy_labels.loc[changed_indices]

    metadata: Dict[str, Any] = {
        "fault_type": "label_noise",
        "injected": True,
        "noise_mode": config["LABEL_NOISE_MODE"],
        "noise_rate": config["LABEL_NOISE_RATE"],
        "train_count": int(len(train_labels)),
        "changed_count": int(len(changed_indices)),
        "changed_indices": [int(index) for index in changed_indices],
        "original_labels_by_index": original_labels,
        "contaminated_splits": ["train"],
        "contaminated_eval_contaminated": False,
        "clean_holdout_contaminated": False,
        "injection_description": "Training labels are corrupted; evaluation labels stay clean.",
    }

    if changed_margins:
        metadata["changed_margin_scores"] = changed_margins
        metadata["mean_changed_margin"] = float(np.mean(list(changed_margins.values())))

    return noisy_labels, metadata


def _direct_leakage_values(
    labels: pd.Series,
    strength: float,
    config: Dict[str, Any],
    noisy_baseline: float | None = None,
) -> pd.Series:
    """Create direct leakage: noisy label copy."""
    rng = np.random.default_rng(config["RANDOM_STATE"])
    baseline = float(labels.mean()) if noisy_baseline is None else float(noisy_baseline)
    signal = labels.astype(float).to_numpy(copy=True)
    noise = rng.normal(loc=baseline, scale=max(0.08, (1.0 - strength) * 0.35), size=len(labels))
    values = strength * signal + (1.0 - strength) * noise
    return pd.Series(np.clip(values, 0.0, 1.0), index=labels.index)


def _inject_data_leakage(
    train_split: Dict[str, pd.DataFrame | pd.Series],
    contaminated_eval_split: Dict[str, pd.DataFrame | pd.Series],
    clean_holdout_split: Dict[str, pd.DataFrame | pd.Series],
    config: Dict[str, Any],
) -> Tuple[Dict[str, pd.DataFrame | pd.Series], Dict[str, Any]]:
    """Inject data leakage: direct or indirect leakage signal."""
    train_features = train_split["features"].copy()
    contaminated_eval_features = contaminated_eval_split["features"].copy()
    clean_holdout_features = clean_holdout_split["features"].copy()

    train_labels = train_split["labels"]
    contaminated_eval_labels = contaminated_eval_split["labels"]
    clean_holdout_labels = clean_holdout_split["labels"]

    metadata: Dict[str, Any] = {
        "fault_type": "data_leakage",
        "injected": True,
        "leakage_mode": config["LEAKAGE_MODE"],
        "contaminated_splits": ["train", "contaminated_eval"],
        "contaminated_eval_contaminated": True,
        "clean_holdout_contaminated": False,
    }

    if config["LEAKAGE_MODE"] == "direct":
        leakage_feature_name = "leakage_direct_signal"
        train_leakage = _direct_leakage_values(train_labels, config["LEAKAGE_STRENGTH"], config)
        contaminated_eval_leakage = _direct_leakage_values(contaminated_eval_labels, config["LEAKAGE_STRENGTH"], config)
        clean_holdout_leakage = pd.Series(
            np.random.default_rng(config["RANDOM_STATE"]).normal(
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
        train_leakage, contaminated_eval_leakage, clean_holdout_leakage, construction_meta = _build_indirect_leakage_signal(
            train_features,
            contaminated_eval_features,
            clean_holdout_features,
            train_labels,
            contaminated_eval_labels,
            config,
        )

        leakage_feature_name = "leakage_indirect_signal"
        metadata.update(
            {
                "leakage_feature_name": leakage_feature_name,
                "construction": construction_meta["construction"],
                "orig_variables": construction_meta["orig_variables"],
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
    config: Dict[str, Any],
) -> Tuple[Dict[str, pd.DataFrame | pd.Series], Dict[str, Any]]:
    """Inject spurious correlation: broken or inverted shortcut in holdout."""
    train_features = train_split["features"].copy()
    contaminated_eval_features = contaminated_eval_split["features"].copy()
    clean_holdout_features = clean_holdout_split["features"].copy()

    train_labels = train_split["labels"]
    contaminated_eval_labels = contaminated_eval_split["labels"]
    clean_holdout_labels = clean_holdout_split["labels"]

    rng = np.random.default_rng(config["RANDOM_STATE"])
    source_frame = pd.concat([train_features, contaminated_eval_features], axis=0)
    numeric = source_frame.select_dtypes(include=[np.number])
    source_feature = numeric.var(axis=0).sort_values(ascending=False).index[0]
    group_edges = (
        _quantile_edges(source_frame[source_feature], bins=4)
        if config["USE_GROUPS_FOR_SPURIOUS"]
        else np.array([-np.inf, np.inf])
    )

    def build_groups(frame: pd.DataFrame) -> pd.Series:
        if config["USE_GROUPS_FOR_SPURIOUS"]:
            return _bin_series(frame[source_feature], group_edges)
        return pd.Series(0, index=frame.index)

    train_groups = build_groups(train_features)
    contaminated_eval_groups = build_groups(contaminated_eval_features)
    clean_holdout_groups = build_groups(clean_holdout_features)

    unique_groups = sorted(pd.Index(train_groups.unique()).tolist())
    group_biases = {
        group: float(bias)
        for group, bias in zip(unique_groups, np.linspace(-0.6, 0.6, num=len(unique_groups)))
    }

    def make_shortcut(labels: pd.Series, groups: pd.Series, mode: str) -> pd.Series:
        signal = 2.0 * labels.astype(float) - 1.0
        group_component = groups.map(group_biases).fillna(0.0).astype(float)
        if mode == "broken":
            # Truly broken: independent noise uncorrelated with label or groups. Correlation -> ~0.
            shortcut = rng.normal(0.0, 0.35, size=len(labels))
        elif mode == "inverted":
            # Inverted but not catastrophic: partial inversion with higher noise to soften the effect.
            inversion_strength = config["INVERTED_SIGNAL_WEIGHT"] * config["SPURIOUS_STRENGTH"]
            shortcut = (
                config["INVERTED_GROUP_WEIGHT"] * group_component
                - inversion_strength * signal
                + rng.normal(0.0, config["INVERTED_NOISE_STD"], size=len(labels))
            )
        else:
            # Train mode: strong signal, label-aligned with group structure.
            shortcut = (
                0.85 * group_component
                + config["SPURIOUS_STRENGTH"] * signal
                + rng.normal(0.0, 0.18, size=len(labels))
            )
        return pd.Series(shortcut, index=labels.index)

    train_shortcut = make_shortcut(train_labels, train_groups, "train")
    contaminated_eval_shortcut = make_shortcut(contaminated_eval_labels, contaminated_eval_groups, "train")
    clean_holdout_shortcut = make_shortcut(clean_holdout_labels, clean_holdout_groups, config["SPURIOUS_MODE"])

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
        "spurious_mode": config["SPURIOUS_MODE"],
        "feature_name": feature_name,
        "contaminated_splits": ["train", "contaminated_eval"],
        "contaminated_eval_contaminated": True,
        "clean_holdout_contaminated": False,
        "domain_logic": {
            "group_variable": source_feature if config["USE_GROUPS_FOR_SPURIOUS"] else "synthetic_constant_group",
            "group_definition": "quantile_bins over train+contaminated_eval for shortcut groups",
            "group_biases": group_biases,
        },
        "holdout_behavior": "correlation breaks in clean_holdout" if config["SPURIOUS_MODE"] == "broken" else "correlation flips in clean_holdout",
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
