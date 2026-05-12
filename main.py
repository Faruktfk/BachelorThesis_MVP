from __future__ import annotations

import json
from typing import Any, Dict, Tuple
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


# ---------------------
# Central configuration
# ---------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200

# Fault selection (can be overridden via CLI)
FAULT_TYPE = "none"  # options: "none","label_noise","data_leakage","spurious_correlation"

# Label noise config
LABEL_NOISE_RATE = 0.10
LABEL_NOISE_MODE = "random"  # options: "random", "hard"
PROBE_N_ESTIMATORS = 30

# Data leakage config
LEAKAGE_MODE = "indirect"  # options: "direct", "indirect"
LEAKAGE_STRENGTH = 0.8  # controls how informative the leakage is (0..1)

# Spurious correlation config
SPURIOUS_MODE = "broken"  # options: "broken", "inverted"
SPURIOUS_STRENGTH = 0.9  # probability that spurious feature aligns with train label
USE_GROUPS_FOR_SPURIOUS = True

# Sanity checks
ENABLE_FAULT_SANITY_CHECKS = True


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer dataset as a DataFrame and Series.

    Returns features and labels. Using as_frame=True keeps feature names.
    """
    ds = load_breast_cancer(as_frame=True)
    return ds.data.copy(), ds.target.copy()


def split_dataset(features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with a central RANDOM_STATE."""
    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def _probe_uncertainty(train_X: pd.DataFrame, train_y: pd.Series) -> np.ndarray:
    """Train a small probe model to estimate sample uncertainty (for 'hard' label noise).

    We use predicted probability margins as an uncertainty proxy: small margin -> high uncertainty.
    """
    probe = RandomForestClassifier(n_estimators=PROBE_N_ESTIMATORS, random_state=RANDOM_STATE)
    probe.fit(train_X, train_y)
    probs = probe.predict_proba(train_X)
    # margin = top_prob - second_top_prob; lower margin = more uncertain
    top_two = np.sort(probs, axis=1)[:, -2:]
    margins = top_two[:, 1] - top_two[:, 0]
    return margins


def inject_label_noise(
    features: pd.DataFrame,
    labels: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Inject label noise into the TRAINING set only.

    Modes:
    - random: flip a random subset of labels (baseline).
    - hard: flip labels for samples where a probe model is most uncertain.

    The original labels are preserved externally; train_labels is returned modified.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    df = features.copy()
    y = labels.copy()

    # We'll flip labels only in the training portion later; here we return modified y_train when called after split.
    fault_metadata: Dict[str, Any] = {
        "fault_type": "label_noise",
        "injected": True,
        "noise_mode": LABEL_NOISE_MODE,
        "noise_rate": LABEL_NOISE_RATE,
        "changed_indices": [],
    }

    # This function will be applied after splitting: so expect y to be y_train
    if LABEL_NOISE_MODE == "random":
        n = max(1, int(round(len(y) * LABEL_NOISE_RATE)))
        changed = rng.choice(y.index.to_numpy(), size=n, replace=False)
    else:
        # hard mode: use a probe model on TRAIN to find low-margin samples
        margins = _probe_uncertainty(df, y)
        # lower margin -> more uncertain -> higher priority to flip
        n = max(1, int(round(len(y) * LABEL_NOISE_RATE)))
        order = np.argsort(margins)  # ascending margins -> uncertain first
        changed = df.index.to_numpy()[order[:n]]
    # flip labels for chosen indices; record them
    y_flipped = y.copy()
    y_flipped.loc[changed] = 1 - y_flipped.loc[changed]

    fault_metadata.update({
        "changed_count": int(len(changed)),
        "changed_indices": sorted(int(i) for i in list(changed)),
    })
    return df, y_flipped, fault_metadata


def construct_indirect_leakage_feature(features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """Construct an indirect leakage feature using global statistics before the split.

    Strategy:
    - choose a numeric feature (the one with highest variance)
    - bin it into quantiles across the WHOLE dataset
    - compute the target mean per bin across the WHOLE dataset (this is privileged information)
    - map each sample's bin to that global mean -> leaky statistic

    This simulates a pipeline mistake where global target statistics leak into feature engineering.
    """
    numeric = features.select_dtypes(include=[np.number])
    var = numeric.var(axis=0)
    chosen = var.idxmax()
    bins = pd.qcut(numeric[chosen], q=10, duplicates="drop")
    global_bin_means = labels.groupby(bins).mean()
    leaky = bins.map(global_bin_means).astype(float)

    metadata = {
        "leakage_mode": "indirect",
        "orig_variable": chosen,
        "statistic": "global_bin_mean",
    }
    return leaky, metadata


def construct_direct_leakage_feature(labels: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """Construct a direct but noisy copy of the label for each sample.

    The feature equals the true label with probability LEAKAGE_STRENGTH, otherwise a noisy value.
    This is less trivial than a perfect copy but still helpful to the model.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(labels)
    noisy = labels.astype(float).to_numpy(copy=True)
    flip_mask = rng.random(n) > LEAKAGE_STRENGTH
    noisy[flip_mask] = rng.integers(0, 2, size=flip_mask.sum())
    metadata = {
        "leakage_mode": "direct",
        "orig_variable": "label",
        "statistic": "noisy_copy",
        "strength": float(LEAKAGE_STRENGTH),
    }
    return pd.Series(noisy, index=labels.index), metadata


def inject_data_leakage_all(features: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a leakage feature on the WHOLE dataset (pre-split) to simulate privileged info.

    Returns modified features (with the leakage column) and metadata.
    """
    if LEAKAGE_MODE == "direct":
        leaky, md = construct_direct_leakage_feature(labels)
    else:
        leaky, md = construct_indirect_leakage_feature(features, labels)
    colname = f"leakage_{md['leakage_mode']}"
    features_with_leak = features.copy()
    features_with_leak[colname] = leaky
    md.update({"leakage_feature_name": colname})
    md.update({"injected": True})
    return features_with_leak, md


def inject_spurious_correlation(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    train_y: pd.Series,
    test_y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Inject a spurious feature that helps in training but fails in holdout.

    Modes:
    - broken: correlation disappears in test
    - inverted: correlation reverses in test

    Optionally uses a group/domain variable derived from an existing feature to make the shortcut realistic.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    tr = train_X.copy()
    te = test_X.copy()
    meta: Dict[str, Any] = {
        "fault_type": "spurious_correlation",
        "injected": True,
        "spurious_mode": SPURIOUS_MODE,
        "train_correlation_strength": float(SPURIOUS_STRENGTH),
    }

    # Define group variable from a numeric feature if requested
    group_col = None
    if USE_GROUPS_FOR_SPURIOUS:
        numeric = pd.concat([train_X, test_X]).select_dtypes(include=[np.number])
        if not numeric.columns.empty:
            var = numeric.var(axis=0)
            group_col = var.idxmax()
            # create groups by tertiles on the chosen feature (applied separately to train/test to simulate domain)
            tr_groups = pd.qcut(train_X[group_col], q=3, labels=False, duplicates="drop")
            te_groups = pd.qcut(test_X[group_col], q=3, labels=False, duplicates="drop")
        else:
            tr_groups = pd.Series(rng.integers(0, 3, size=len(train_X)), index=train_X.index)
            te_groups = pd.Series(rng.integers(0, 3, size=len(test_X)), index=test_X.index)
        meta["group_variable"] = group_col or "synthetic_random_group"
    else:
        tr_groups = pd.Series(0, index=train_X.index)
        te_groups = pd.Series(0, index=test_X.index)

    feature_name = "spurious_feature"

    # Build spurious feature for training: correlated with label within groups
    tr_spurious = np.zeros(len(train_X), dtype=float)
    for g in np.unique(tr_groups):
        idx = tr_groups[tr_groups == g].index
        # with probability SPURIOUS_STRENGTH, set equal to label, else random
        mask = rng.random(len(idx)) < SPURIOUS_STRENGTH
        sel_idx = idx.to_numpy()[mask]
        rest_idx = idx.to_numpy()[~mask]
        tr_spurious[np.isin(train_X.index.to_numpy(), sel_idx)] = train_y.loc[sel_idx]
        if len(rest_idx) > 0:
            tr_spurious[np.isin(train_X.index.to_numpy(), rest_idx)] = rng.integers(0, 2, size=len(rest_idx))

    # Build test spurious according to mode
    te_spurious = np.zeros(len(test_X), dtype=float)
    if SPURIOUS_MODE == "broken":
        te_spurious = rng.integers(0, 2, size=len(test_X)).astype(float)
    else:  # inverted
        te_spurious = 1.0 - test_y.to_numpy()
        # add some noise so it's not perfectly inverted
        flip_mask = rng.random(len(te_spurious)) > SPURIOUS_STRENGTH
        te_spurious[flip_mask] = rng.integers(0, 2, size=flip_mask.sum())

    tr[feature_name] = tr_spurious
    te[feature_name] = te_spurious

    meta.update({
        "feature_name": feature_name,
        "holdout_behavior": SPURIOUS_MODE,
    })
    return tr, te, meta


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Train a RandomForestClassifier on the provided training split."""
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


def evaluate_model(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Compute accuracy, F1, and ROC-AUC for a fitted classifier. Handles degenerate cases."""
    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)[:, 1]
        roc = float(roc_auc_score(y, probs))
    except Exception:
        roc = float("nan")

    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds)),
        "roc_auc": roc,
    }


def fault_sanity_check(
    fault_type: str,
    fault_metadata: Dict[str, Any],
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    injected_train_X: pd.DataFrame,
    injected_test_X: pd.DataFrame,
    injected_train_y: pd.Series,
    injected_test_y: pd.Series,
) -> None:
    """Provide quick diagnostics whether the injected fault is weak/strong/trivial.

    Prints generalization gap and specific fault diagnostics.
    """
    gap = train_metrics["accuracy"] - test_metrics["accuracy"]
    print("\n--- Fault Sanity Check ---")
    print(f"Generalization gap (train - test accuracy): {gap:.4f}")
    print(f"Train accuracy: {train_metrics['accuracy']:.4f}, Test accuracy: {test_metrics['accuracy']:.4f}")

    if fault_type == "label_noise":
        print(f"Label noise mode: {fault_metadata.get('noise_mode')}")
        print(f"Changed labels: {fault_metadata.get('changed_count')}")

    if fault_type == "data_leakage":
        lf = fault_metadata.get("leakage_feature_name")
        if lf and lf in injected_train_X.columns:
            corr_train = injected_train_X[lf].corr(injected_train_y)
            corr_test = injected_test_X[lf].corr(injected_test_y)
            print(f"Leakage feature: {lf}")
            print(f"Leakage correlation with label - train: {corr_train:.4f}, test: {corr_test:.4f}")

    if fault_type == "spurious_correlation":
        fn = fault_metadata.get("feature_name")
        if fn and fn in injected_train_X.columns:
            corr_train = injected_train_X[fn].corr(injected_train_y)
            corr_test = injected_test_X[fn].corr(injected_test_y)
            print(f"Spurious feature: {fn}")
            print(f"Correlation with label - train: {corr_train:.4f}, test: {corr_test:.4f}")
            print(f"Holdout behavior: {fault_metadata.get('holdout_behavior')}")
    print("--- End Sanity Check ---\n")


def main() -> Dict[str, Any]:
    # Load and (optionally) create leakage features pre-split
    features, labels = load_dataset()

    # If data leakage mode: construct leaky feature on whole dataset BEFORE split
    leakage_metadata: Dict[str, Any] = {}
    if FAULT_TYPE == "data_leakage":
        features, leakage_metadata = inject_data_leakage_all(features, labels)

    # Now split
    X_train, X_test, y_train, y_test = split_dataset(features, labels)

    fault_metadata: Dict[str, Any] = {"fault_type": "none", "injected": False}

    # Apply fault injection per type
    if FAULT_TYPE == "label_noise":
        # label noise operates on TRAIN only
        X_train, y_train, lm = inject_label_noise(X_train, y_train)
        fault_metadata = lm
    elif FAULT_TYPE == "data_leakage":
        # leakage already constructed before split; metadata from earlier
        fault_metadata = leakage_metadata
    elif FAULT_TYPE == "spurious_correlation":
        X_train, X_test, spm = inject_spurious_correlation(X_train, X_test, y_train, y_test)
        fault_metadata = spm

    # Train model on (possibly) injected training data
    model = train_model(X_train, y_train)

    # Evaluate on train and test
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    # Merge any leakage-specific metadata
    if FAULT_TYPE == "data_leakage":
        fault_metadata.setdefault("injected", True)

    # Print results
    print(f"Active fault type: {FAULT_TYPE}")
    print("Fault metadata:")
    print(json.dumps(fault_metadata, indent=2, ensure_ascii=False))
    print("Train metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    if ENABLE_FAULT_SANITY_CHECKS:
        fault_sanity_check(FAULT_TYPE, fault_metadata, train_metrics, test_metrics, X_train, X_test, y_train, y_test)

    return {
        "fault_type": FAULT_TYPE,
        "fault_metadata": fault_metadata,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fault-injection experiment")
    parser.add_argument("--fault", default=FAULT_TYPE, help="Fault type: none,label_noise,data_leakage,spurious_correlation")
    parser.add_argument("--mode", type=int, choices=(0, 1), default=0, help="Mode selector: 0 or 1 (mapped per fault)")
    args = parser.parse_args()

    # Map numeric mode to mode strings per fault type (concise maintenance logic)
    ft = args.fault
    m = int(args.mode)
    if ft == "label_noise":
        LABEL_NOISE_MODE = "random" if m == 0 else "hard"
    if ft == "data_leakage":
        LEAKAGE_MODE = "direct" if m == 0 else "indirect"
    if ft == "spurious_correlation":
        SPURIOUS_MODE = "broken" if m == 0 else "inverted"

    # set selected fault type
    FAULT_TYPE = ft

    print(f"Running with FAULT_TYPE={FAULT_TYPE}, mode={m}")
    main()
