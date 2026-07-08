from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None


BASELINE = "baseline"
XAI = "xai_shap"
ORACLE = "oracle_true_fix"

WORKFLOWS = [BASELINE, XAI, ORACLE]

CASE_ORDER = [
    ("none", "none"),
    ("label_noise", "random"),
    ("label_noise", "hard"),
    ("data_leakage", "direct"),
    ("data_leakage", "indirect"),
    ("spurious_correlation", "broken"),
    ("spurious_correlation", "inverted"),
]


def case_label(fault_type: str, fault_mode: str) -> str:
    if fault_type == "none":
        return "none"
    
    pretty_faults = {
        "label_noise": "Label Noise",
        "data_leakage": "Data Leakage",
        "spurious_correlation": "Spurious Correlation",
    }

    fault = pretty_faults.get(fault_type, fault_type)
    return f"{fault} / {fault_mode}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_columns = {
        "fault_type",
        "fault_mode",
        "workflow",
        "true_feature",
        "top_candidate_feature",
        "oracle_target",
        "oracle_fix_applied",
        "repair_effect_quality",
        "repair_effect_reason",
        "top5_suspect_features",
        "top5_suspect_indices",
    }

    for column in df.columns:
        if column not in text_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def read_and_repair_experiments_csv(input_csv: Path, repaired_csv: Path) -> pd.DataFrame:
    """
    Reads experiments.csv.

    If the file has the known malformed format:
    - header has 83 columns
    - non-none rows have 84 fields because top5_suspect_* was appended later

    then this function repairs it by adding:
    - top5_suspect_indices
    - top5_suspect_features
    """

    try:
        df = pd.read_csv(input_csv)

        if "top5_suspect_indices" not in df.columns:
            df["top5_suspect_indices"] = ""
        if "top5_suspect_features" not in df.columns:
            df["top5_suspect_features"] = ""

        df = coerce_numeric_columns(df)
        df.to_csv(repaired_csv, index=False, encoding="utf-8")
        return df

    except pd.errors.ParserError:
        pass

    with input_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    if not rows:
        raise ValueError(f"CSV is empty: {input_csv}")

    original_header = rows[0]
    expected_len = len(original_header)

    fixed_header = list(original_header)
    if "top5_suspect_indices" not in fixed_header:
        fixed_header.append("top5_suspect_indices")
    if "top5_suspect_features" not in fixed_header:
        fixed_header.append("top5_suspect_features")

    fixed_rows: list[list[Any]] = []
    repaired_rows = 0

    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) == expected_len:
            fixed_rows.append(row + ["", ""])
            continue

        if len(row) == expected_len + 1:
            repaired_rows += 1

            extra = row[-1]
            base = row[:-1]

            fault_type = base[1] if len(base) > 1 else ""
            workflow = base[3] if len(base) > 3 else ""

            if workflow == ORACLE or extra == "":
                top5_indices = ""
                top5_features = ""
            elif fault_type == "label_noise":
                top5_indices = extra
                top5_features = ""
            else:
                top5_indices = ""
                top5_features = extra

            fixed_rows.append(base + [top5_indices, top5_features])
            continue

        raise ValueError(
            f"Unexpected field count in line {line_number}: "
            f"expected {expected_len} or {expected_len + 1}, got {len(row)}"
        )

    df = pd.DataFrame(fixed_rows, columns=fixed_header)
    df = coerce_numeric_columns(df)

    repaired_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(repaired_csv, index=False, encoding="utf-8")

    print(f"Repaired malformed rows: {repaired_rows}")
    print(f"Wrote repaired CSV: {repaired_csv}")

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["case"] = [
        case_label(fault_type, fault_mode)
        for fault_type, fault_mode in zip(df["fault_type"], df["fault_mode"])
    ]

    df["improvement_clean_holdout_accuracy"] = df.get("delta_clean_holdout_accuracy", np.nan)
    df["improvement_clean_holdout_balanced_accuracy"] = df.get("delta_clean_holdout_balanced_accuracy", np.nan)
    df["improvement_clean_holdout_f1"] = df.get("delta_clean_holdout_f1", np.nan)
    df["improvement_clean_holdout_roc_auc"] = df.get("delta_clean_holdout_roc_auc", np.nan)

    if "delta_clean_holdout_log_loss" in df.columns:
        df["improvement_clean_holdout_log_loss"] = -df["delta_clean_holdout_log_loss"]

    if "delta_clean_holdout_brier_score" in df.columns:
        df["improvement_clean_holdout_brier_score"] = -df["delta_clean_holdout_brier_score"]

    return df


def validate_dataset(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    expected_rows = 30 * 7 * 3
    actual_rows = len(df)

    checks.append(
        {
            "check": "row_count",
            "expected": expected_rows,
            "actual": actual_rows,
            "ok": actual_rows == expected_rows,
        }
    )

    counts = (
        df.groupby(["fault_type", "fault_mode", "workflow"], dropna=False)
        .size()
        .reset_index(name="rows")
    )

    counts.to_csv(output_dir / "workflow_counts.csv", index=False)

    for fault_type, fault_mode in CASE_ORDER:
        for workflow in WORKFLOWS:
            current = counts[
                (counts["fault_type"] == fault_type)
                & (counts["fault_mode"] == fault_mode)
                & (counts["workflow"] == workflow)
            ]

            actual = int(current["rows"].iloc[0]) if not current.empty else 0

            checks.append(
                {
                    "check": "case_workflow_count",
                    "case": case_label(fault_type, fault_mode),
                    "workflow": workflow,
                    "expected": 30,
                    "actual": actual,
                    "ok": actual == 30,
                }
            )

    validation = pd.DataFrame(checks)
    validation.to_csv(output_dir / "validation_checks.csv", index=False)
    return validation


def aggregate_workflow_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metrics = [
        "steps_to_detect",
        "mrr",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "precision_at_k",
        "recall_at_k",
        "runtime_sec",
        "improvement_clean_holdout_accuracy",
        "improvement_clean_holdout_balanced_accuracy",
        "improvement_clean_holdout_f1",
        "improvement_clean_holdout_roc_auc",
        "improvement_clean_holdout_log_loss",
        "improvement_clean_holdout_brier_score",
        "oracle_normalized_clean_holdout_accuracy",
        "oracle_normalized_clean_holdout_f1",
        "oracle_normalized_clean_holdout_log_loss",
        "oracle_normalized_clean_holdout_brier_score",
    ]

    metrics = [metric for metric in metrics if metric in df.columns]

    summary = (
        df.groupby(["fault_type", "fault_mode", "workflow"], dropna=False)[metrics]
        .agg(["count", "mean", "std", "median", "min", "max"])
    )

    summary.columns = ["_".join(column).strip() for column in summary.columns]
    summary = summary.reset_index()

    summary["case"] = [
        case_label(fault_type, fault_mode)
        for fault_type, fault_mode in zip(summary["fault_type"], summary["fault_mode"])
    ]

    summary.to_csv(output_dir / "workflow_summary_by_case.csv", index=False)
    return summary


def paired_arrays(case_df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    subset = case_df[case_df["workflow"].isin([BASELINE, XAI])]

    pivot = subset.pivot_table(
        index="seed",
        columns="workflow",
        values=metric,
        aggfunc="first",
    )

    if BASELINE not in pivot.columns or XAI not in pivot.columns:
        return np.array([]), np.array([])

    paired = pivot[[BASELINE, XAI]].dropna()

    return (
        paired[BASELINE].to_numpy(dtype=float),
        paired[XAI].to_numpy(dtype=float),
    )


def run_paired_tests(
    baseline_values: np.ndarray,
    xai_values: np.ndarray,
    higher_is_better: bool,
) -> dict[str, Any]:
    if len(baseline_values) == 0 or len(xai_values) == 0:
        return {
            "n_pairs": 0,
            "baseline_mean": np.nan,
            "baseline_std": np.nan,
            "baseline_median": np.nan,
            "xai_mean": np.nan,
            "xai_std": np.nan,
            "xai_median": np.nan,
            "mean_xai_minus_baseline": np.nan,
            "mean_improvement_by_xai": np.nan,
            "median_improvement_by_xai": np.nan,
            "wilcoxon_p": np.nan,
            "paired_ttest_p": np.nan,
            "cohen_dz": np.nan,
        }

    raw_delta = xai_values - baseline_values
    improvement = raw_delta if higher_is_better else -raw_delta

    if len(improvement) > 1 and np.nanstd(improvement, ddof=1) > 1e-12:
        cohen_dz = float(np.nanmean(improvement) / np.nanstd(improvement, ddof=1))
    else:
        cohen_dz = np.nan

    wilcoxon_p = np.nan
    paired_ttest_p = np.nan

    if stats is not None and len(improvement) >= 2:
        non_zero = improvement[~np.isclose(improvement, 0.0)]

        if len(non_zero) == 0:
            wilcoxon_p = 1.0
        else:
            try:
                wilcoxon_p = float(
                    stats.wilcoxon(
                        improvement,
                        zero_method="wilcox",
                        alternative="two-sided",
                    ).pvalue
                )
            except Exception:
                wilcoxon_p = np.nan

        try:
            paired_ttest_p = float(
                stats.ttest_rel(
                    xai_values,
                    baseline_values,
                    nan_policy="omit",
                ).pvalue
            )
        except Exception:
            paired_ttest_p = np.nan

    return {
        "n_pairs": int(len(improvement)),
        "baseline_mean": float(np.nanmean(baseline_values)),
        "baseline_std": float(np.nanstd(baseline_values, ddof=1)) if len(baseline_values) > 1 else np.nan,
        "baseline_median": float(np.nanmedian(baseline_values)),
        "xai_mean": float(np.nanmean(xai_values)),
        "xai_std": float(np.nanstd(xai_values, ddof=1)) if len(xai_values) > 1 else np.nan,
        "xai_median": float(np.nanmedian(xai_values)),
        "mean_xai_minus_baseline": float(np.nanmean(raw_delta)),
        "mean_improvement_by_xai": float(np.nanmean(improvement)),
        "median_improvement_by_xai": float(np.nanmedian(improvement)),
        "wilcoxon_p": wilcoxon_p,
        "paired_ttest_p": paired_ttest_p,
        "cohen_dz": cohen_dz,
    }


def build_paired_comparisons(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metric_specs = [
        ("steps_to_detect", False, "H1: fewer manual inspection steps is better"),
        ("mrr", True, "H2: higher localization quality is better"),
        ("hit_at_1", True, "H2: true cause in first candidate"),
        ("hit_at_5", True, "H2: true cause in top 5"),
        ("hit_at_10", True, "H2: true cause in top 10"),
        ("precision_at_k", True, "Label noise: higher top-k precision is better"),
        ("recall_at_k", True, "Label noise: higher top-k recall is better"),
        ("runtime_sec", False, "Runtime: lower runtime is better"),

        ("improvement_clean_holdout_accuracy", True, "H3: larger accuracy improvement is better"),
        ("improvement_clean_holdout_balanced_accuracy", True, "H3: larger balanced-accuracy improvement is better"),
        ("improvement_clean_holdout_f1", True, "H3: larger F1 improvement is better"),
        ("improvement_clean_holdout_roc_auc", True, "H3: larger ROC-AUC improvement is better"),
        ("improvement_clean_holdout_log_loss", True, "H3: larger log-loss improvement is better"),
        ("improvement_clean_holdout_brier_score", True, "H3: larger Brier improvement is better"),

        ("oracle_normalized_clean_holdout_accuracy", True, "H3: higher oracle-normalized accuracy effect is better"),
        ("oracle_normalized_clean_holdout_balanced_accuracy", True, "H3: higher oracle-normalized balanced-accuracy effect is better"),
        ("oracle_normalized_clean_holdout_f1", True, "H3: higher oracle-normalized F1 effect is better"),
        ("oracle_normalized_clean_holdout_roc_auc", True, "H3: higher oracle-normalized ROC-AUC effect is better"),
        ("oracle_normalized_clean_holdout_log_loss", True, "H3: higher oracle-normalized log-loss effect is better"),
        ("oracle_normalized_clean_holdout_brier_score", True, "H3: higher oracle-normalized Brier effect is better"),
    ]

    rows: list[dict[str, Any]] = []

    for fault_type, fault_mode in CASE_ORDER:
        case_df = df[
            (df["fault_type"] == fault_type)
            & (df["fault_mode"] == fault_mode)
        ]

        if case_df.empty:
            continue

        for metric, higher_is_better, interpretation in metric_specs:
            if metric not in df.columns:
                continue

            baseline_values, xai_values = paired_arrays(case_df, metric)

            result = run_paired_tests(
                baseline_values,
                xai_values,
                higher_is_better,
            )

            result.update(
                {
                    "fault_type": fault_type,
                    "fault_mode": fault_mode,
                    "case": case_label(fault_type, fault_mode),
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "interpretation": interpretation,
                }
            )

            rows.append(result)

    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(output_dir / "paired_baseline_vs_xai_tests.csv", index=False)
    return comparisons


def create_repair_quality_counts(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    counts = (
        df[df["workflow"].isin([BASELINE, XAI])]
        .groupby(
            ["fault_type", "fault_mode", "workflow", "repair_effect_quality"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    counts["case"] = [
        case_label(fault_type, fault_mode)
        for fault_type, fault_mode in zip(counts["fault_type"], counts["fault_mode"])
    ]

    counts.to_csv(output_dir / "repair_quality_counts.csv", index=False)
    return counts


def ordered_cases(df: pd.DataFrame) -> list[str]:
    existing = set(df["case"].dropna().unique().tolist())

    ordered = [
        case_label(fault_type, fault_mode)
        for fault_type, fault_mode in CASE_ORDER
        if case_label(fault_type, fault_mode) in existing
    ]

    rest = sorted(existing - set(ordered))
    return ordered + rest


def save_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    ylabel: str,
    include_none: bool = False,
) -> None:
    plot_df = df[df["workflow"].isin([BASELINE, XAI])].copy()

    if not include_none:
        plot_df = plot_df[plot_df["fault_type"] != "none"]

    plot_df = plot_df.dropna(subset=[metric])

    if plot_df.empty:
        return

    cases = ordered_cases(plot_df)
    workflows = [BASELINE, XAI]

    data = []
    positions = []
    labels = []

    position = 1

    for case in cases:
        for workflow in workflows:
            values = plot_df[
                (plot_df["case"] == case)
                & (plot_df["workflow"] == workflow)
            ][metric].dropna().to_numpy()

            data.append(values)
            positions.append(position)
            labels.append("B" if workflow == BASELINE else "X")
            position += 1

        position += 0.8

    plt.figure(figsize=(max(10, len(cases) * 1.8), 6))
    plt.boxplot(data, positions=positions, widths=0.6, showmeans=True, medianprops={"linewidth": 0}, meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6})

    plt.xticks(positions, labels)
    plt.xlabel("B = Baseline    |    X = XAI-SHAP")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    y_min, y_max = plt.ylim()
    text_y = y_min - 0.12 * (y_max - y_min)

    center_position = 1

    for case in cases:
        center = (center_position + center_position + 1) / 2
        plt.text(center, text_y, case, ha="center", va="top", fontsize=8)
        center_position += 2.8

    plt.ylim(y_min - 0.18 * (y_max - y_min), y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_hit10_barplot(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[
        (df["workflow"].isin([BASELINE, XAI]))
        & (df["fault_type"] != "none")
    ]

    if plot_df.empty or "hit_at_10" not in plot_df.columns:
        return

    aggregate = (
        plot_df.groupby(["case", "workflow"])["hit_at_10"]
        .mean()
        .reset_index()
    )

    cases = ordered_cases(aggregate)
    x = np.arange(len(cases))
    width = 0.35

    baseline_values = []
    xai_values = []

    for case in cases:
        baseline_row = aggregate[
            (aggregate["case"] == case)
            & (aggregate["workflow"] == BASELINE)
        ]

        xai_row = aggregate[
            (aggregate["case"] == case)
            & (aggregate["workflow"] == XAI)
        ]

        baseline_values.append(
            float(baseline_row["hit_at_10"].iloc[0])
            if not baseline_row.empty
            else np.nan
        )

        xai_values.append(
            float(xai_row["hit_at_10"].iloc[0])
            if not xai_row.empty
            else np.nan
        )

    plt.figure(figsize=(max(10, len(cases) * 1.5), 6))

    plt.bar(x - width / 2, baseline_values, width, label="Baseline")
    plt.bar(x + width / 2, xai_values, width, label="XAI SHAP")

    plt.xticks(x, cases, rotation=30, ha="right")
    plt.ylabel("Hit@10-Anteil")
    plt.ylim(0, 1.05)
    plt.title("Hit@10: wahre Fehlerursache unter den Top-10-Kandidaten")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()


def create_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    ensure_dir(plots_dir)

    save_boxplot(
        df,
        plots_dir / "mrr_boxplot.png",
        "mrr",
        "Mean Reciprocal Rank nach Fehlerklasse",
        "MRR",
    )

    save_boxplot(
        df,
        plots_dir / "steps_to_detect_boxplot.png",
        "steps_to_detect",
        "Steps-to-detect nach Fehlerklasse",
        "Steps-to-detect",
    )

    save_boxplot(
        df,
        plots_dir / "runtime_boxplot.png",
        "runtime_sec",
        "Laufzeit nach Fehlerklasse",
        "Laufzeit in Sekunden",
    )

    if "precision_at_k" in df.columns:
        label_noise_df = df[df["fault_type"] == "label_noise"]

        save_boxplot(
            label_noise_df,
            plots_dir / "label_noise_precision_at_k_boxplot.png",
            "precision_at_k",
            "Precision@k bei Label Noise",
            "Precision@k",
            include_none=True,
        )

    if "improvement_clean_holdout_accuracy" in df.columns:
        save_boxplot(
            df,
            plots_dir / "clean_holdout_accuracy_improvement_boxplot.png",
            "improvement_clean_holdout_accuracy",
            "Clean-Holdout-Accuracy-Verbesserung nach Fehlerklasse",
            "Accuracy-Verbesserung",
        )

    save_hit10_barplot(df, plots_dir / "hit_at_10_barplot.png")


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"

    if not np.isfinite(number):
        return "n/a"

    return f"{number:.4f}"


def fmt_de(value: Any, digits: int = 4) -> str:
    """Format numbers with German decimal comma for thesis tables."""
    try:
        number = float(value)
    except Exception:
        return "n/a"

    if not np.isfinite(number):
        return "n/a"

    return f"{number:.{digits}f}".replace(".", ",")


def fmt_p_de(value: Any) -> str:
    """Format p-values in a thesis-friendly way."""
    try:
        number = float(value)
    except Exception:
        return "n/a"

    if not np.isfinite(number):
        return "n/a"

    if number < 0.001:
        return "< 0,001"

    return f"{number:.4f}".replace(".", ",")


def thesis_digits_for_column(column: str) -> int:
    """Choose readable thesis precision per column."""
    if "Faktor" in column:
        return 2

    if "Steps" in column:
        return 2

    if "Laufzeit" in column or "Runtime" in column:
        return 2

    if "Effektstärke" in column or column.startswith("dz "):
        return 2

    if "MRR" in column or "Hit@" in column:
        return 3

    if "Precision" in column or "Recall" in column:
        return 3

    # Fix-impact values are often small, so keep four decimals.
    if "Fix-Impact" in column or "Oracle" in column:
        return 4

    return 3


def beautify_repair_quality(value: str) -> str:
    """Translate internal repair-quality labels for thesis tables."""
    mapping = {
        "repair_usable": "reparaturrelevant",
        "repair_weak": "schwach reparaturrelevant",
        "repair_too_weak": "zu schwach",
        "no_oracle_fix": "kein Oracle-Fix",
        "unknown": "unklar",
    }
    return mapping.get(str(value), str(value))


def beautify_metric_name(value: str) -> str:
    """Translate metric suffixes for appendix tables."""
    mapping = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "f1": "F1-Score",
        "roc_auc": "ROC-AUC",
        "log_loss": "Log-Loss",
        "brier_score": "Brier-Score",
    }
    return mapping.get(str(value), str(value))


def format_thesis_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply German labels and readable numeric formatting to a thesis table."""
    rename_columns = {
        "Eingesparte Steps durch XAI": "Δ Steps (Baseline - XAI)",
        "Runtime Baseline (s)": "Laufzeit Baseline (s)",
        "Runtime XAI (s)": "Laufzeit XAI (s)",
        "XAI/Baseline-Faktor": "Laufzeitfaktor XAI/Baseline",
        "dz Steps": "Effektstärke dz (Steps)",
        "dz MRR": "Effektstärke dz (MRR)",
        "dz Runtime": "Effektstärke dz (Laufzeit)",
        "dz Fix-Impact": "Effektstärke dz (Fix-Impact)",
        "Repair-Quality": "Oracle-Bewertung",
        "Oracle-normalisiert Baseline": "Oracle-normalisiert Baseline",
        "Oracle-normalisiert XAI": "Oracle-normalisiert XAI",
    }

    formatted = raw.rename(columns=rename_columns).copy()

    text_columns = {
        "Fehlerklasse",
        "Label-Noise-Modus",
        "Interpretation",
        "Oracle-Bewertung",
        "Metrik",
    }

    if "Oracle-Bewertung" in formatted.columns:
        formatted["Oracle-Bewertung"] = formatted["Oracle-Bewertung"].map(beautify_repair_quality)

    if "Metrik" in formatted.columns:
        formatted["Metrik"] = formatted["Metrik"].map(beautify_metric_name)

    for column in formatted.columns:
        if column in text_columns:
            continue

        if column.startswith("p "):
            formatted[column] = formatted[column].map(fmt_p_de)
        else:
            digits = thesis_digits_for_column(column)
            formatted[column] = formatted[column].map(lambda value: fmt_de(value, digits))

    return formatted


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Convert a DataFrame to a simple markdown table without extra dependencies."""
    columns = list(table.columns)

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in table.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def comparison_row(comparisons: pd.DataFrame, case: str, metric: str) -> pd.Series | None:
    rows = comparisons[
        (comparisons["case"] == case)
        & (comparisons["metric"] == metric)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def mcnemar_exact_pvalue_for_case(df: pd.DataFrame, case: str, metric: str) -> float:
    """Exact McNemar-style binomial p-value for paired binary metrics such as Hit@10.

    This is useful because Hit@10 is binary per seed.
    """
    if stats is None:
        return np.nan

    subset = df[
        (df["case"] == case)
        & (df["workflow"].isin([BASELINE, XAI]))
    ]

    if subset.empty or metric not in subset.columns:
        return np.nan

    pivot = subset.pivot_table(
        index="seed",
        columns="workflow",
        values=metric,
        aggfunc="first",
    )

    if BASELINE not in pivot.columns or XAI not in pivot.columns:
        return np.nan

    paired = pivot[[BASELINE, XAI]].dropna()

    if paired.empty:
        return np.nan

    baseline = (paired[BASELINE].astype(float) >= 0.5).astype(int)
    xai = (paired[XAI].astype(float) >= 0.5).astype(int)

    baseline_wrong_xai_right = int(((baseline == 0) & (xai == 1)).sum())
    baseline_right_xai_wrong = int(((baseline == 1) & (xai == 0)).sum())

    discordant = baseline_wrong_xai_right + baseline_right_xai_wrong

    if discordant == 0:
        return 1.0

    smaller = min(baseline_wrong_xai_right, baseline_right_xai_wrong)

    try:
        return float(
            stats.binomtest(
                smaller,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    except AttributeError:
        # Fallback for older SciPy versions.
        p_value = 2.0 * float(stats.binom.cdf(smaller, discordant, 0.5))
        return min(1.0, p_value)


def mode_or_na(series: pd.Series) -> str:
    values = series.dropna()

    if values.empty:
        return "n/a"

    mode_values = values.mode()

    if mode_values.empty:
        return "n/a"

    return str(mode_values.iloc[0])


def classify_fix_impact_interpretation(
    oracle_mean: float,
    baseline_mean: float,
    xai_mean: float,
    p_value: float,
) -> str:
    """Generate a conservative interpretation for the H3/Fix-Impact table."""
    if not np.isfinite(oracle_mean) or abs(oracle_mean) < 0.005:
        return "Oracle-Potenzial zu schwach"

    delta = xai_mean - baseline_mean

    if np.isfinite(p_value) and p_value < 0.05:
        if delta > 0:
            return "XAI höherer Fix-Impact"
        if delta < 0:
            return "Baseline höherer Fix-Impact"

    if abs(delta) < 0.005:
        return "kein klarer Unterschied"

    if delta > 0:
        return "XAI tendenziell höher"

    return "Baseline tendenziell höher"


def build_thesis_localization_table(
    df: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Create thesis-ready localization table for H1/H2."""
    rows: list[dict[str, Any]] = []

    for fault_type, fault_mode in CASE_ORDER:
        if fault_type == "none":
            continue

        case = case_label(fault_type, fault_mode)

        steps = comparison_row(comparisons, case, "steps_to_detect")
        mrr = comparison_row(comparisons, case, "mrr")
        hit10 = comparison_row(comparisons, case, "hit_at_10")

        if steps is None or mrr is None or hit10 is None:
            continue

        hit10_mcnemar_p = mcnemar_exact_pvalue_for_case(df, case, "hit_at_10")

        rows.append(
            {
                "Fehlerklasse": case,
                "Steps Baseline": steps["baseline_mean"],
                "Steps XAI": steps["xai_mean"],
                "Eingesparte Steps durch XAI": steps["mean_improvement_by_xai"],
                "MRR Baseline": mrr["baseline_mean"],
                "MRR XAI": mrr["xai_mean"],
                "Hit@10 Baseline": hit10["baseline_mean"],
                "Hit@10 XAI": hit10["xai_mean"],
                "p Steps": steps["wilcoxon_p"],
                "p MRR": mrr["wilcoxon_p"],
                "p Hit@10": hit10_mcnemar_p,
                "dz Steps": steps["cohen_dz"],
                "dz MRR": mrr["cohen_dz"],
            }
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "thesis_table_localization_raw.csv", index=False, encoding="utf-8")

    formatted = format_thesis_table(raw)

    markdown = dataframe_to_markdown(formatted)
    (output_dir / "thesis_table_localization.md").write_text(markdown, encoding="utf-8")

    return formatted


def build_thesis_label_noise_table(
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Create thesis-ready Precision@k/Recall@k table for Label Noise."""
    rows: list[dict[str, Any]] = []

    for fault_mode in ["random", "hard"]:
        case = case_label("label_noise", fault_mode)

        precision = comparison_row(comparisons, case, "precision_at_k")
        recall = comparison_row(comparisons, case, "recall_at_k")

        if precision is None or recall is None:
            continue

        precision_delta = float(precision["xai_mean"]) - float(precision["baseline_mean"])
        recall_delta = float(recall["xai_mean"]) - float(recall["baseline_mean"])

        if precision_delta > 0:
            interpretation = "XAI besser"
        elif precision_delta < 0:
            interpretation = "Baseline besser"
        else:
            interpretation = "kein Unterschied"

        rows.append(
            {
                "Label-Noise-Modus": fault_mode,
                "Precision@k Baseline": precision["baseline_mean"],
                "Precision@k XAI": precision["xai_mean"],
                "Δ Precision@k (XAI - Baseline)": precision_delta,
                "Recall@k Baseline": recall["baseline_mean"],
                "Recall@k XAI": recall["xai_mean"],
                "Δ Recall@k (XAI - Baseline)": recall_delta,
                "p Precision@k": precision["wilcoxon_p"],
                "p Recall@k": recall["wilcoxon_p"],
                "Interpretation": interpretation,
            }
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "thesis_table_label_noise_topk_raw.csv", index=False, encoding="utf-8")

    formatted = format_thesis_table(raw)

    markdown = dataframe_to_markdown(formatted)
    (output_dir / "thesis_table_label_noise_topk.md").write_text(markdown, encoding="utf-8")

    return formatted


def build_thesis_runtime_table(
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Create thesis-ready runtime table."""
    rows: list[dict[str, Any]] = []

    for fault_type, fault_mode in CASE_ORDER:
        if fault_type == "none":
            continue

        case = case_label(fault_type, fault_mode)
        runtime = comparison_row(comparisons, case, "runtime_sec")

        if runtime is None:
            continue

        baseline_mean = float(runtime["baseline_mean"])
        xai_mean = float(runtime["xai_mean"])

        if baseline_mean > 1e-12:
            ratio = xai_mean / baseline_mean
        else:
            ratio = np.nan

        if xai_mean > baseline_mean:
            interpretation = "XAI langsamer"
        elif xai_mean < baseline_mean:
            interpretation = "XAI schneller"
        else:
            interpretation = "kein Unterschied"

        rows.append(
            {
                "Fehlerklasse": case,
                "Runtime Baseline (s)": baseline_mean,
                "Runtime XAI (s)": xai_mean,
                "XAI/Baseline-Faktor": ratio,
                "p Runtime": runtime["wilcoxon_p"],
                "dz Runtime": runtime["cohen_dz"],
                "Interpretation": interpretation,
            }
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "thesis_table_runtime_raw.csv", index=False, encoding="utf-8")

    formatted = format_thesis_table(raw)

    markdown = dataframe_to_markdown(formatted)
    (output_dir / "thesis_table_runtime.md").write_text(markdown, encoding="utf-8")

    return formatted


def build_thesis_fix_impact_table(
    df: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
    metric_suffix: str = "accuracy",
) -> pd.DataFrame:
    """Create compact thesis-ready H3/Fix-Impact table for one primary metric.

    Default metric:
    - clean_holdout_accuracy

    Positive values always mean improvement because add_derived_columns()
    converts lower-is-better metrics such as log_loss and brier_score.
    """
    metric = f"improvement_clean_holdout_{metric_suffix}"
    normalized_metric = f"oracle_normalized_clean_holdout_{metric_suffix}"

    if metric not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for fault_type, fault_mode in CASE_ORDER:
        if fault_type == "none":
            continue

        case = case_label(fault_type, fault_mode)
        case_df = df[df["case"] == case]

        if case_df.empty:
            continue

        baseline_rows = case_df[case_df["workflow"] == BASELINE]
        xai_rows = case_df[case_df["workflow"] == XAI]
        oracle_rows = case_df[case_df["workflow"] == ORACLE]

        if baseline_rows.empty or xai_rows.empty or oracle_rows.empty:
            continue

        baseline_mean = float(baseline_rows[metric].mean())
        xai_mean = float(xai_rows[metric].mean())
        oracle_mean = float(oracle_rows[metric].mean())

        baseline_norm = (
            float(baseline_rows[normalized_metric].mean())
            if normalized_metric in baseline_rows.columns
            else np.nan
        )

        xai_norm = (
            float(xai_rows[normalized_metric].mean())
            if normalized_metric in xai_rows.columns
            else np.nan
        )

        comp = comparison_row(comparisons, case, metric)

        if comp is not None:
            p_value = float(comp["wilcoxon_p"])
            cohen_dz = float(comp["cohen_dz"])
        else:
            p_value = np.nan
            cohen_dz = np.nan

        repair_quality = mode_or_na(baseline_rows["repair_effect_quality"])

        interpretation = classify_fix_impact_interpretation(
            oracle_mean=oracle_mean,
            baseline_mean=baseline_mean,
            xai_mean=xai_mean,
            p_value=p_value,
        )

        rows.append(
            {
                "Fehlerklasse": case,
                "Oracle-Potenzial": oracle_mean,
                "Fix-Impact Baseline": baseline_mean,
                "Fix-Impact XAI": xai_mean,
                "Δ Fix-Impact (XAI - Baseline)": xai_mean - baseline_mean,
                "Oracle-normalisiert Baseline": baseline_norm,
                "Oracle-normalisiert XAI": xai_norm,
                "Repair-Quality": repair_quality,
                "p Fix-Impact": p_value,
                "dz Fix-Impact": cohen_dz,
                "Interpretation": interpretation,
            }
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(
        output_dir / f"thesis_table_h3_fix_impact_{metric_suffix}_raw.csv",
        index=False,
        encoding="utf-8",
    )

    formatted = format_thesis_table(raw)

    markdown = dataframe_to_markdown(formatted)
    (output_dir / f"thesis_table_h3_fix_impact_{metric_suffix}.md").write_text(
        markdown,
        encoding="utf-8",
    )

    return formatted


def build_thesis_fix_impact_long_table(
    df: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Create a longer appendix-style H3 table across multiple clean-holdout metrics."""
    metric_suffixes = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "brier_score",
    ]

    rows: list[dict[str, Any]] = []

    for fault_type, fault_mode in CASE_ORDER:
        if fault_type == "none":
            continue

        case = case_label(fault_type, fault_mode)
        case_df = df[df["case"] == case]

        if case_df.empty:
            continue

        for metric_suffix in metric_suffixes:
            metric = f"improvement_clean_holdout_{metric_suffix}"
            normalized_metric = f"oracle_normalized_clean_holdout_{metric_suffix}"

            if metric not in df.columns:
                continue

            baseline_rows = case_df[case_df["workflow"] == BASELINE]
            xai_rows = case_df[case_df["workflow"] == XAI]
            oracle_rows = case_df[case_df["workflow"] == ORACLE]

            baseline_mean = float(baseline_rows[metric].mean())
            xai_mean = float(xai_rows[metric].mean())
            oracle_mean = float(oracle_rows[metric].mean())

            baseline_norm = (
                float(baseline_rows[normalized_metric].mean())
                if normalized_metric in baseline_rows.columns
                else np.nan
            )

            xai_norm = (
                float(xai_rows[normalized_metric].mean())
                if normalized_metric in xai_rows.columns
                else np.nan
            )

            comp = comparison_row(comparisons, case, metric)

            rows.append(
                {
                    "Fehlerklasse": case,
                    "Metrik": metric_suffix,
                    "Oracle-Potenzial": oracle_mean,
                    "Fix-Impact Baseline": baseline_mean,
                    "Fix-Impact XAI": xai_mean,
                    "Δ Fix-Impact (XAI - Baseline)": xai_mean - baseline_mean,
                    "Oracle-normalisiert Baseline": baseline_norm,
                    "Oracle-normalisiert XAI": xai_norm,
                    "p Fix-Impact": float(comp["wilcoxon_p"]) if comp is not None else np.nan,
                    "dz Fix-Impact": float(comp["cohen_dz"]) if comp is not None else np.nan,
                }
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "thesis_table_h3_fix_impact_long_raw.csv", index=False, encoding="utf-8")

    formatted = format_thesis_table(raw)

    markdown = dataframe_to_markdown(formatted)
    (output_dir / "thesis_table_h3_fix_impact_long.md").write_text(markdown, encoding="utf-8")

    return formatted


def create_thesis_ready_tables(
    df: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create all thesis-ready tables from the experiment data."""
    localization = build_thesis_localization_table(df, comparisons, output_dir)
    label_noise = build_thesis_label_noise_table(comparisons, output_dir)
    runtime = build_thesis_runtime_table(comparisons, output_dir)
    fix_impact_accuracy = build_thesis_fix_impact_table(
        df,
        comparisons,
        output_dir,
        metric_suffix="accuracy",
    )
    fix_impact_long = build_thesis_fix_impact_long_table(df, comparisons, output_dir)

    sections: list[tuple[str, pd.DataFrame]] = [
        ("## Tabelle: Lokalisierungsergebnisse für H1/H2", localization),
        ("## Tabelle: Precision@k und Recall@k bei Label Noise", label_noise),
        ("## Tabelle: Laufzeitvergleich", runtime),
        ("## Tabelle: Fix-Impact für H3 anhand Clean-Holdout Accuracy", fix_impact_accuracy),
        ("## Anhangstabelle: Fix-Impact über alle Clean-Holdout-Metriken", fix_impact_long),
    ]

    lines: list[str] = ["# Thesis-ready Tabellen", ""]

    for title, table in sections:
        if table.empty:
            continue

        lines.append(title)
        lines.append("")
        lines.append(dataframe_to_markdown(table))
        lines.append("")

    (output_dir / "thesis_ready_tables.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_markdown_report(
    df: pd.DataFrame,
    validation: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_path = output_dir / "analysis_report.md"

    lines: list[str] = []

    lines.append("# Experiment Analysis Report")
    lines.append("")
    lines.append(f"Input rows: {len(df)}")
    lines.append("")

    lines.append("## Validation")
    lines.append("")

    failed_checks = validation[validation["ok"] != True]

    if failed_checks.empty:
        lines.append("All validation checks passed.")
    else:
        lines.append("Some validation checks failed. See `validation_checks.csv`.")

    lines.append("")

    lines.append("## Key Baseline vs. XAI Results")
    lines.append("")

    key_metrics = [
        "steps_to_detect",
        "mrr",
        "hit_at_10",
        "precision_at_k",
        "runtime_sec",
    ]

    for metric in key_metrics:
        rows = comparisons[comparisons["metric"] == metric]

        if rows.empty:
            continue

        lines.append(f"### {metric}")
        lines.append("")
        lines.append(
            "| Case | Baseline mean | XAI mean | Mean improvement by XAI | Wilcoxon p | Cohen dz |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")

        for _, row in rows.iterrows():
            lines.append(
                f"| {row['case']} | "
                f"{fmt(row['baseline_mean'])} | "
                f"{fmt(row['xai_mean'])} | "
                f"{fmt(row['mean_improvement_by_xai'])} | "
                f"{fmt(row['wilcoxon_p'])} | "
                f"{fmt(row['cohen_dz'])} |"
            )

        lines.append("")

    lines.append("## Generated files")
    lines.append("")
    lines.append("- `experiments_repaired.csv`")
    lines.append("- `validation_checks.csv`")
    lines.append("- `workflow_counts.csv`")
    lines.append("- `workflow_summary_by_case.csv`")
    lines.append("- `paired_baseline_vs_xai_tests.csv`")
    lines.append("- `repair_quality_counts.csv`")
    lines.append("- `plots/mrr_boxplot.png`")
    lines.append("- `plots/steps_to_detect_boxplot.png`")
    lines.append("- `plots/hit_at_10_barplot.png`")
    lines.append("- `plots/runtime_boxplot.png`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Baseline vs. XAI debugging experiment results."
    )

    parser.add_argument(
        "--input",
        default="results/itr_1/experiments.csv",
        help="Path to experiments.csv",
    )

    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: <input_parent>/analysis",
    )

    args = parser.parse_args()

    input_csv = Path(args.input)

    if args.outdir is None:
        output_dir = input_csv.parent / "analysis"
    else:
        output_dir = Path(args.outdir)

    plots_dir = output_dir / "plots"

    ensure_dir(output_dir)
    ensure_dir(plots_dir)

    repaired_csv = output_dir / "experiments_repaired.csv"

    df = read_and_repair_experiments_csv(input_csv, repaired_csv)
    df = add_derived_columns(df)

    df.to_csv(repaired_csv, index=False, encoding="utf-8")

    validation = validate_dataset(df, output_dir)
    aggregate_workflow_summary(df, output_dir)
    comparisons = build_paired_comparisons(df, output_dir)
    create_repair_quality_counts(df, output_dir)
    create_thesis_ready_tables(df, comparisons, output_dir)
    create_plots(df, plots_dir)
    write_markdown_report(df, validation, comparisons, output_dir)

    print("Analysis complete ✅")
    print(f"Input CSV: {input_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Repaired CSV: {repaired_csv}")
    print(f"Report: {output_dir / 'analysis_report.md'}")
    print(f"Statistical tests: {output_dir / 'paired_baseline_vs_xai_tests.csv'}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()