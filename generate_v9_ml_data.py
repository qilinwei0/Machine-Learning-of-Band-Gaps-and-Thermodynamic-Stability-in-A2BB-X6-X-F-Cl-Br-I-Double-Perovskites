#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the machine-learning data tables used by final_complete_corrected_package_v9.

The script intentionally treats the supplied ``split_formula_group`` column as
authoritative.  It never creates a random row-wise train/test split, so
polymorphs sharing the same formula cannot leak across evaluation partitions.

Outputs
-------
``data/raw``
    Copies of the three task CSV files used by the analysis.
``data/tables``
    Model comparison, selected-model metrics, held-out predictions, feature
    ablation, grouped cross-validation, bootstrap confidence intervals,
    permutation importance, conformal predictions, and candidate screening.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REGRESSION_COLUMNS = [
    "task", "model", "split", "MAE", "RMSE", "R2", "ROC_AUC",
    "PR_AUC", "F1", "Balanced_Accuracy", "MCC", "Precision",
    "Recall", "threshold",
]

TASK_FILES = {
    "bandgap": "A2BBX6_task_bandgap_regression.csv",
    "hull": "A2BBX6_task_hull_regression.csv",
    "stability": "A2BBX6_task_stability_classification_005.csv",
}


@dataclass(frozen=True)
class TaskSpec:
    key: str
    display: str
    display_zh: str
    filename: str
    target: str
    kind: str
    unit: str


TASKS = {
    "bandgap": TaskSpec(
        "bandgap", "Bandgap regression", "带隙回归",
        TASK_FILES["bandgap"], "target_band_gap_eV", "regression", "eV",
    ),
    "hull": TaskSpec(
        "hull", "Energy-above-hull regression", "凸包能回归",
        TASK_FILES["hull"], "target_energy_above_hull_mp", "regression", "eV/atom",
    ),
    "stability": TaskSpec(
        "stability", "Stability classification", "稳定性分类",
        TASK_FILES["stability"], "target_stable_within_0_05_eV_atom",
        "classification", "class",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final_complete_corrected_package_v9/data from A2BBX6 task CSVs."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("."),
        help="Directory containing the three A2BBX6_task_*.csv files.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("final_complete_corrected_package_v9"),
        help="Package root under which data/raw and data/tables are written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Miscoverage rate for split conformal prediction.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast smoke-test mode: fewer trees/repeats; output schemas are unchanged.",
    )
    return parser.parse_args()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, columns: Iterable[str], filename: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{filename}: missing required columns: {missing}")


def load_task(input_dir: Path, spec: TaskSpec) -> tuple[pd.DataFrame, list[str]]:
    path = input_dir / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    require_columns(
        df,
        [
            "mpid", "formula", "canonical_formula", "A_element", "B1_element",
            "B2_element", "X_element", "split_formula_group", spec.target,
        ],
        spec.filename,
    )
    features = [c for c in df.columns if c.startswith("feat_")]
    if not features:
        raise ValueError(f"{spec.filename}: no feat_ columns found")
    if df[spec.target].isna().any():
        raise ValueError(f"{spec.filename}: target contains missing values")
    invalid_splits = sorted(set(df["split_formula_group"].dropna()) - {"train", "validation", "test"})
    if invalid_splits:
        raise ValueError(f"{spec.filename}: invalid split labels: {invalid_splits}")
    for split in ("train", "validation", "test"):
        if not df["split_formula_group"].eq(split).any():
            raise ValueError(f"{spec.filename}: empty {split} split")
    # Coerce features once and fail on unexpected non-numeric strings.
    numeric = df[features].apply(pd.to_numeric, errors="coerce")
    unexpected = df[features].notna() & numeric.isna()
    if unexpected.any().any():
        bad = unexpected.sum().sort_values(ascending=False)
        raise ValueError(
            f"{spec.filename}: non-numeric feature values found in "
            f"{bad[bad.gt(0)].index[:10].tolist()}"
        )
    df.loc[:, features] = numeric
    if spec.kind == "classification":
        classes = sorted(pd.unique(df[spec.target]))
        if classes != [0, 1] and classes != [False, True]:
            raise ValueError(f"{spec.filename}: classification target must be binary, got {classes}")
        df[spec.target] = df[spec.target].astype(int)
    return df, features


def assert_no_formula_overlap(df: pd.DataFrame, filename: str) -> None:
    split_groups = {
        split: set(df.loc[df["split_formula_group"].eq(split), "formula"].astype(str))
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train-validation": split_groups["train"] & split_groups["validation"],
        "train-test": split_groups["train"] & split_groups["test"],
        "validation-test": split_groups["validation"] & split_groups["test"],
    }
    bad = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
    if bad:
        raise ValueError(f"{filename}: formula-group leakage detected: {bad}")


def tree_preprocessor() -> SimpleImputer:
    return SimpleImputer(strategy="median", keep_empty_features=True)


def scaled_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])


def make_regression_models(seed: int, n_jobs: int, quick: bool) -> dict[str, BaseEstimator]:
    trees = 120 if quick else 500
    hist_iter = 120 if quick else 500
    return {
        "Mean baseline": Pipeline([
            ("imputer", tree_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "Ridge": Pipeline([
            ("preprocess", scaled_preprocessor()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Extra Trees": Pipeline([
            ("imputer", tree_preprocessor()),
            ("model", ExtraTreesRegressor(
                n_estimators=trees, max_features=0.80, min_samples_leaf=1,
                random_state=seed, n_jobs=n_jobs,
            )),
        ]),
        "HistGB": Pipeline([
            ("imputer", tree_preprocessor()),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.03, max_iter=hist_iter, max_leaf_nodes=31,
                min_samples_leaf=20, l2_regularization=0.0,
                early_stopping=False, random_state=seed,
            )),
        ]),
    }


def make_classification_models(seed: int, n_jobs: int, quick: bool) -> dict[str, BaseEstimator]:
    trees = 120 if quick else 500
    hist_iter = 120 if quick else 500
    return {
        "Majority baseline": Pipeline([
            ("imputer", tree_preprocessor()),
            ("model", DummyClassifier(strategy="prior")),
        ]),
        "Logistic regression": Pipeline([
            ("preprocess", scaled_preprocessor()),
            ("model", LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=5000,
                solver="liblinear", random_state=seed,
            )),
        ]),
        "Extra Trees": Pipeline([
            ("imputer", tree_preprocessor()),
            ("model", ExtraTreesClassifier(
                n_estimators=trees, max_features=0.80, min_samples_leaf=1,
                class_weight="balanced", random_state=seed, n_jobs=n_jobs,
            )),
        ]),
        "HistGB": Pipeline([
            ("imputer", tree_preprocessor()),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.03, max_iter=hist_iter, max_leaf_nodes=31,
                min_samples_leaf=20, l2_regularization=0.0,
                early_stopping=False, random_state=seed,
            )),
        ]),
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def classification_metrics(
    y_true: Sequence[int], probabilities: Sequence[float], threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, predicted),
        "MCC": matthews_corrcoef(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "threshold": float(threshold),
    }


def choose_threshold(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    """Tune a binary threshold on validation data by MCC, with stable tie breaks."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    grid = np.linspace(0.05, 0.95, 181)
    ranked = []
    for threshold in grid:
        pred = (probabilities >= threshold).astype(int)
        ranked.append((matthews_corrcoef(y_true, pred), -abs(threshold - 0.5), -threshold, threshold))
    return float(max(ranked)[-1])


def empty_metric_row(task: str, model: str, split: str) -> dict[str, object]:
    return {c: np.nan for c in REGRESSION_COLUMNS} | {
        "task": task, "model": model, "split": split,
    }


def masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {name: df["split_formula_group"].eq(name) for name in ("train", "validation", "test")}


def fit_model_comparison(
    spec: TaskSpec,
    df: pd.DataFrame,
    features: list[str],
    seed: int,
    n_jobs: int,
    quick: bool,
) -> tuple[list[dict[str, object]], dict[str, BaseEstimator], dict[str, float]]:
    split = masks(df)
    X_train, y_train = df.loc[split["train"], features], df.loc[split["train"], spec.target]
    candidates = (
        make_regression_models(seed, n_jobs, quick)
        if spec.kind == "regression"
        else make_classification_models(seed, n_jobs, quick)
    )
    rows: list[dict[str, object]] = []
    fitted: dict[str, BaseEstimator] = {}
    thresholds: dict[str, float] = {}
    for name, model in candidates.items():
        fitted[name] = clone(model).fit(X_train, y_train)
        if spec.kind == "regression":
            for part in ("validation", "test"):
                pred = fitted[name].predict(df.loc[split[part], features])
                row = empty_metric_row(spec.display, name, part)
                row.update(regression_metrics(df.loc[split[part], spec.target], pred))
                rows.append(row)
        else:
            val_prob = fitted[name].predict_proba(df.loc[split["validation"], features])[:, 1]
            threshold = 0.50 if name == "Majority baseline" else choose_threshold(
                df.loc[split["validation"], spec.target], val_prob
            )
            thresholds[name] = threshold
            for part in ("validation", "test"):
                prob = fitted[name].predict_proba(df.loc[split[part], features])[:, 1]
                row = empty_metric_row(spec.display, name, part)
                row.update(classification_metrics(df.loc[split[part], spec.target], prob, threshold))
                rows.append(row)
    return rows, fitted, thresholds


def selected_model_name(spec: TaskSpec) -> str:
    return {"bandgap": "Extra Trees", "hull": "HistGB", "stability": "HistGB"}[spec.key]


def selected_model_long_name(spec: TaskSpec) -> str:
    return {
        "bandgap": "Extra Trees",
        "hull": "HistGradientBoosting",
        "stability": "HistGradientBoosting",
    }[spec.key]


def prediction_metadata(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    cols = ["mpid", "formula", "canonical_formula", "A_element", "B1_element", "B2_element", "X_element"]
    out = df.loc[mask, cols].copy()
    return out.rename(columns={
        "A_element": "A", "B1_element": "B1", "B2_element": "B2", "X_element": "X"
    }).reset_index(drop=True)


def data_quality_rows(spec: TaskSpec, df: pd.DataFrame, features: list[str]) -> tuple[dict, list[dict]]:
    split = masks(df)
    train = df.loc[split["train"], features]
    missing_rates = df[features].isna().mean()
    groups = {
        name: set(df.loc[split[name], "formula"].astype(str))
        for name in ("train", "validation", "test")
    }
    target = df[spec.target]
    q = target.quantile([0.25, 0.50, 0.75])
    row = {
        "task": spec.display,
        "samples": len(df),
        "features": len(features),
        "train": int(split["train"].sum()),
        "validation": int(split["validation"].sum()),
        "test": int(split["test"].sum()),
        "features_with_missing": int(missing_rates.gt(0).sum()),
        "max_missing_rate": float(missing_rates.max()),
        "constant_features_in_train": int(train.nunique(dropna=False).le(1).sum()),
        "duplicate_mpid": int(df["mpid"].duplicated().sum()),
        "train_val_formula_overlap": len(groups["train"] & groups["validation"]),
        "train_test_formula_overlap": len(groups["train"] & groups["test"]),
        "val_test_formula_overlap": len(groups["validation"] & groups["test"]),
        "target_min": float(target.min()),
        "target_q1": float(q.loc[0.25]),
        "target_median": float(q.loc[0.50]),
        "target_q3": float(q.loc[0.75]),
        "target_max": float(target.max()),
    }
    missing_rows = [
        {
            "task": spec.display,
            "feature": feature,
            "missing_count": int(df[feature].isna().sum()),
            "missing_rate": float(df[feature].isna().mean()),
        }
        for feature in features if df[feature].isna().any()
    ]
    return row, missing_rows


def evaluate_ablation(
    spec: TaskSpec,
    df: pd.DataFrame,
    features: list[str],
    base_model: BaseEstimator,
) -> list[dict[str, object]]:
    split = masks(df)
    sets = {
        "All features": features,
        "Non-structural features": [f for f in features if not f.startswith("feat_struct_")],
        "Structural features only": [f for f in features if f.startswith("feat_struct_")],
    }
    rows = []
    for label, selected in sets.items():
        if not selected:
            raise ValueError(f"No features selected for ablation set: {label}")
        model = clone(base_model).fit(df.loc[split["train"], selected], df.loc[split["train"], spec.target])
        row = {
            "task": spec.display,
            "feature_set": label,
            "feature_count": len(selected),
            "val_MAE": np.nan, "val_R2": np.nan, "test_MAE": np.nan, "test_R2": np.nan,
            "val_ROC_AUC": np.nan, "val_MCC": np.nan,
            "test_ROC_AUC": np.nan, "test_MCC": np.nan,
        }
        if spec.kind == "regression":
            for part, prefix in (("validation", "val"), ("test", "test")):
                pred = model.predict(df.loc[split[part], selected])
                met = regression_metrics(df.loc[split[part], spec.target], pred)
                row[f"{prefix}_MAE"] = met["MAE"]
                row[f"{prefix}_R2"] = met["R2"]
        else:
            val_prob = model.predict_proba(df.loc[split["validation"], selected])[:, 1]
            threshold = choose_threshold(df.loc[split["validation"], spec.target], val_prob)
            for part, prefix in (("validation", "val"), ("test", "test")):
                prob = model.predict_proba(df.loc[split[part], selected])[:, 1]
                met = classification_metrics(df.loc[split[part], spec.target], prob, threshold)
                row[f"{prefix}_ROC_AUC"] = met["ROC_AUC"]
                row[f"{prefix}_MCC"] = met["MCC"]
        rows.append(row)
    return rows


def shortened_feature_label(feature: str) -> str:
    label = feature.removeprefix("feat_")
    replacements = [
        ("struct_", "结构·"), ("phys_", "物理·"), ("orig_", "原始·"),
        ("B_mean_", "B_均值·"), ("B_absdiff_", "B_差值·"),
        ("B_range_", "B_极差·"), ("A_", "A·"), ("X_", "X·"),
    ]
    for old, new in replacements:
        if label.startswith(old):
            return new + label[len(old):]
    return label


def feature_importance_rows(
    spec: TaskSpec,
    df: pd.DataFrame,
    features: list[str],
    model: BaseEstimator,
    seed: int,
    repeats: int,
    n_jobs: int,
) -> list[dict[str, object]]:
    test = df["split_formula_group"].eq("test")
    scoring = "neg_mean_absolute_error" if spec.kind == "regression" else "roc_auc"
    result = permutation_importance(
        model, df.loc[test, features], df.loc[test, spec.target], scoring=scoring,
        n_repeats=repeats, random_state=seed, n_jobs=n_jobs,
    )
    order = np.argsort(result.importances_mean)[::-1][:25]
    return [
        {
            "task": spec.display_zh,
            "rank": rank,
            "feature": features[idx],
            "short_label": shortened_feature_label(features[idx]),
            "importance_mean": float(result.importances_mean[idx]),
        }
        for rank, idx in enumerate(order, start=1)
    ]


def repeated_grouped_cv(
    spec: TaskSpec,
    df: pd.DataFrame,
    features: list[str],
    model: BaseEstimator,
    seed: int,
    folds: int,
    repeats: int,
) -> list[dict[str, object]]:
    """Repeated grouped CV on development data only; the fixed test set stays untouched."""
    dev = df["split_formula_group"].isin(["train", "validation"])
    work = df.loc[dev].reset_index(drop=True)
    X, y, groups = work[features], work[spec.target], work["formula"].astype(str)
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        if spec.kind == "classification":
            splitter = StratifiedGroupKFold(
                n_splits=folds, shuffle=True, random_state=seed + repeat
            )
            iterator = splitter.split(X, y, groups)
        else:
            # Randomize formula groups, then greedily balance their row counts
            # across folds. This provides genuinely different repeats without
            # relying on the newer GroupKFold(shuffle=...) API.
            rng = np.random.default_rng(seed + repeat)
            counts = groups.value_counts().rename_axis("group").reset_index(name="size")
            counts["tie"] = rng.random(len(counts))
            counts = counts.sort_values(["size", "tie"], ascending=[False, True])
            fold_sizes = np.zeros(folds, dtype=int)
            assignments: dict[str, int] = {}
            for item in counts.itertuples(index=False):
                smallest = np.flatnonzero(fold_sizes == fold_sizes.min())
                chosen_fold = int(rng.choice(smallest))
                assignments[str(item.group)] = chosen_fold
                fold_sizes[chosen_fold] += int(item.size)
            fold_id = groups.map(assignments).to_numpy()
            iterator = (
                (np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold))
                for fold in range(folds)
            )
        for fold, (train_idx, valid_idx) in enumerate(iterator, start=1):
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            row = {
                "task": spec.display, "repeat": repeat + 1, "fold": fold,
                "train_size": len(train_idx), "valid_size": len(valid_idx),
                "MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                "ROC_AUC": np.nan, "PR_AUC": np.nan, "F1": np.nan,
                "Balanced_Accuracy": np.nan, "MCC": np.nan,
            }
            if spec.kind == "regression":
                row.update(regression_metrics(y.iloc[valid_idx], fitted.predict(X.iloc[valid_idx])))
            else:
                # The threshold is selected inside each fold using training
                # out-of-fold-independent probabilities at the neutral 0.5 cutoff.
                prob = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
                row.update(classification_metrics(y.iloc[valid_idx], prob, 0.5))
                row.pop("Precision", None); row.pop("Recall", None); row.pop("threshold", None)
            rows.append(row)
    return rows


def summarize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    metrics = ["MAE", "RMSE", "R2", "ROC_AUC", "PR_AUC", "F1", "Balanced_Accuracy", "MCC"]
    rows = []
    for task, group in cv.groupby("task", sort=False):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            rows.append({
                "task": task, "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "q2.5": float(np.quantile(values, 0.025)),
                "q97.5": float(np.quantile(values, 0.975)),
                "n_folds": len(values),
            })
    return pd.DataFrame(rows)


def bootstrap_ci(
    spec: TaskSpec,
    y_true: np.ndarray,
    predictions: np.ndarray,
    threshold: float | None,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    if spec.kind == "regression":
        metric_functions: dict[str, Callable] = {
            "MAE": mean_absolute_error,
            "RMSE": lambda a, b: math.sqrt(mean_squared_error(a, b)),
            "R2": r2_score,
        }
    else:
        assert threshold is not None
        metric_functions = {
            "ROC_AUC": roc_auc_score,
            "PR_AUC": average_precision_score,
            "F1": lambda a, b: f1_score(a, b >= threshold, zero_division=0),
            "Balanced_Accuracy": lambda a, b: balanced_accuracy_score(a, b >= threshold),
            "MCC": lambda a, b: matthews_corrcoef(a, b >= threshold),
        }
    values = {name: [] for name in metric_functions}
    n = len(y_true)
    attempts = 0
    while min((len(v) for v in values.values()), default=0) < samples:
        attempts += 1
        if attempts > samples * 10:
            raise RuntimeError("Could not obtain enough valid bootstrap replicates")
        idx = rng.integers(0, n, n)
        # AUC is undefined for a one-class bootstrap sample.
        if spec.kind == "classification" and len(np.unique(y_true[idx])) < 2:
            continue
        for name, func in metric_functions.items():
            values[name].append(float(func(y_true[idx], predictions[idx])))
    rows = []
    for name, func in metric_functions.items():
        distribution = np.asarray(values[name][:samples])
        rows.append({
            "task": spec.display, "metric": name,
            "point_estimate": float(func(y_true, predictions)),
            "ci_lower": float(np.quantile(distribution, 0.025)),
            "ci_upper": float(np.quantile(distribution, 0.975)),
            "bootstrap_samples": samples,
        })
    return rows


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    level = min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(scores, level, interpolation="higher"))


def regression_conformal(
    spec: TaskSpec,
    df: pd.DataFrame,
    features: list[str],
    model: BaseEstimator,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    split = masks(df)
    val_pred = model.predict(df.loc[split["validation"], features])
    scores = np.abs(df.loc[split["validation"], spec.target].to_numpy() - val_pred)
    q = conformal_quantile(scores, alpha)
    test_pred = model.predict(df.loc[split["test"], features])
    actual = df.loc[split["test"], spec.target].to_numpy()
    lower, upper = test_pred - q, test_pred + q
    meta = df.loc[split["test"], ["mpid", "formula", "canonical_formula"]].reset_index(drop=True)
    table = meta.assign(
        actual=actual, predicted=test_pred, lower_90=lower, upper_90=upper,
        covered=((actual >= lower) & (actual <= upper)).astype(int),
    )
    summary = {
        "task": spec.display,
        "method": "split_conformal_interval",
        "nominal_coverage": 1 - alpha,
        "quantile": q,
        "empirical_coverage": float(table["covered"].mean()),
        "average_width_or_set_size": float((upper - lower).mean()),
        "n_samples": len(table), "note": "",
    }
    return table, summary


def classification_conformal(
    spec: TaskSpec,
    df: pd.DataFrame,
    features: list[str],
    model: BaseEstimator,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    split = masks(df)
    val_prob = model.predict_proba(df.loc[split["validation"], features])[:, 1]
    val_y = df.loc[split["validation"], spec.target].to_numpy(dtype=int)
    true_prob = np.where(val_y == 1, val_prob, 1 - val_prob)
    q = conformal_quantile(1 - true_prob, alpha)
    cutoff = 1 - q
    test_prob = model.predict_proba(df.loc[split["test"], features])[:, 1]
    actual = df.loc[split["test"], spec.target].to_numpy(dtype=int)
    include_unstable = (1 - test_prob) >= cutoff
    include_stable = test_prob >= cutoff
    labels, sizes, covered = [], [], []
    for y, inc0, inc1 in zip(actual, include_unstable, include_stable):
        members = (["unstable"] if inc0 else []) + (["stable"] if inc1 else [])
        if not members:  # finite-sample guard; use prediction only, never the test label
            members = ["stable" if test_prob[len(labels)] >= 0.5 else "unstable"]
        labels.append("{" + ",".join(members) + "}")
        sizes.append(len(members))
        covered.append(int((y == 0 and "unstable" in members) or (y == 1 and "stable" in members)))
    meta = df.loc[split["test"], ["mpid", "formula", "canonical_formula"]].reset_index(drop=True)
    table = meta.assign(
        actual=actual, prob_stable=test_prob, prediction_set_90=labels,
        set_size=sizes, covered=covered,
    )
    summary = {
        "task": spec.display,
        "method": "split_conformal_set",
        "nominal_coverage": 1 - alpha,
        "quantile": q,
        "empirical_coverage": float(np.mean(covered)),
        "average_width_or_set_size": float(np.mean(sizes)),
        "n_samples": len(table), "note": "avg_set_size",
    }
    return table, summary


def make_candidate_screening(
    bandgap: pd.DataFrame,
    hull: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    merged = bandgap.merge(
        hull[["mpid", "predicted", "actual"]].rename(columns={
            "predicted": "pred_hull_eV_atom", "actual": "actual_hull_eV_atom"
        }), on="mpid", how="inner", validate="one_to_one",
    ).merge(
        stability[["mpid", "probability", "actual"]].rename(columns={
            "probability": "pred_stable_probability", "actual": "actual_stable"
        }), on="mpid", how="inner", validate="one_to_one",
    )
    merged = merged.rename(columns={
        "predicted": "pred_bandgap_eV", "actual": "actual_bandgap_eV"
    })
    gap_score = np.exp(-0.5 * ((merged["pred_bandgap_eV"] - 1.65) / 0.75) ** 2)
    hull_score = np.exp(-np.maximum(merged["pred_hull_eV_atom"], 0) / 0.05)
    merged["screening_score"] = (
        0.25 * gap_score + 0.30 * hull_score + 0.45 * merged["pred_stable_probability"]
    )
    merged["passes_filter"] = (
        merged["pred_bandgap_eV"].between(0.8, 2.5)
        & merged["pred_hull_eV_atom"].le(0.05)
        & merged["pred_stable_probability"].ge(0.5)
    )
    merged["joint_actual_hit"] = (
        merged["actual_bandgap_eV"].between(0.8, 2.5)
        & merged["actual_hull_eV_atom"].le(0.05)
        & merged["actual_stable"].eq(1)
    ).astype(int)
    order = [
        "mpid", "formula", "canonical_formula", "A", "B1", "B2", "X",
        "pred_bandgap_eV", "actual_bandgap_eV", "pred_hull_eV_atom",
        "actual_hull_eV_atom", "pred_stable_probability", "actual_stable",
        "screening_score", "passes_filter", "joint_actual_hit",
    ]
    return merged.sort_values(
        ["passes_filter", "screening_score"], ascending=[False, False], kind="stable"
    )[order].reset_index(drop=True)


def translate_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["task"] = out["task"].replace({spec.display: spec.display_zh for spec in TASKS.values()})
    out["model"] = out["model"].replace({
        "Mean baseline": "均值基线", "Majority baseline": "多数类基线",
        "Logistic regression": "逻辑回归",
    })
    return out


def run(args: argparse.Namespace) -> dict[str, object]:
    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    raw_dir = output_root / "data" / "raw"
    table_dir = output_root / "data" / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    task_data: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for key, spec in TASKS.items():
        df, features = load_task(input_dir, spec)
        assert_no_formula_overlap(df, spec.filename)
        task_data[key] = (df, features)
        shutil.copy2(input_dir / spec.filename, raw_dir / spec.filename)

    feature_lists = [features for _, features in task_data.values()]
    if any(features != feature_lists[0] for features in feature_lists[1:]):
        raise ValueError("The three task files do not have the same ordered feat_ columns")

    model_rows: list[dict[str, object]] = []
    fitted_by_task: dict[str, BaseEstimator] = {}
    thresholds: dict[str, float] = {}
    quality_rows, missing_rows = [], []
    ablation_rows, importance_rows, cv_rows, bootstrap_rows = [], [], [], []
    test_prediction_tables: dict[str, pd.DataFrame] = {}
    conformal_rows: list[dict[str, object]] = []

    for task_index, (key, spec) in enumerate(TASKS.items()):
        df, features = task_data[key]
        print(f"[{task_index + 1}/3] {spec.display}: model comparison", flush=True)
        rows, fitted, task_thresholds = fit_model_comparison(
            spec, df, features, args.seed, args.n_jobs, args.quick
        )
        model_rows.extend(rows)
        selected_name = selected_model_name(spec)
        selected = fitted[selected_name]
        fitted_by_task[key] = selected
        if spec.kind == "classification":
            thresholds[key] = task_thresholds[selected_name]

        quality, missing = data_quality_rows(spec, df, features)
        quality_rows.append(quality); missing_rows.extend(missing)

        split = masks(df)
        meta = prediction_metadata(df, split["test"])
        actual = df.loc[split["test"], spec.target].to_numpy()
        if spec.kind == "regression":
            pred = selected.predict(df.loc[split["test"], features])
            prediction = meta.assign(
                actual=actual, predicted=pred, residual=pred - actual,
                absolute_error=np.abs(pred - actual),
            )
            filename = f"{key}_test_predictions.csv"
            conformal, conformal_summary = regression_conformal(
                spec, df, features, selected, args.alpha
            )
            conformal_filename = f"{key}_conformal_predictions.csv"
            bootstrap_rows.extend(bootstrap_ci(
                spec, actual.astype(float), pred.astype(float), None,
                args.bootstrap_samples if not args.quick else min(80, args.bootstrap_samples),
                args.seed + task_index,
            ))
        else:
            prob = selected.predict_proba(df.loc[split["test"], features])[:, 1]
            threshold = thresholds[key]
            predicted = (prob >= threshold).astype(int)
            prediction = meta.assign(
                actual=actual.astype(int), probability=prob, predicted=predicted,
                correct=(predicted == actual).astype(int),
            )
            filename = "stability_test_predictions.csv"
            conformal, conformal_summary = classification_conformal(
                spec, df, features, selected, args.alpha
            )
            conformal_filename = "stability_conformal_predictions.csv"
            bootstrap_rows.extend(bootstrap_ci(
                spec, actual.astype(int), prob.astype(float), threshold,
                args.bootstrap_samples if not args.quick else min(80, args.bootstrap_samples),
                args.seed + task_index,
            ))
        test_prediction_tables[key] = prediction
        write_csv(prediction, table_dir / filename)
        write_csv(conformal, table_dir / conformal_filename)
        conformal_rows.append(conformal_summary)

        print(f"[{task_index + 1}/3] {spec.display}: ablation and importance", flush=True)
        ablation_rows.extend(evaluate_ablation(spec, df, features, selected))
        importance_rows.extend(feature_importance_rows(
            spec, df, features, selected, args.seed + task_index,
            args.permutation_repeats if not args.quick else min(3, args.permutation_repeats),
            args.n_jobs,
        ))

        print(f"[{task_index + 1}/3] {spec.display}: repeated grouped CV", flush=True)
        cv_rows.extend(repeated_grouped_cv(
            spec, df, features, selected, args.seed,
            args.cv_folds,
            args.cv_repeats if not args.quick else 1,
        ))

    comparison = pd.DataFrame(model_rows)[REGRESSION_COLUMNS]
    write_csv(comparison, table_dir / "model_comparison.csv")
    write_csv(translate_model_comparison(comparison), table_dir / "model_comparison_base_zh.csv")

    selected_rows = []
    for spec in TASKS.values():
        short_name = selected_model_name(spec)
        chosen = comparison[(comparison["task"] == spec.display) & (comparison["model"] == short_name)].copy()
        chosen["model"] = selected_model_long_name(spec)
        selected_rows.append(chosen)
    write_csv(pd.concat(selected_rows, ignore_index=True), table_dir / "selected_model_metrics.csv")
    write_csv(pd.DataFrame(quality_rows), table_dir / "data_quality_summary.csv")
    write_csv(pd.DataFrame(missing_rows), table_dir / "missing_features.csv")
    write_csv(pd.DataFrame(ablation_rows), table_dir / "feature_ablation.csv")
    write_csv(pd.DataFrame(importance_rows), table_dir / "permutation_feature_importance.csv")
    cv = pd.DataFrame(cv_rows)
    cv_columns = [
        "task", "repeat", "fold", "train_size", "valid_size", "MAE", "RMSE", "R2",
        "ROC_AUC", "PR_AUC", "F1", "Balanced_Accuracy", "MCC",
    ]
    write_csv(cv[cv_columns], table_dir / "repeated_grouped_cv_results.csv")
    write_csv(summarize_cv(cv), table_dir / "repeated_grouped_cv_summary.csv")
    write_csv(pd.DataFrame(bootstrap_rows), table_dir / "bootstrap_confidence_intervals.csv")
    write_csv(pd.DataFrame(conformal_rows), table_dir / "conformal_prediction_summary.csv")
    screening = make_candidate_screening(
        test_prediction_tables["bandgap"],
        test_prediction_tables["hull"],
        test_prediction_tables["stability"],
    )
    write_csv(screening, table_dir / "heldout_candidate_screening.csv")

    run_info = {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "seed": args.seed, "quick_mode": args.quick,
        "cv_folds": args.cv_folds,
        "cv_repeats": args.cv_repeats if not args.quick else 1,
        "bootstrap_samples": args.bootstrap_samples if not args.quick else min(80, args.bootstrap_samples),
        "permutation_repeats": args.permutation_repeats if not args.quick else min(3, args.permutation_repeats),
        "conformal_alpha": args.alpha,
        "feature_count": len(feature_lists[0]),
        "outputs": sorted(p.name for p in table_dir.glob("*.csv")),
    }
    (table_dir / "ml_run_metadata.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_info, ensure_ascii=False, indent=2), flush=True)
    return run_info


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")
    try:
        run(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
