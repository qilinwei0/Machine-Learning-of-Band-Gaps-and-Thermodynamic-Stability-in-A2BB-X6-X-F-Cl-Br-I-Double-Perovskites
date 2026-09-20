#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate schemas and key leakage safeguards for generated v9 data tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_TABLES = {
    "bandgap_conformal_predictions.csv": 8,
    "bandgap_test_predictions.csv": 11,
    "bootstrap_confidence_intervals.csv": 6,
    "conformal_prediction_summary.csv": 8,
    "data_quality_summary.csv": 18,
    "feature_ablation.csv": 11,
    "heldout_candidate_screening.csv": 16,
    "hull_conformal_predictions.csv": 8,
    "hull_test_predictions.csv": 11,
    "missing_features.csv": 4,
    "model_comparison.csv": 14,
    "model_comparison_base_zh.csv": 14,
    "permutation_feature_importance.csv": 5,
    "repeated_grouped_cv_results.csv": 13,
    "repeated_grouped_cv_summary.csv": 7,
    "selected_model_metrics.csv": 14,
    "stability_conformal_predictions.csv": 8,
    "stability_test_predictions.csv": 11,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path("final_complete_corrected_package_v9"))
    args = parser.parse_args()
    root = args.package_root.resolve()
    table_dir, raw_dir = root / "data" / "tables", root / "data" / "raw"
    errors = []
    summary = {}
    for name, ncols in EXPECTED_TABLES.items():
        path = table_dir / name
        if not path.exists():
            errors.append(f"missing table: {name}")
            continue
        frame = pd.read_csv(path)
        if len(frame.columns) != ncols:
            errors.append(f"{name}: expected {ncols} columns, found {len(frame.columns)}")
        if len(frame) == 0:
            errors.append(f"{name}: empty table")
        summary[name] = {"rows": len(frame), "columns": len(frame.columns)}
    for name in (
        "A2BBX6_task_bandgap_regression.csv",
        "A2BBX6_task_hull_regression.csv",
        "A2BBX6_task_stability_classification_005.csv",
    ):
        path = raw_dir / name
        if not path.exists():
            errors.append(f"missing raw input copy: {name}")
            continue
        frame = pd.read_csv(path, low_memory=False)
        groups = {
            split: set(frame.loc[frame["split_formula_group"].eq(split), "formula"].astype(str))
            for split in ("train", "validation", "test")
        }
        if groups["train"] & groups["validation"]:
            errors.append(f"{name}: train/validation formula overlap")
        if groups["train"] & groups["test"]:
            errors.append(f"{name}: train/test formula overlap")
        if groups["validation"] & groups["test"]:
            errors.append(f"{name}: validation/test formula overlap")
    result = {"status": "failed" if errors else "passed", "tables": summary, "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

