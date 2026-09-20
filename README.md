# `final_complete_corrected_package_v9/data` Machine Learning Reproducibility Code

This code package generates all machine learning data for `data/raw` and `data/tables` based on the original task CSVs. It reads the 398 features starting with `feat_` and uses the existing `split_formula_group` as the sole basis for the training, validation, and test splits.

## 1. Input Files

Place the following three files in the same folder:

* `A2BBX6_task_bandgap_regression.csv`
* `A2BBX6_task_hull_regression.csv`
* `A2BBX6_task_stability_classification_005.csv`

## 2. Install Dependencies

Python 3.10–3.13 is recommended:

```bash
python -m pip install -r requirements.txt

```

## 3. Full Run

Windows:

```bat
run_v9_ml.bat "D:\Your_Data_Folder" "D:\Output\v9"

```

Linux/macOS:

```bash
bash run_v9_ml.sh "/path/to/input" "/path/to/output/v9"

```

Alternatively, you can run it directly:

```bash
python generate_v9_ml_data.py --input-dir /path/to/input --output-root /path/to/v9

```

The full mode defaults include: 500 Extra Trees, HistGradientBoosting, 3 iterations × 4 folds repeated grouped cross-validation, 500 Bootstraps, and 10 permutation importances. Different CPUs will affect the runtime, but the results are reproducible given the same Python/NumPy/pandas/scikit-learn versions and random seed.

For an initial check, you can use the quick mode:

```bash
python generate_v9_ml_data.py --input-dir /path/to/input --output-root /path/to/test_output --quick

```

The quick mode is solely for verifying the environment and output format, and cannot replace the formal results in the paper.

## 4. Methodological Design

* **Data splitting:** Strictly reuses `split_formula_group`. Random row-wise splitting is prohibited to prevent cross-set leakage of polymorphs with the same chemical formula.
* **Input features:** Only `feat_` columns are used; `mpid`, element symbols, space groups, quality control fields, and `target_` labels are excluded from the models.
* **Preprocessing:** Median imputation for missing values is fitted only on the training set or within the cross-validation training folds; standardization for Ridge and Logistic Regression is also exclusively fitted at the training stage.
* **Model comparison:** Regression compares a mean baseline, Ridge, Extra Trees, and HistGradientBoosting; classification compares a majority class baseline, Logistic Regression, Extra Trees, and HistGradientBoosting.
* **Fixed primary models:** Extra Trees is used for bandgap regression; HistGradientBoosting is used for hull energy regression and stability classification.
* **Classification thresholds:** Selected solely based on MCC on the fixed validation set; the test set is not involved in threshold selection.
* **Repeated cross-validation:** Formula-grouped cross-validation is performed strictly within the development set (composed of the training set + validation set); the fixed test set is permanently held out.
* **Bootstrap:** Pairwise resampling is performed on the fixed test set to generate 95% confidence intervals.
* **Conformal prediction:** The fixed validation set acts as the calibration set; the test set is used exclusively for coverage evaluation.
* **Feature ablation:** Three groups evaluated: all features, non-structural features, and only structural features (`feat_struct_`).
* **Candidate screening:** Predicted bandgap of 0.8–2.5 eV, predicted hull energy no higher than 0.05 eV/atom, and predicted stability probability no lower than 0.5. The composite score is used only for intra-group ranking.

## 5. Outputs

The program generates:

* `data/raw/`: Exact copies of the three task inputs;
* `data/tables/`: 18 CSV tables;
* `data/tables/ml_run_metadata.json`: Records of the runtime environment, random seed, and parameters.

After generation, execute:

```bash
python validate_v9_data.py --package-root /path/to/v9

```

The validation script checks for the existence of the 18 tables, column counts, empty tables, and formula group cross-leakage across the three original task datasets.

## 6. Important Notes

Model results will be slightly influenced by the scikit-learn version and the parallel computing environment. To achieve exact numerical reproducibility of a specific historical run, please retain the `ml_run_metadata.json` generated during that run and use the exact same dependency versions.
