---
date: 2026-04-17
owner: GitHub Copilot
git_commit: 61c0e77604e642adfc9009e2f2d9d3891085b0f6
branch: main
repository: 4gpus-Stroke-outcome-prediction-code
status: proposed
topic: "Implementation plan: TRACE 3D + tabular regression for gs_rankin_6isdeath"
---

# Plan: TRACE 3D + Tabular Regression (gs_rankin_6isdeath)

## 1) Goal
Adapt the baseline at `code/baseline/Multimodal-mRS90-Outcome-Prediction` to train a regression model for stroke outcome `gs_rankin_6isdeath` using:
- TRACE 3D image input
- Tabular clinical features
- Existing split files under `code/datasets/fold_raw_trace_fullmodal_mask`

Primary objective: produce a reproducible training/evaluation path for regression while preserving the current multimodal architecture style (image encoder + tabular tokens + attention fusion).

## 2) Grounded data contract (from current split artifacts)
Source folder: `code/datasets/fold_raw_trace_fullmodal_mask`

- Files present: `train.csv`, `valid.csv`, `test.csv`, `split_summary.json`
- Declared stratification: `gs_rankin_6isdeath`
- Split sizes from summary:
  - train: 435
  - valid: 93
  - test: 94
- Target distribution values: ordinal-like range 0..6 (stored as float)
- CSV columns include:
  - IDs and paths: `subject_id`, `image_path`, `mask_path`, `trace_dir`, `t1_dir`, `flair_dir`, `adc_dir`
  - labels: `nihss`, `gs_rankin_6isdeath`
  - tabular fields: `sex`, `age`, `race`, `acuteischaemicstroke`, `priorstroke`, `bmi`, `etiology`
  - serialized tabular JSON: `tabular_features`

Important data quality notes found in the split files:
- Missing values exist in tabular columns (e.g., `bmi`, `age`, `race`, `etiology`).
- At least one row contains a likely typo path token (`lesionAcute_mask`) instead of the common `lesion_mask` naming.
- Paths are Linux-style absolute paths (`/mnt/disk1/...`) that may need environment mapping.

## 3) Scope
In scope:
- Convert baseline task from binary classification to regression on `gs_rankin_6isdeath`.
- Add/adjust data loading to consume split CSV schema (`image_path` + tabular columns or `tabular_features`).
- Keep model backbone/fusion design mostly intact.
- Add regression-appropriate losses/metrics and result reporting.

Out of scope (for v1):
- New architecture research (e.g., ordinal heads, uncertainty modeling).
- Full hyperparameter sweep infrastructure.
- Cross-project refactors across all baseline_encoder models.

## 4) File-level change plan

### A. Baseline config and run wiring
1. `code/baseline/Multimodal-mRS90-Outcome-Prediction/python/model/config.py`
- Keep current classification settings untouched and add a second config block for TRACE regression.
- Add these exact keys:
  - `experiment_mode = "trace_regression"`
  - `trace_split_dir = "./code/datasets/fold_raw_trace_fullmodal_mask"`
  - `trace_target = "gs_rankin_6isdeath"`
  - `trace_subject_id_col = "subject_id"`
  - `trace_tabular_continuous = ["age", "bmi"]`
  - `trace_tabular_categorical = ["sex", "race", "acuteischaemicstroke", "priorstroke", "etiology"]`
  - `trace_tabular_all = ["sex", "age", "race", "acuteischaemicstroke", "priorstroke", "bmi", "etiology"]`
  - `trace_image_shape = (256, 256, 26)`
  - `trace_timepoints = 1`
  - `trace_batch_size = 1`
  - `trace_train_csv = trace_split_dir + "/train.csv"`
  - `trace_valid_csv = trace_split_dir + "/valid.csv"`
  - `trace_test_csv = trace_split_dir + "/test.csv"`
- In existing `params`, add parallel keys for trace mode:
  - `params['trace_dim'] = (256, 256, 26)`
  - `params['trace_timepoints'] = 1`
  - `params['trace_n_classes'] = 1`
  - `params['trace_features'] = [trace_tabular_continuous, trace_tabular_categorical]`

2. `code/baseline/Multimodal-mRS90-Outcome-Prediction/python/model/main.py`
- Add a branch at data-load stage:
  - If `experiment_mode == "classification"`: run existing path as-is.
  - If `experiment_mode == "trace_regression"`: load split CSV files directly with pandas.
- For trace mode, use exact split sources:
  - train dataframe from `trace_train_csv`
  - valid dataframe from `trace_valid_csv`
  - test dataframe from `trace_test_csv`
- Build generators from row-based split data (not patient_dictionary).
- Use exact regression head/compile for trace mode:
  - final layer `Dense(1, activation='linear')`
  - loss `tf.keras.losses.Huber(delta=1.0)` (or MSE in ablation run)
  - metrics `[tf.keras.metrics.MeanAbsoluteError(name='mae'), tf.keras.metrics.RootMeanSquaredError(name='rmse')]`
- Save outputs in trace mode as:
  - `predictions_regression.npz` with arrays `pred`, `y_true`, `subject_id`
  - `metrics_regression.json` with keys `val_mae`, `val_rmse`, `test_mae`, `test_rmse`, `n_train`, `n_valid`, `n_test`

### B. Data loading adaptation
3. `code/baseline/Multimodal-mRS90-Outcome-Prediction/python/model/data_generator.py`
- Keep `DataGenerator_CTP` unchanged for compatibility.
- Add a new class in this file: `DataGenerator_TRACE_Regression(Sequence)` with constructor:
  - `split_df`
  - `target_col`
  - `continuous_features`
  - `categorical_features`
  - `dim=(256, 256, 26)`
  - `batch_size=1`
  - `shuffle=True`
  - optional `path_remap_from`, `path_remap_to`
- In `__data_generation` implement exact tensor contract:
  - `X` shape `(batch_size, 256, 256, 26, 1)` float32
  - `C_continuous` shape `(batch_size, 2)` float32 for `age`, `bmi`
  - `C_categorical` shape `(batch_size, 5)` int32 for `sex`, `race`, `acuteischaemicstroke`, `priorstroke`, `etiology`
  - `y` shape `(batch_size,)` float32 from `gs_rankin_6isdeath`
- Image loading path:
  - read `image_path` NIfTI
  - apply optional prefix remap (`/mnt/disk1` -> local root)
  - ensure output volume is resized/padded/cropped to `(256,256,26)`
  - normalize intensity per-volume (min-max to [0,1])
- Tabular extraction path:
  - preferred: explicit CSV columns in `trace_tabular_all`
  - fallback: parse `tabular_features` JSON when explicit column missing
  - impute missing: median for continuous (`age`, `bmi`), mode for categorical
- Return format must mirror current model input style:
  - `X_list = [X]` (single timepoint/token)
  - `C_categorical_list = np.split(C_categorical, 5, axis=1)`
  - `C_continuous_list = np.split(C_continuous, 2, axis=1)`
  - return `([X_list, C_categorical_list, C_continuous_list], y)`

4. (Optional, cleaner) New helper module in same folder, e.g. `trace_data_io.py`
- If created, include exactly these helpers:
  - `load_trace_nifti(path, out_shape=(256,256,26)) -> np.ndarray`
  - `remap_path(path, src_prefix, dst_prefix) -> str`
  - `extract_tabular_row(row, continuous_cols, categorical_cols, defaults) -> (np.ndarray, np.ndarray)`
- Keep helper-only logic here; batching remains in generator.

### C. Utility and evaluation outputs
5. `code/baseline/Multimodal-mRS90-Outcome-Prediction/python/model/utils.py`
- No required edits if model receives image tokens and tabular tokens in the same structure as current path.
- Confirm one shape assumption during implementation:
  - `MultimodalFusion.build(... embed_dim=cfg.transformer_params['projection_dim'])` currently references `projection_dim` not present in config.
  - For trace mode, set `embed_dim = imaging_encoded.shape[-1]` or define `projection_dim` in config.

6. New output artifacts in `resultsPath` (existing folder)
- `predictions_regression.npz` with keys: `y_true`, `y_pred`, `subject_id`.
- `metrics_regression.json` with MAE/RMSE and run metadata.

### D. Exact model input shape conversion for TRACE
7. `code/baseline/Multimodal-mRS90-Outcome-Prediction/python/model/main.py`
- Replace the current loop over `cfg.params['timepoints']` for trace mode only.
- Existing CTP path creates 32 Inputs each shaped `(*dim,1)` and concatenates them.
- Trace path must create one input tensor:
  - `trace_input = Input(shape=(256,256,26,1), name='TRACE')`
  - `trace_output = base_network(trace_input)[0]`
  - `imaging_encoded = Concatenate(axis=1)([trace_output])` or directly `trace_output`
- Keep downstream self-attention and fusion calls unchanged.
- Keep tabular tokenization structure, but feature names come from trace config lists.

## 5) Implementation phases

### Phase 0: Reproducible setup and checks
- Confirm environment dependencies for NIfTI IO (`nibabel` if not already installed).
- Verify split files can be read end-to-end.
- Add sanity check script/function to validate required columns and count missing values.

Exit criteria:
- Split files parse with no schema errors.
- Required columns and target found.

Concrete checks to run:
- Assert each split has columns: `subject_id`, `image_path`, `gs_rankin_6isdeath`
- Assert tabular columns exist or `tabular_features` exists
- Print missing counts for: `age`, `bmi`, `sex`, `race`, `acuteischaemicstroke`, `priorstroke`, `etiology`

### Phase 1: Data pipeline adaptation
- Implement CSV split reader and path resolver.
- Implement TRACE volume loading from `image_path`.
- Implement tabular extraction and imputation policy.
- Return model-ready batch tensors + regression target.

Exit criteria:
- One batch from train/valid/test loads without crash.
- Batch tensor shapes and dtypes are stable.

Required shape assertions:
- Image branch batch element: `(256,256,26,1)`
- `len(X_list) == 1`
- `len(C_continuous_list) == 2`
- `len(C_categorical_list) == 5`
- `y.dtype == float32`

### Phase 2: Model objective conversion
- Keep backbone and multimodal fusion.
- Change output head/compile settings to regression mode.
- Ensure training loop, validation, and inference run with regression outputs.

Exit criteria:
- 1-epoch smoke training completes.
- Prediction export includes true and predicted values.

Concrete code changes in this phase:
- Prediction head: replace sigmoid with linear only in trace mode.
- Compile: replace BCE/accuracy with Huber(MSE ablation)/MAE/RMSE in trace mode.
- Generators: replace dictionary partition usage with train/valid/test split dataframes in trace mode.

### Phase 3: Evaluation and reporting
- Compute and persist MAE/RMSE on validation and test.
- Add simple calibration/error diagnostics (e.g., residual summary by target bin 0..6).

Exit criteria:
- `metrics_regression.json` created with consistent schema.
- Test predictions can be joined back to subject IDs.

Exact evaluation table fields:
- `subject_id`
- `y_true`
- `y_pred`
- `abs_error`
- `split`

### Phase 4: First experiment matrix
Run at least 3 controlled experiments:
1. Baseline regression (MSE)
2. Robust regression (Huber)
3. With vs without tabular branch (ablation)

Report each with fixed seed and identical splits.

## 6) Validation criteria

### Automated checks
- Data contract tests:
  - Required columns present in split CSVs.
  - No missing target labels in any split.
- Loader smoke test:
  - Iterate 2-3 batches per split.
- Training smoke test:
  - 1 epoch end-to-end, no runtime errors.
- Artifact test:
  - predictions and metrics files written and readable.

Expected baseline training command (after edits):
- `python ./python/model/main.py --mode trace_regression`

### Manual checks
- Inspect outlier rows where paths are malformed or missing.
- Verify predicted range behavior (not exploding, reasonable against 0..6 labels).
- Verify train/valid/test split sizes match `split_summary.json`.

## 7) Risks and mitigations

Risk: Path portability (`/mnt/disk1/...`) differs by machine.
- Mitigation: add configurable path-prefix remap in loader.

Risk: Missing tabular values and mixed encoding in `tabular_features`.
- Mitigation: explicit imputation policy (median for continuous, mode/unknown for categorical) with logging.

Risk: One-off malformed mask/image path entries.
- Mitigation: sample-level error handling + skip list report; fail-fast option for strict runs.

Risk: Treating ordinal target as plain regression may underperform.
- Mitigation: keep v1 as regression baseline; evaluate ordinal-aware head in future phase.

## 8) Suggested concrete command flow (after implementation)
1. Run data sanity check against `fold_raw_trace_fullmodal_mask`.
2. Run 1-epoch smoke training.
3. Run full training with fixed seed.
4. Export metrics/predictions and summarize MAE/RMSE.

Concrete sequence:
1. `python ./python/model/main.py --mode trace_regression --smoke 1`
2. `python ./python/model/main.py --mode trace_regression --epochs 100 --loss huber`
3. `python ./python/model/main.py --mode trace_regression --epochs 100 --loss mse`
4. `python ./python/model/main.py --mode trace_regression --epochs 100 --disable-tabular 1`

## 9) Deliverables
- Updated baseline code supporting regression mode and TRACE split CSV input.
- `metrics_regression.json` and `predictions_regression.npz` in results output.
- Brief run note in research folder summarizing first experiment outcomes.

## 10) Definition of done
- End-to-end training + test inference works using `fold_raw_trace_fullmodal_mask`.
- Target is `gs_rankin_6isdeath` (regression objective).
- Reproducible metrics (MAE/RMSE) and prediction artifacts are produced.
- Original classification pathway remains runnable (no regression-induced breakage).

## Execution Status (2026-04-17)
- [x] Phase 0 implemented: split schema and missingness validator created and executed.
- [x] Phase 1 implemented: TRACE data IO and regression generator implemented with required tensor contracts.
- [x] Phase 2 implemented: regression training entrypoint implemented with linear head, Huber/MSE, MAE/RMSE.
- [ ] Phase 3 runtime completion pending: full metric artifact generation blocked by local TensorFlow DLL runtime import failure.
- [ ] Phase 4 runtime completion pending: experiment matrix runs depend on resolving TensorFlow runtime environment.
