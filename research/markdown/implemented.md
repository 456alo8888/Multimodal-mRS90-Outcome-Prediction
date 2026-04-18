---
date: 2026-04-17
owner: GitHub Copilot
status: implemented
scope: TRACE 3D + tabular regression execution in research folder
---

# Implemented Work Log

## Constraint followed
All newly added code was placed under:
- `code/baseline/Multimodal-mRS90-Outcome-Prediction/research`

No new files were created outside this folder.

## New code files
1. `trace_regression_config.py`
- Adds TRACE regression configuration block.
- Encodes split paths, target column, tabular columns, image shape `(256, 256, 26)`, and runtime defaults.

2. `trace_data_io.py`
- Implements:
  - `remap_path(path, src_prefix, dst_prefix)`
  - `load_trace_nifti(path, out_shape=(256,256,26))`
  - `extract_tabular_row(...)`
- Includes resize/pad/crop to exact TRACE shape and per-volume min-max normalization.
- Supports tabular fallback from `tabular_features` JSON.

3. `trace_data_generator.py`
- Adds class `DataGenerator_TRACE_Regression(Sequence)`.
- Batch tensor contracts implemented exactly:
  - `X`: `(batch_size, 256, 256, 26, 1)` float32
  - `C_continuous`: `(batch_size, 2)` float32 (`age`, `bmi`)
  - `C_categorical`: `(batch_size, 5)` int32 (`sex`, `race`, `acuteischaemicstroke`, `priorstroke`, `etiology`)
  - `y`: `(batch_size,)` float32 (`gs_rankin_6isdeath`)
- Returns model input structure:
  - `X_list = [X]`
  - split categorical and continuous tensors to per-feature lists
  - output: `([X_list, C_categorical_list, C_continuous_list], y)`

4. `check_trace_split.py`
- Implements Phase 0 data-contract checks for `train.csv`, `valid.csv`, `test.csv`.
- Verifies required columns and reports missingness for tabular fields and target.

5. `trace_regression_main.py`
- Adds CLI entrypoint for TRACE regression training flow.
- Loads split CSVs directly (no patient dictionary).
- Fits train-only tabular defaults/mappings.
- Builds model with TRACE input shape `(256,256,26,1)` and tabular token branches.
- Uses regression objective:
  - output head: `Dense(1, activation='linear')`
  - loss: Huber or MSE
  - metrics: MAE and RMSE
- Writes artifacts:
  - `results_trace_regression/predictions_regression.npz`
  - `results_trace_regression/metrics_regression.json`
- Supports ablation via `--disable-tabular 1`.

## Verification executed
1. Syntax compile
- Command: `python -m compileall <new research files>`
- Result: success

2. Data contract check
- Command: `python research/check_trace_split.py`
- Result: success
- Confirmed rows:
  - train: 435
  - valid: 93
  - test: 94
- Confirmed required columns and zero missing target values in all splits.

3. Entrypoint interface check
- Command: `python research/trace_regression_main.py --help`
- Result: success
- CLI options exposed correctly.

## Environment limitation encountered during runtime training
- TensorFlow native runtime import fails in this environment with DLL load error (`_pywrap_tensorflow_internal`).
- This blocks running end-to-end training on this machine until TensorFlow runtime is fixed.
- Data-contract and code-path validation are completed; full train/eval run remains pending environment readiness.

## Output locations
- Source code: `code/baseline/Multimodal-mRS90-Outcome-Prediction/research`
- Runtime outputs (when training runs):
  - `code/baseline/Multimodal-mRS90-Outcome-Prediction/research/results_trace_regression/predictions_regression.npz`
  - `code/baseline/Multimodal-mRS90-Outcome-Prediction/research/results_trace_regression/metrics_regression.json`
