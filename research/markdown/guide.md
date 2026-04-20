---
date: 2026-04-17
owner: GitHub Copilot
status: ready
---

# Run Guide: TRACE Regression (Research Folder)

## 1) Paths
All commands below are from repository root:
- `D:\4gpus-Stroke-outcome-prediction-code`

Implementation files:
- `code/baseline/Multimodal-mRS90-Outcome-Prediction/research/trace_regression_config.py`
- `code/baseline/Multimodal-mRS90-Outcome-Prediction/research/trace_data_io.py`
- `code/baseline/Multimodal-mRS90-Outcome-Prediction/research/trace_data_generator.py`
- `code/baseline/Multimodal-mRS90-Outcome-Prediction/research/check_trace_split.py`
- `code/baseline/Multimodal-mRS90-Outcome-Prediction/research/trace_regression_main.py`

## 2) Environment requirements
Python packages required:
- numpy
- pandas
- scipy
- nibabel
- tensorflow

Install example:
```powershell
pip install numpy pandas scipy nibabel tensorflow
```

## 3) Phase 0 checks (required first)
Run split schema and missingness checks:
```powershell
python code/baseline/Multimodal-mRS90-Outcome-Prediction/research/check_trace_split.py
```

Expected:
- required columns present in train/valid/test
- missing target count is zero
- missingness report for `age`, `bmi`, `sex`, `race`, `acuteischaemicstroke`, `priorstroke`, `etiology`

## 4) Path remapping for Linux-style paths
Split CSV image paths are `/mnt/disk1/...`.
If your local data path differs, pass remap options:
- `--path-remap-from "/mnt/disk1"`
- `--path-remap-to "<local-prefix>"`

Example:
```powershell
python code/baseline/Multimodal-mRS90-Outcome-Prediction/research/trace_regression_main.py --path-remap-from "/mnt/disk1" --path-remap-to "D:/aiotlab"
```

## 5) Smoke run (quick)
```powershell
python code/baseline/Multimodal-mRS90-Outcome-Prediction/research/trace_regression_main.py --smoke 1 --epochs 1
```

## 6) Full runs from plan matrix
1. Huber regression:
```powershell
python code/baseline/Multimodal-mRS90-Outcome-Prediction/research/trace_regression_main.py --epochs 100 --loss huber
```

2. MSE regression:
```powershell
python code/baseline/Multimodal-mRS90-Outcome-Prediction/research/trace_regression_main.py --epochs 100 --loss mse
```

3. Ablation without tabular branch:
```powershell
python code/baseline/Multimodal-mRS90-Outcome-Prediction/research/trace_regression_main.py --epochs 100 --loss huber --disable-tabular 1
```

## 7) Outputs
Generated under:
- `code/baseline/Multimodal-mRS90-Outcome-Prediction/research/results_trace_regression`

Files:
- `predictions_regression.npz`
  - keys: `pred`, `y_true`, `subject_id`
- `metrics_regression.json`
  - includes: `val_mae`, `val_rmse`, `test_mae`, `test_rmse`, split sizes, loss type

## 8) W&B logging options
Optional experiment tracking flags are available in `trace_regression_main.py`:
- `--wandb-project`
- `--wandb-run-name`
- `--wandb-entity`
- `--disable-wandb`

Smoke run with W&B disabled:
```bash
conda run -n hieupcvp python research/trace_regression_main.py --smoke 1 --epochs 1 --disable-wandb
```

Smoke run with W&B enabled (offline mode example):
```bash
WANDB_MODE=offline conda run -n hieupcvp python research/trace_regression_main.py --smoke 1 --epochs 1 --wandb-project stroke-outcome-prediction --wandb-run-name trace_smoke
```

## 9) Output isolation options
To avoid overwriting artifacts across runs, use:
- `--output-dir`
- `--run-tag`

Example:
```bash
conda run -n hieupcvp python research/trace_regression_main.py --smoke 1 --epochs 1 --disable-wandb --output-dir research/results_trace_regression/runs --run-tag smoke_a
```

Output files remain named:
- `predictions_regression.npz`
- `metrics_regression.json`

They are saved under:
- `output-dir/run-tag/` when `--run-tag` is provided
- `output-dir/` when `--run-tag` is omitted
- default `research/results_trace_regression/` when `--output-dir` is omitted

## 10) Matrix runner script
Run the standard matrix from the research folder script:
```bash
bash research/run_trace_regression_matrix.sh
```

Script environment controls:
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `TRACE_PATH_REMAP_FROM`
- `TRACE_PATH_REMAP_TO`
- `RUNS_BASE_DIR`
- `DISABLE_WANDB` (`1` disables W&B, `0` enables)
- `RUN_PRESET` (`all` for full matrix, `smoke` for smoke-only)

Smoke-only matrix example:
```bash
RUN_PRESET=smoke DISABLE_WANDB=1 bash research/run_trace_regression_matrix.sh
```

## 11) Wrapper script
An example wrapper script is available at repository root:
```bash
bash ./bash.sh
```

## 12) Notes on current environment
If TensorFlow fails to import with DLL error on Windows, training will not start. In that case:
1. Verify Python/TensorFlow compatibility.
2. Verify Microsoft Visual C++ runtime installation.
3. Re-run `--help` first, then smoke run after environment fix.
