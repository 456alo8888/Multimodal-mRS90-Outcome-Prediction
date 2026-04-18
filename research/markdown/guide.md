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

## 8) Notes on current environment
If TensorFlow fails to import with DLL error on Windows, training will not start. In that case:
1. Verify Python/TensorFlow compatibility.
2. Verify Microsoft Visual C++ runtime installation.
3. Re-run `--help` first, then smoke run after environment fix.
