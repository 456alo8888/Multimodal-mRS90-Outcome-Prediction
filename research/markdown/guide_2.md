# TRACE Regression Guide 2: W&B and Matrix Runs

## Environment
All commands in this guide use environment hieupcvp.

## 1) Single run with W&B disabled
```bash
python research/trace_regression_main.py \
  --smoke 1 \
  --epochs 1 \
  --disable-wandb
```

## 2) Single run with W&B enabled
```bash
python research/trace_regression_main.py \
  --epochs 100 \
  --loss huber \
  --wandb-project Kimberly-stroke-outcome-prediction \
  --wandb-run-name trace_huber_full
```

Optional W&B identity:
```bash
python research/trace_regression_main.py \
  --epochs 100 \
  --loss huber \
  --wandb-project Kimberly-stroke-outcome-prediction \
  --wandb-entity YOUR_ENTITY \
  --wandb-run-name trace_huber_full
```

## 3) Output isolation controls
Use output directory and run tag to keep artifacts from each run separated.

```bash
python research/trace_regression_main.py \
  --epochs 1 \
  --smoke 1 \
  --disable-wandb \
  --output-dir research/results_trace_regression/runs \
  --run-tag smoke_a
```

Artifacts are written to:
- output-dir/run-tag/predictions_regression.npz
- output-dir/run-tag/metrics_regression.json

If run-tag is omitted, artifacts are written directly under output-dir.
If output-dir is omitted, default is research/results_trace_regression.

## 4) Run full matrix script
```bash
bash research/run_trace_regression_matrix.sh
```

Optional environment controls:
- WANDB_PROJECT
- WANDB_ENTITY
- TRACE_PATH_REMAP_FROM
- TRACE_PATH_REMAP_TO
- RUNS_BASE_DIR
- DISABLE_WANDB (1 to disable, 0 to enable)

Example:
```bash
WANDB_PROJECT=Kimberly-stroke-outcome-prediction \
DISABLE_WANDB=1 \
RUNS_BASE_DIR=research/results_trace_regression/runs_local \
bash research/run_trace_regression_matrix.sh
```

## 5) Example wrapper script
A simple wrapper is available at bash.sh:
```bash
bash ./bash.sh
```
