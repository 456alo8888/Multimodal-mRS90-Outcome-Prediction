#!/usr/bin/env bash
set -euo pipefail

# Example entrypoint for running the TRACE matrix script.
# Override env vars before calling if needed.
DISABLE_WANDB="${DISABLE_WANDB:-1}" \
RUNS_BASE_DIR="${RUNS_BASE_DIR:-./research/results_trace_regression/runs_example}" \
bash ./research/run_trace_regression_matrix.sh
