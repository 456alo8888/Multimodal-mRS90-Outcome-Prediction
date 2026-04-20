#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Paths
# =========================================================
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_SCRIPT="$ROOT/research/trace_regression_main.py"
OUT_BASE="${RUNS_BASE_DIR:-$ROOT/research/results_trace_regression/runs}"
export WANDB_API_KEY="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"
mkdir -p "$OUT_BASE"

# =========================================================
# Environment
# =========================================================
export WANDB_PROJECT="${WANDB_PROJECT:-Kimberly-stroke-outcome-prediction}"
export WANDB_ENTITY="hieupcvp-hust"
export DISABLE_WANDB="${DISABLE_WANDB:-0}"

export TRACE_PATH_REMAP_FROM="${TRACE_PATH_REMAP_FROM:-/mnt/disk1}"
export TRACE_PATH_REMAP_TO="${TRACE_PATH_REMAP_TO:-}"

export RUN_PRESET="${RUN_PRESET:-all}"

if [[ -z "${WANDB_API_KEY:-}" && "${DISABLE_WANDB}" != "1" ]]; then
	echo "[WARN] WANDB_API_KEY is not set. wandb may fail to sync online."
fi

# =========================================================
# Common explicit arguments
# =========================================================
COMMON_ARGS=(
	--wandb-project "$WANDB_PROJECT"
	--output-dir "$OUT_BASE"
	--path-remap-from "$TRACE_PATH_REMAP_FROM"
	--path-remap-to "$TRACE_PATH_REMAP_TO"
)

if [[ -n "$WANDB_ENTITY" ]]; then
	COMMON_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

if [[ "$DISABLE_WANDB" == "1" ]]; then
	COMMON_ARGS+=(--disable-wandb)
fi

run_cfg() {
	local run_name="$1"
	shift

	local timestamp
	timestamp="$(date +%Y%m%d_%H%M%S)"
	local out_dir="$OUT_BASE/${run_name}_${timestamp}"

	mkdir -p "$out_dir"

	echo "[$(date -Iseconds)] START $run_name"
	python "$PYTHON_SCRIPT" \
		"${COMMON_ARGS[@]}" \
		--output-dir "$out_dir" \
		--wandb-run-name "$run_name" \
		--run-tag "$run_name" \
		"$@"
	echo "[$(date -Iseconds)] END $run_name"
}

# =========================================================
# Presets
# =========================================================
if [[ "$RUN_PRESET" == "smoke" ]]; then
	run_cfg "trace_smoke" \
		--smoke 1 \
		--epochs 1
else
	run_cfg "trace_huber" \
		--epochs 100 \
		--loss huber

	run_cfg "trace_mse" \
		--epochs 100 \
		--loss mse

	run_cfg "trace_huber_no_tab" \
		--epochs 100 \
		--loss huber \
		--disable-tabular 1
fi

echo "All trace regression runs completed. Outputs are under: $OUT_BASE"