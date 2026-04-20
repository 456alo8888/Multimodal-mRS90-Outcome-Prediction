# TRACE W&B Logging and Run Matrix Implementation Plan

## Overview

Implement optional Weights & Biases logging for the TRACE regression pipeline and add a reproducible bash run matrix script for smoke/full experiments.

This plan targets the current research entrypoint and run flow, keeping model behavior unchanged while improving experiment tracking, reproducibility, and run ergonomics.

## Current State Analysis

Current implementation is in the research folder and already supports the required run variants via CLI:
- Existing CLI arguments in `research/trace_regression_main.py:15-25` include `--epochs`, `--loss`, `--smoke`, `--disable-tabular`, and path remap options.
- Training/evaluation flow is in `research/trace_regression_main.py:143-283`.
- Artifacts are written to fixed names in a fixed output directory:
  - `predictions_regression.npz` in `research/trace_regression_main.py:251-257`
  - `metrics_regression.json` in `research/trace_regression_main.py:277-279`
  - output directory configured in `research/trace_regression_config.py:47`
- Guide already defines the desired run matrix commands (smoke, huber, mse, no-tabular) in `research/markdown/guide.md:55-74`.
- There is currently no W&B usage in this TRACE research code path.

Key technical constraints discovered:
- Generator length uses floor division (`research/trace_data_generator.py:40-41`), so trailing samples may be dropped when split size is not divisible by batch size.
- Subject IDs are trimmed to used prediction length (`research/trace_regression_main.py:246-249`) to preserve alignment.
- Current output filenames are static and can be overwritten between runs.

## Desired End State

After implementation:
1. TRACE training can run with or without W&B (`--disable-wandb` default-safe behavior).
2. Per-epoch and final validation/test metrics are logged to W&B in a stable schema.
3. Artifacts are isolated per run (or at minimum copied to per-run subdirectories) to avoid overwrite across matrix runs.
4. A bash script runs the standard matrix from `guide.md` with optional environment controls and clear per-run naming.
5. All run commands for execution in this plan use environment `hieupcvp`.

### Key Discoveries
- Best integration points for W&B are:
  - argument surface near `research/trace_regression_main.py:15-25`
  - run init after parse/main setup near `research/trace_regression_main.py:143-151`
  - training logging around `model.fit(...)` at `research/trace_regression_main.py:234-239`
  - final logging after eval/metrics assembly at `research/trace_regression_main.py:241-279`
- Existing project patterns for W&B init/log/finish are available in:
  - `code/proposal/experiment2/run_mrs_v2.py`
  - `code/proposal/experiment1/run_mrs_v1.py:888-946`
- Existing project patterns for matrix bash runners are available in:
  - `code/proposal/experiment1/run_soop_mrs_v1.sh`
  - `code/proposal/experiment2/run_soop_mrs_v2.sh`

## What We're NOT Doing

- No model architecture changes (encoder/fusion/head remain unchanged).
- No dataset schema or split regeneration changes.
- No hyperparameter sweep tooling in this task.
- No migration of research TRACE flow into another framework.

## Implementation Approach

Use additive, low-risk changes:
1. Add optional W&B integration to the existing script with graceful fallback/disable.
2. Add lightweight run metadata and run directory handling to prevent artifact overwrite.
3. Add a dedicated bash matrix script for the four guide-defined runs.
4. Keep existing direct CLI usage backward compatible.

---

## Phase 1: Add Optional W&B Instrumentation in TRACE Entrypoint

### Overview
Add CLI flags and runtime lifecycle for W&B (init, per-epoch logging, final metrics logging, finish), while preserving current behavior when disabled.

### Changes Required

#### 1. Extend TRACE CLI for experiment tracking
**File**: `research/trace_regression_main.py`
**Changes**:
- Add args:
  - `--wandb-project`
  - `--wandb-run-name`
  - `--wandb-entity`
  - `--disable-wandb`
- Add helper functions:
  - `_build_wandb_run(args)`
  - `_build_wandb_payload(prefix, metrics, samples, epoch_or_step=None)`

```python
parser.add_argument("--wandb-project", type=str, default="stroke-outcome-prediction")
parser.add_argument("--wandb-run-name", type=str, default="")
parser.add_argument("--wandb-entity", type=str, default="")
parser.add_argument("--disable-wandb", action="store_true")
```

#### 2. Add Keras-compatible epoch logging
**File**: `research/trace_regression_main.py`
**Changes**:
- Add a small `tf.keras.callbacks.Callback` to log epoch metrics (loss/mae/rmse/r2 for train/val) with explicit epoch index.
- Pass callback into `model.fit(..., callbacks=[...])` only when W&B is enabled.

```python
class WandbMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs contains loss, mae, rmse, r2, val_loss, val_mae, val_rmse, val_r2
        ...
```

#### 3. Add final metrics and config logging + cleanup
**File**: `research/trace_regression_main.py`
**Changes**:
- Log final validation/test metrics after `model.evaluate(...)`.
- Log run metadata (loss type, disable_tabular, epochs, split sizes).
- Ensure `wandb_run.finish()` executes in a `finally` block.

### Success Criteria

#### Automated Verification:
- [x] CLI shows W&B flags: `conda run -n hieupcvp python research/trace_regression_main.py --help`
- [x] Smoke run with W&B disabled succeeds: `conda run -n hieupcvp python research/trace_regression_main.py --smoke 1 --epochs 1 --disable-wandb`
- [x] Smoke run with W&B enabled initializes and finishes a run cleanly.
- [x] Epoch and final metrics appear in run logs with stable metric names.

#### Manual Verification:
- [ ] W&B dashboard shows one epoch-series per epoch (no duplicate epoch-step inflation).
- [ ] Logged config fields match CLI values for a sample run.
- [ ] Final val/test metrics in W&B match `metrics_regression.json`.

**Implementation Note**: After Phase 1 and automated checks pass, pause for manual dashboard confirmation before Phase 2.

---

## Phase 2: Prevent Artifact Overwrite Across Matrix Runs

### Overview
Ensure matrix runs do not overwrite each other’s outputs.

### Changes Required

#### 1. Add run output isolation strategy
**File**: `research/trace_regression_main.py` and/or `research/trace_regression_config.py`
**Changes**:
- Add optional output controls:
  - `--output-dir` (explicit absolute/relative path)
  - `--run-tag` (optional suffix)
- Compute active result dir at runtime:
  - if `--output-dir` provided, write there
  - else keep current default for backward compatibility
- Keep filenames (`predictions_regression.npz`, `metrics_regression.json`) stable inside each run directory.

```python
active_results_dir = Path(args.output_dir) if args.output_dir else cfg.results_dir
active_results_dir.mkdir(parents=True, exist_ok=True)
```

#### 2. Include run metadata in saved JSON
**File**: `research/trace_regression_main.py`
**Changes**:
- Add fields for:
  - run name/tag
  - timestamp
  - active output directory
  - path remap args used

### Success Criteria

#### Automated Verification:
- [x] Two back-to-back runs with different `--output-dir` generate separate artifact folders.
- [x] Existing invocation without `--output-dir` still writes to current default location.
- [x] `metrics_regression.json` contains run metadata fields.

#### Manual Verification:
- [ ] Directory layout is easy to inspect for run comparison.
- [ ] No accidental overwrite occurs when executing the full matrix.

**Implementation Note**: After Phase 2 and automated checks pass, pause for confirmation that run directory layout is acceptable.

---

## Phase 3: Add Bash Matrix Runner for Standard TRACE Runs

### Overview
Add a dedicated bash script in the research folder to execute the four guide-defined runs consistently.

### Changes Required

#### 1. Create matrix script for smoke/huber/mse/no-tabular
**File**: `research/run_trace_regression_matrix.sh`
**Changes**:
- Add strict shell mode: `set -euo pipefail`
- Resolve repository root dynamically.
- Use `conda run -n hieupcvp python ...` for all run invocations.
- Implement helper `run_cfg <name> <extra args...>`.
- Implement four configs based on `guide.md:55-74`:
  - smoke: `--smoke 1 --epochs 1`
  - huber full: `--epochs 100 --loss huber`
  - mse full: `--epochs 100 --loss mse`
  - no-tabular: `--epochs 100 --loss huber --disable-tabular 1`
- Pass per-run `--wandb-run-name` and unique `--output-dir`.

```bash
run_cfg "trace_smoke" --smoke 1 --epochs 1
run_cfg "trace_huber" --epochs 100 --loss huber
run_cfg "trace_mse" --epochs 100 --loss mse
run_cfg "trace_huber_no_tab" --epochs 100 --loss huber --disable-tabular 1
```

#### 2. Optional script controls
**File**: `research/run_trace_regression_matrix.sh`
**Changes**:
- Optional env-vars:
  - `WANDB_PROJECT`
  - `WANDB_ENTITY`
  - `TRACE_PATH_REMAP_FROM`
  - `TRACE_PATH_REMAP_TO`
  - `RUNS_BASE_DIR`
- Log start/end timestamps for each run.

### Success Criteria

#### Automated Verification:
- [x] Script syntax passes: `bash -n research/run_trace_regression_matrix.sh`
- [x] Dry/smoke subset runs successfully and creates run directories.
- [x] Script runs with `--disable-wandb` path and without W&B credentials.

#### Manual Verification:
- [ ] Script output logs clearly indicate which config is running.
- [ ] W&B run names match script config names for enabled runs.
- [ ] Produced run folders are understandable for downstream comparison.

**Implementation Note**: After Phase 3 and automated checks pass, pause for human confirmation before full 100-epoch matrix execution.

---

## Phase 4: Documentation and Operational Hand-off

### Overview
Document the updated execution flow and expected artifacts.

### Changes Required

#### 1. Update guide with tracking/script usage
**File**: `research/markdown/guide.md`
**Changes**:
- Add section: "W&B logging options".
- Add section: "Matrix runner script".
- Include examples with and without W&B.
- Clarify output location behavior with `--output-dir`.

#### 2. Add short implementation record
**File**: `research/markdown/implemented.md`
**Changes**:
- Add concise changelog entry for:
  - W&B integration points
  - new matrix script
  - output isolation behavior

### Success Criteria

#### Automated Verification:
- [x] All documented commands are executable in `hieupcvp` env.
- [x] No broken command examples in markdown.

#### Manual Verification:
- [ ] A teammate can run one command from the guide without additional tribal knowledge.
- [ ] Documentation clearly differentiates smoke vs full matrix usage.

**Implementation Note**: After Phase 4 completion, pause for final human review and sign-off.

---

## Testing Strategy

### Unit Tests:
- Argument parsing for new W&B/output flags.
- Metric payload formatting helper behavior.
- Output directory resolution function behavior.

### Integration Tests:
- 1-epoch smoke with W&B disabled.
- 1-epoch smoke with W&B enabled.
- Matrix script smoke subset with per-run output directories.

### Manual Testing Steps:
1. Run smoke with `--disable-wandb` and verify local artifacts.
2. Run smoke with W&B enabled and verify dashboard metrics/config.
3. Run matrix script and confirm four named run directories plus expected JSON/NPZ artifacts.

## Performance Considerations

- W&B callback logging should remain lightweight (epoch-level, not per-batch) to avoid overhead.
- Run matrix defaults (100 epochs) can be expensive; keep smoke mode as first gate.
- Avoid unnecessary artifact uploads for very large files unless explicitly needed.

## Migration Notes

- Backward compatibility preserved for current direct CLI usage.
- Default output path remains `research/results_trace_regression` unless `--output-dir` is provided.
- Existing old artifacts remain valid and do not require migration.

## References

- Existing TRACE CLI and train/eval flow:
  - `research/trace_regression_main.py:15-25`
  - `research/trace_regression_main.py:143-283`
- Existing output write paths:
  - `research/trace_regression_main.py:251-257`
  - `research/trace_regression_main.py:277-279`
  - `research/trace_regression_config.py:47`
- Run matrix command source:
  - `research/markdown/guide.md:55-74`
- Generator behavior relevant to run accounting:
  - `research/trace_data_generator.py:40-41`
  - `research/trace_regression_main.py:246-249`
- W&B and run-script patterns in this repository:
  - `code/proposal/experiment2/run_mrs_v2.py`
  - `code/proposal/experiment1/run_mrs_v1.py:888-946`
  - `code/proposal/experiment1/run_soop_mrs_v1.sh`
  - `code/proposal/experiment2/run_soop_mrs_v2.sh`
