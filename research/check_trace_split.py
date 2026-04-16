import json
from pathlib import Path

import pandas as pd

import trace_regression_config as cfg


def check_split(path: Path):
    df = pd.read_csv(path)
    required = [cfg.trace_subject_id_col, "image_path", cfg.trace_target]
    missing_required = [c for c in required if c not in df.columns]
    has_tabular_columns = all(c in df.columns for c in cfg.trace_tabular_all)
    has_tabular_json = "tabular_features" in df.columns
    return {
        "file": str(path),
        "rows": int(len(df)),
        "missing_required_columns": missing_required,
        "has_tabular_columns": has_tabular_columns,
        "has_tabular_json": has_tabular_json,
        "missing_counts": {
            c: int(df[c].isna().sum()) for c in cfg.trace_tabular_all if c in df.columns
        },
        "missing_target": int(df[cfg.trace_target].isna().sum()) if cfg.trace_target in df.columns else None,
    }


def main():
    report = {
        "train": check_split(cfg.trace_train_csv),
        "valid": check_split(cfg.trace_valid_csv),
        "test": check_split(cfg.trace_test_csv),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
