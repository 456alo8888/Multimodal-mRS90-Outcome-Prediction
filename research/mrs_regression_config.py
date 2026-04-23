from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE_MODEL_DIR = REPO_ROOT / "code" / "baseline" / "Multimodal-mRS90-Outcome-Prediction" / "python" / "model"

experiment_mode = "trace_regression"
trace_split_dir = REPO_ROOT / "code" / "datasets" / "fold_nonstripped_synthetic_mask" / "MRS"
trace_target = "gs_rankin_6isdeath"
trace_subject_id_col = "subject_id"

trace_tabular_continuous = ["age", "bmi", "nihss"]
trace_tabular_categorical = [
    "sex",
    "race",
    "priorstroke",
    "etiology_1",
    "etiology_2",
    "etiology_3",
    "etiology_4",
    "etiology_5",
]
trace_tabular_all = [
    "sex",
    "age",
    "nihss",
    "race",
    "priorstroke",
    "bmi",
    "etiology_1",
    "etiology_2",
    "etiology_3",
    "etiology_4",
    "etiology_5",

]

trace_image_shape = (224, 224, 26)
trace_timepoints = 1
trace_batch_size = 1

trace_train_csv = trace_split_dir / "train.csv"
trace_valid_csv = trace_split_dir / "valid.csv"
trace_test_csv = trace_split_dir / "test.csv"

results_dir = Path(__file__).resolve().parent / "results_trace_regression"

random_seed = 42
n_epochs = 10
learning_rate = 0.001
loss_name = "huber"

transformer_params = {
    "n_layers": 1,
    "n_heads": 8,
    "dropout_rate": 0.2,
}
