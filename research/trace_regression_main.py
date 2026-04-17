import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import trace_regression_config as cfg
from tensorflow import keras



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=cfg.experiment_mode, choices=["trace_regression"])
    parser.add_argument("--epochs", type=int, default=cfg.n_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.trace_batch_size)
    parser.add_argument("--loss", choices=["huber", "mse"], default=cfg.loss_name)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--disable-tabular", type=int, default=0)
    parser.add_argument("--path-remap-from", type=str, default="/mnt/disk1")
    parser.add_argument("--path-remap-to", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _mode(series: pd.Series, default: float = 0.0) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return float(default)
    return float(cleaned.mode().iloc[0])


def fit_tabular_preprocess(train_df: pd.DataFrame, continuous_cols, categorical_cols):
    defaults = {}
    for col in continuous_cols:
        vals = pd.to_numeric(train_df[col], errors="coerce") if col in train_df.columns else pd.Series(dtype=float)
        defaults[col] = float(vals.median()) if not vals.dropna().empty else 0.0
    for col in categorical_cols:
        vals = pd.to_numeric(train_df[col], errors="coerce") if col in train_df.columns else pd.Series(dtype=float)
        defaults[col] = _mode(vals, default=0.0)

    cat_maps = {}
    cat_sizes = []
    for col in categorical_cols:
        vals = pd.to_numeric(train_df[col], errors="coerce") if col in train_df.columns else pd.Series(dtype=float)
        vals = vals.fillna(defaults[col]).astype(float)
        uniq = sorted(vals.unique().tolist())
        mapping = {v: i for i, v in enumerate(uniq)}
        cat_maps[col] = mapping
        cat_sizes.append(max(1, len(mapping)))

    return defaults, cat_maps, cat_sizes


def build_model(image_shape, continuous_cols, categorical_cols, categorical_sizes, disable_tabular=False):
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Input
    from tensorflow.keras.regularizers import l2

    if str(cfg.BASELINE_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(cfg.BASELINE_MODEL_DIR))
    from utils import ENCODER, MultimodalFusion, SelfAttention

    base_network = ENCODER.build(reg=l2(0.00005), shape=image_shape)

    trace_input = Input(shape=(*image_shape, 1), name="TRACE")
    trace_output = base_network(trace_input)[0]
    imaging_encoded = trace_output

    self_imaging = SelfAttention.build(
        imaging_encoded,
        num_heads=cfg.transformer_params["n_heads"],
        num_layers=cfg.transformer_params["n_layers"],
        dropout=cfg.transformer_params["dropout_rate"],
    )

    model_inputs = ((trace_input,), tuple(), tuple())

    if disable_tabular:
        features = self_imaging[:, 0]
    else:
        embed_dim = int(imaging_encoded.shape[-1])

        C_inputs = []
        C_embedding_outputs = []
        for idx, feature_name in enumerate(categorical_cols):
            categorical_i = Input(shape=(1,), dtype="int32", name=feature_name)
            categorical_i_ = Embedding(input_dim=categorical_sizes[idx], output_dim=embed_dim)(categorical_i)
            C_inputs.append(categorical_i)
            C_embedding_outputs.append(categorical_i_)
        categorical_inputs = Concatenate(axis=1)(C_embedding_outputs)

        N_inputs = []
        N_embedding_outputs = []
        for feature_name in continuous_cols:
            continuous_i = Input(shape=(1,), dtype="float32", name=feature_name)
            continuous_i_ = Dense(embed_dim, activation="relu")(continuous_i)
            N_inputs.append(continuous_i)
            N_embedding_outputs.append(
                keras.layers.Lambda(lambda x: keras.ops.expand_dims(x, axis=1))(continuous_i_)
            )
        continuous_inputs = Concatenate(axis=1)(N_embedding_outputs)

        metadata_encoded = Concatenate(axis=1)([continuous_inputs, categorical_inputs])
        self_metadata = SelfAttention.build(
            metadata_encoded,
            num_heads=cfg.transformer_params["n_heads"],
            num_layers=cfg.transformer_params["n_layers"],
            dropout=cfg.transformer_params["dropout_rate"],
        )

        features = MultimodalFusion.build(
            self_imaging,
            self_metadata,
            num_heads=cfg.transformer_params["n_heads"],
            embed_dim=embed_dim,
            dropout=cfg.transformer_params["dropout_rate"],
        )
        model_inputs = ((trace_input,), tuple(C_inputs), tuple(N_inputs))

    mlp_hidden_units_factors = [2, 1]
    mlp_hidden_units = [int(factor * int(features.shape[-1])) for factor in mlp_hidden_units_factors]
    for units in mlp_hidden_units:
        features = Dense(units, activation="relu")(features)
        features = Dropout(0.2)(features)

    prediction = Dense(1, activation="linear", name="gs_rankin_6isdeath")(features)
    model = Model(inputs=model_inputs, outputs=prediction)
    return model


def main():
    args = parse_args()

    import tensorflow as tf
    from tensorflow.keras import optimizers
    from trace_data_generator import DataGenerator_TRACE_Regression
    set_seed(cfg.random_seed)

    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(cfg.trace_train_csv)
    valid_df = pd.read_csv(cfg.trace_valid_csv)
    test_df = pd.read_csv(cfg.trace_test_csv)

    continuous_cols = [] if args.disable_tabular else cfg.trace_tabular_continuous
    categorical_cols = [] if args.disable_tabular else cfg.trace_tabular_categorical

    defaults, cat_maps, cat_sizes = fit_tabular_preprocess(
        train_df=train_df,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )

    if args.smoke:
        train_df = train_df.head(max(args.batch_size * 2, 2)).copy()
        valid_df = valid_df.head(max(args.batch_size * 2, 2)).copy()
        test_df = test_df.head(max(args.batch_size * 2, 2)).copy()

    train_gen = DataGenerator_TRACE_Regression(
        split_df=train_df,
        target_col=cfg.trace_target,
        continuous_features=continuous_cols,
        categorical_features=categorical_cols,
        dim=cfg.trace_image_shape,
        batch_size=args.batch_size,
        shuffle=True,
        path_remap_from=args.path_remap_from,
        path_remap_to=args.path_remap_to,
        defaults=defaults,
        category_maps=cat_maps,
    )
    valid_gen = DataGenerator_TRACE_Regression(
        split_df=valid_df,
        target_col=cfg.trace_target,
        continuous_features=continuous_cols,
        categorical_features=categorical_cols,
        dim=cfg.trace_image_shape,
        batch_size=args.batch_size,
        shuffle=False,
        path_remap_from=args.path_remap_from,
        path_remap_to=args.path_remap_to,
        defaults=defaults,
        category_maps=cat_maps,
    )
    test_gen = DataGenerator_TRACE_Regression(
        split_df=test_df,
        target_col=cfg.trace_target,
        continuous_features=continuous_cols,
        categorical_features=categorical_cols,
        dim=cfg.trace_image_shape,
        batch_size=args.batch_size,
        shuffle=False,
        path_remap_from=args.path_remap_from,
        path_remap_to=args.path_remap_to,
        defaults=defaults,
        category_maps=cat_maps,
    )

    model = build_model(
        image_shape=cfg.trace_image_shape,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        categorical_sizes=cat_sizes,
        disable_tabular=bool(args.disable_tabular),
    )

    if args.loss == "huber":
        loss_fn = tf.keras.losses.Huber(delta=1.0)
    else:
        loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(
        loss=loss_fn,
        optimizer=optimizers.Adam(learning_rate=cfg.learning_rate),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.R2Score(name="r2")
        ],
    )

    history = model.fit(
        x=train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        verbose=1,
    )

    val_eval = model.evaluate(valid_gen, verbose=0)
    test_eval = model.evaluate(test_gen, verbose=0)

    preds = model.predict(test_gen, verbose=0).reshape(-1)

    used_subject_ids = test_gen.ordered_subject_ids_for_steps()
    used_len = len(preds)
    y_true = test_df[cfg.trace_target].astype(np.float32).values[:used_len]
    used_subject_ids = used_subject_ids[:used_len]

    predictions_path = cfg.results_dir / "predictions_regression.npz"
    np.savez_compressed(
        predictions_path,
        pred=preds.astype(np.float32),
        y_true=y_true.astype(np.float32),
        subject_id=np.asarray(used_subject_ids),
    )

    metrics = {
        "loss_name": args.loss,
        "disable_tabular": int(args.disable_tabular),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "val_loss": float(val_eval[0]),
        "val_mae": float(val_eval[1]),
        "val_rmse": float(val_eval[2]),
        "val_r2": float(val_eval[3]),
        "test_loss": float(test_eval[0]),
        "test_mae": float(test_eval[1]),
        "test_rmse": float(test_eval[2]),
        "test_r2": float(test_eval[3]),
        "epochs": int(args.epochs),
        "history_keys": list(history.history.keys()),
    }

    metrics_path = cfg.results_dir / "metrics_regression.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
