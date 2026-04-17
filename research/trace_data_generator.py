from typing import Dict, List, Optional, Tuple

import numpy as np
from tensorflow.keras.utils import Sequence

from trace_data_io import extract_tabular_row, load_trace_nifti, remap_path


class DataGenerator_TRACE_Regression(Sequence):
    def __init__(
        self,
        split_df,
        target_col: str,
        continuous_features: List[str],
        categorical_features: List[str],
        dim: Tuple[int, int, int] = (256, 256, 26),
        batch_size: int = 1,
        shuffle: bool = True,
        path_remap_from: Optional[str] = None,
        path_remap_to: Optional[str] = None,
        defaults: Optional[Dict[str, float]] = None,
        category_maps: Optional[Dict[str, Dict[float, int]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.df = split_df.reset_index(drop=True)
        self.target_col = target_col
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.path_remap_from = path_remap_from
        self.path_remap_to = path_remap_to
        self.defaults = defaults or {}
        self.category_maps = category_maps or {}
        self.subject_ids = self.df["subject_id"].astype(str).tolist()
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(batch_indexes)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def ordered_subject_ids_for_steps(self):
        total = len(self) * self.batch_size
        return [self.subject_ids[i] for i in self.indexes[:total]]

    def __data_generation(self, batch_indexes):
        X = np.empty((self.batch_size, *self.dim, 1), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=np.float32)
        C_continuous = np.empty((self.batch_size, len(self.continuous_features)), dtype=np.float32)
        C_categorical = np.empty((self.batch_size, len(self.categorical_features)), dtype=np.int32)

        for k, idx in enumerate(batch_indexes):
            row = self.df.iloc[idx]
            image_path = remap_path(str(row["image_path"]), self.path_remap_from, self.path_remap_to)
            volume = load_trace_nifti(image_path, out_shape=self.dim)
            X[k, ..., 0] = volume

            cont_vec, cat_vec = extract_tabular_row(
                row=row,
                continuous_cols=self.continuous_features,
                categorical_cols=self.categorical_features,
                defaults=self.defaults,
                category_maps=self.category_maps,
            )
            C_continuous[k, :] = cont_vec
            C_categorical[k, :] = cat_vec
            y[k] = np.float32(row[self.target_col])

        image_inputs = (X,)
        categorical_inputs = (
            tuple(np.split(C_categorical, len(self.categorical_features), axis=1))
            if self.categorical_features
            else tuple()
        )
        continuous_inputs = (
            tuple(np.split(C_continuous, len(self.continuous_features), axis=1))
            if self.continuous_features
            else tuple()
        )
        return (image_inputs, categorical_inputs, continuous_inputs), y
