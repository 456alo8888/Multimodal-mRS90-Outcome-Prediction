import json
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import zoom


def remap_path(path: str, src_prefix: Optional[str], dst_prefix: Optional[str]) -> str:
    if not src_prefix or not dst_prefix:
        return path
    if path.startswith(src_prefix):
        return dst_prefix + path[len(src_prefix):]
    return path


def _center_crop_or_pad(volume: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(out_shape, dtype=np.float32)
    src_slices = []
    dst_slices = []
    for src_size, dst_size in zip(volume.shape, out_shape):
        if src_size >= dst_size:
            src_start = (src_size - dst_size) // 2
            src_end = src_start + dst_size
            dst_start = 0
            dst_end = dst_size
        else:
            src_start = 0
            src_end = src_size
            dst_start = (dst_size - src_size) // 2
            dst_end = dst_start + src_size
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


def load_trace_nifti(path: str, out_shape: Tuple[int, int, int] = (256, 256, 26)) -> np.ndarray:
    import nibabel as nib

    nii = nib.load(path)
    volume = np.asarray(nii.get_fdata(), dtype=np.float32)
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    if volume.shape != out_shape:
        factors = tuple(float(o) / float(s) for s, o in zip(volume.shape, out_shape))
        volume = zoom(volume, zoom=factors, order=1)
        volume = _center_crop_or_pad(volume, out_shape)
    vmin = np.nanmin(volume)
    vmax = np.nanmax(volume)
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    else:
        volume = np.zeros(out_shape, dtype=np.float32)
    return volume.astype(np.float32)


def _parse_tabular_features(raw_json: object) -> Dict[str, float]:
    if raw_json is None or (isinstance(raw_json, float) and np.isnan(raw_json)):
        return {}
    if isinstance(raw_json, dict):
        return raw_json
    if isinstance(raw_json, str):
        text = raw_json.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def extract_tabular_row(
    row: pd.Series,
    continuous_cols: Iterable[str],
    categorical_cols: Iterable[str],
    defaults: Dict[str, float],
    category_maps: Optional[Dict[str, Dict[float, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    parsed = _parse_tabular_features(row.get("tabular_features"))

    cont_values = []
    for col in continuous_cols:
        value = row.get(col, np.nan)
        if pd.isna(value):
            value = parsed.get(col, np.nan)
        if pd.isna(value):
            value = defaults[col]
        cont_values.append(float(value))

    cat_values = []
    for col in categorical_cols:
        value = row.get(col, np.nan)
        if pd.isna(value):
            value = parsed.get(col, np.nan)
        if pd.isna(value):
            value = defaults[col]
        value = float(value)
        if category_maps and col in category_maps:
            value = category_maps[col].get(value, 0)
        cat_values.append(int(value))

    return np.asarray(cont_values, dtype=np.float32), np.asarray(cat_values, dtype=np.int32)
