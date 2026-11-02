"""Wearable-only feature extraction utilities for DREAMT-style recordings."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from statistics import median, pstdev

import pandas as pd


def _to_float(value: object) -> float | None:
    """Convert a numeric payload to a float or return None for null-like values."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _series_values(series: Sequence[tuple[str, float | str]] | None) -> list[float]:
    """Return numeric values from a timestamp/value sequence."""
    if not series:
        return []
    values: list[float] = []
    for _, value in series:
        numeric = _to_float(value)
        if numeric is not None:
            values.append(float(numeric))
    return values


def _slice_time_window(
    series: Sequence[tuple[str, float | str]] | None,
    start: float | str,
    end: float | str,
) -> list[float]:
    """Return values whose timestamps fall into the epoch time window."""
    if not series:
        return []
    start_value = _to_float(start)
    end_value = _to_float(end)
    if start_value is None or end_value is None:
        return []
    values: list[float] = []
    for raw_ts, raw_value in series:
        ts = _to_float(raw_ts)
        value = _to_float(raw_value)
        if ts is None or value is None:
            continue
        if start_value <= ts < end_value:
            values.append(float(value))
    return values


def compute_hrv_features(
    modality_data: dict[str, Sequence[tuple[str, float]]],
) -> dict[str, float | bool | None]:
    """Compute HRV features from beat-to-beat intervals when available."""
    rr_series = modality_data.get("rr") or modality_data.get("ibi") or modality_data.get("hrv")
    if rr_series is None:
        return {"hrv_available": False, "rmssd": None, "sdnn": None}

    values = _series_values(rr_series)
    if len(values) < 2:
        return {"hrv_available": False, "rmssd": None, "sdnn": None}

    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    if not diffs:
        return {"hrv_available": False, "rmssd": None, "sdnn": None}

    rmssd = sqrt(sum(diff * diff for diff in diffs) / len(diffs))
    mean_value = sum(values) / len(values)
    sdnn = sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
    return {"hrv_available": True, "rmssd": rmssd, "sdnn": sdnn}


def extract_wearable_epoch_features(
    record: dict | object,
    epoch_index: int = 0,
    epoch_seconds: int = 30,
    time_origin: float | None = None,
    epoch_start: float | None = None,
    epoch_end: float | None = None,
) -> dict[str, object]:
    """Compute wearable-only features for one epoch while ignoring PSG-derived data."""
    if hasattr(record, "modality_data"):
        modality_data = getattr(record, "modality_data")
    else:
        modality_data = record.get("modality_data", {})

    if epoch_start is None:
        if time_origin is None:
            time_origin = 0.0
        epoch_start = time_origin + epoch_index * epoch_seconds
    if epoch_end is None:
        epoch_end = epoch_start + epoch_seconds

    available: dict[str, bool] = {
        "hr": "hr" in modality_data,
        "ibi": "ibi" in modality_data or "rr" in modality_data,
        "accel_x": "accel_x" in modality_data,
        "accel_y": "accel_y" in modality_data,
        "accel_z": "accel_z" in modality_data,
        "eda": "eda" in modality_data,
    }

    features: dict[str, object] = {
        "modality_availability": available,
        "missingness": {},
    }

    hr_values = _slice_time_window(modality_data.get("hr"), epoch_start, epoch_end)
    if hr_values:
        features["hr_mean"] = float(sum(hr_values) / len(hr_values))
        features["hr_median"] = float(median(hr_values))
        features["hr_std"] = float(pstdev(hr_values)) if len(hr_values) > 1 else 0.0
        features["hr_min"] = float(min(hr_values))
        features["hr_max"] = float(max(hr_values))
        features["hr_valid_count"] = len(hr_values)
        features["hr_missingness"] = 0.0
    else:
        features["hr_mean"] = None
        features["hr_median"] = None
        features["hr_std"] = None
        features["hr_min"] = None
        features["hr_max"] = None
        features["hr_valid_count"] = 0
        features["hr_missingness"] = 1.0
    features["missingness"]["hr"] = 1.0 if not hr_values else 0.0

    ibi_values = _slice_time_window(modality_data.get("ibi"), epoch_start, epoch_end)
    if not ibi_values:
        ibi_values = _slice_time_window(modality_data.get("rr"), epoch_start, epoch_end)
    if ibi_values:
        hrv = compute_hrv_features({"ibi": [("0", value) for value in ibi_values]})
        features["hrv_available"] = bool(hrv["hrv_available"])
        features["hrv_rmssd"] = hrv["rmssd"]
        features["hrv_sdnn"] = hrv["sdnn"]
    else:
        features["hrv_available"] = False
        features["hrv_rmssd"] = None
        features["hrv_sdnn"] = None
    features["missingness"]["ibi"] = 1.0 if not ibi_values else 0.0

    accel_x = _slice_time_window(modality_data.get("accel_x"), epoch_start, epoch_end)
    accel_y = _slice_time_window(modality_data.get("accel_y"), epoch_start, epoch_end)
    accel_z = _slice_time_window(modality_data.get("accel_z"), epoch_start, epoch_end)
    for axis_name, values in {"accel_x": accel_x, "accel_y": accel_y, "accel_z": accel_z}.items():
        if values:
            features[f"{axis_name}_mean"] = float(sum(values) / len(values))
            features[f"{axis_name}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
            features[f"{axis_name}_valid_count"] = len(values)
            features[f"{axis_name}_missingness"] = 0.0
        else:
            features[f"{axis_name}_mean"] = None
            features[f"{axis_name}_std"] = None
            features[f"{axis_name}_valid_count"] = 0
            features[f"{axis_name}_missingness"] = 1.0
        features["missingness"][axis_name] = 1.0 if not values else 0.0

    magnitudes = []
    for index in range(min(len(accel_x), len(accel_y), len(accel_z))):
        mag = sqrt(accel_x[index] ** 2 + accel_y[index] ** 2 + accel_z[index] ** 2)
        magnitudes.append(mag)
    if magnitudes:
        features["accel_magnitude_mean"] = float(sum(magnitudes) / len(magnitudes))
        features["accel_magnitude_std"] = float(pstdev(magnitudes)) if len(magnitudes) > 1 else 0.0
        features["movement_intensity"] = float(sum(magnitudes) / len(magnitudes))
        features["accel_valid_count"] = len(magnitudes)
    else:
        features["accel_magnitude_mean"] = None
        features["accel_magnitude_std"] = None
        features["movement_intensity"] = None
        features["accel_valid_count"] = 0
    features["missingness"]["accel"] = 1.0 if not magnitudes else 0.0

    eda_values = _slice_time_window(modality_data.get("eda"), epoch_start, epoch_end)
    if eda_values:
        features["eda_mean"] = float(sum(eda_values) / len(eda_values))
        features["eda_std"] = float(pstdev(eda_values)) if len(eda_values) > 1 else 0.0
        features["eda_min"] = float(min(eda_values))
        features["eda_max"] = float(max(eda_values))
        features["eda_valid_count"] = len(eda_values)
        features["eda_missingness"] = 0.0
        if len(eda_values) >= 8:
            baseline = median(eda_values)
            phasic = [max(0.0, value - baseline) for value in eda_values]
            features["eda_tonic_mean"] = float(baseline)
            features["eda_phasic_mean"] = float(sum(phasic) / len(phasic))
        else:
            features["eda_tonic_mean"] = None
            features["eda_phasic_mean"] = None
    else:
        features["eda_mean"] = None
        features["eda_std"] = None
        features["eda_min"] = None
        features["eda_max"] = None
        features["eda_valid_count"] = 0
        features["eda_missingness"] = 1.0
        features["eda_tonic_mean"] = None
        features["eda_phasic_mean"] = None
    features["missingness"]["eda"] = 1.0 if not eda_values else 0.0

    return features


def build_wearable_feature_table(record: dict) -> pd.DataFrame:
    """Construct a wearable-only feature table for a single recording."""
    modality_data = record.get("modality_data", {})
    feature_rows: list[dict[str, float | str | None]] = []

    hr_values = _series_values(modality_data.get("hr"))
    accel_x = _series_values(modality_data.get("accel_x"))
    accel_y = _series_values(modality_data.get("accel_y"))
    accel_z = _series_values(modality_data.get("accel_z"))
    if not hr_values and not accel_x and not accel_y and not accel_z:
        raise ValueError("Recording has no wearable features available.")

    magnitudes = []
    max_len = max(len(accel_x), len(accel_y), len(accel_z))
    for i in range(min(len(accel_x), len(accel_y), len(accel_z), max_len)):
        if i >= len(accel_x) or i >= len(accel_y) or i >= len(accel_z):
            continue
        mag = sqrt(accel_x[i] ** 2 + accel_y[i] ** 2 + accel_z[i] ** 2)
        magnitudes.append(mag)

    row: dict[str, float | str | None] = {
        "subject_id": record.get("subject_id", "unknown_subject"),
        "recording_id": record.get("recording_id", "unknown_record"),
    }
    if hr_values:
        row["hr_mean"] = float(sum(hr_values) / len(hr_values))
        row["hr_std"] = float(pstdev(hr_values)) if len(hr_values) > 1 else 0.0
    else:
        row["hr_mean"] = float("nan")
        row["hr_std"] = float("nan")
    if magnitudes:
        row["accel_magnitude_mean"] = float(sum(magnitudes) / len(magnitudes))
        row["accel_magnitude_std"] = float(pstdev(magnitudes)) if len(magnitudes) > 1 else 0.0
    else:
        row["accel_magnitude_mean"] = float("nan")
        row["accel_magnitude_std"] = float("nan")
    row.update(
        compute_hrv_features(
            {"ibi": [("0", value) for value in _series_values(modality_data.get("ibi"))]}
        )
    )
    feature_rows.append(row)

    dataframe = pd.DataFrame(feature_rows)
    if "eeg_fpz_cz" in modality_data:
        eeg_columns = [col for col in dataframe.columns if col.startswith("eeg_")]
        dataframe = dataframe.drop(columns=eeg_columns, errors="ignore")
    return dataframe


__all__ = [
    "build_wearable_feature_table",
    "compute_hrv_features",
    "extract_wearable_epoch_features",
]
