"""Wearable-only feature extraction utilities for DREAMT-style recordings."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import pandas as pd


def _signal_vector(values: Sequence[float]) -> list[float]:
    """Return a float list from a sequence of values, raising on invalid input."""
    if not values:
        raise ValueError("No values were provided for feature computation.")
    return [float(value) for value in values]


def compute_hrv_features(
    modality_data: dict[str, Sequence[tuple[str, float]]],
) -> dict[str, float | bool | None]:
    """Compute HRV features from beat-to-beat intervals when available.

    Units: HR is in beats per minute (BPM); RMSSD and SDNN are in milliseconds.
    This function intentionally returns None when beat-to-beat data is insufficient.
    We only consider HRV valid if explicit RR/IBI-style interval series are present;
    HR values alone are insufficient for HRV recovery.
    """
    rr_series = modality_data.get("rr") or modality_data.get("ibi") or modality_data.get("hrv")
    if rr_series is None:
        return {"hrv_available": False, "rmssd": None, "sdnn": None}

    values = [float(value) for _, value in rr_series]
    if len(values) < 2:
        return {"hrv_available": False, "rmssd": None, "sdnn": None}

    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    if not diffs:
        return {"hrv_available": False, "rmssd": None, "sdnn": None}

    rmssd = sqrt(sum(diff * diff for diff in diffs) / len(diffs))
    mean_value = sum(values) / len(values)
    sdnn = sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
    return {"hrv_available": True, "rmssd": rmssd, "sdnn": sdnn}


def build_wearable_feature_table(record: dict) -> pd.DataFrame:
    """Construct a wearable-only feature table for a single recording.

    This intentionally excludes EEG and PSG columns from wearable-model inputs.
    """
    modality_data = record.get("modality_data", {})
    feature_rows: list[dict[str, float | str]] = []

    accel_x = [float(value) for _, value in modality_data.get("accel_x", [])]
    accel_y = [float(value) for _, value in modality_data.get("accel_y", [])]
    accel_z = [float(value) for _, value in modality_data.get("accel_z", [])]
    hr_values = [float(value) for _, value in modality_data.get("hr", [])]

    if not accel_x and not hr_values:
        raise ValueError("Recording has no wearable features available.")

    magnitudes = []
    for i in range(min(len(accel_x), len(accel_y), len(accel_z))):
        mag = (accel_x[i] ** 2 + accel_y[i] ** 2 + accel_z[i] ** 2) ** 0.5
        magnitudes.append(mag)

    row = {
        "subject_id": record.get("subject_id", "unknown_subject"),
        "recording_id": record.get("recording_id", "unknown_record"),
        "hr_mean": float(sum(hr_values) / len(hr_values)) if hr_values else float("nan"),
        "accel_magnitude_mean": (
            float(sum(magnitudes) / len(magnitudes)) if magnitudes else float("nan")
        ),
        "accel_magnitude_std": (
            float(pd.Series(magnitudes).std(ddof=0)) if magnitudes else float("nan")
        ),
    }
    feature_rows.append(row)

    dataframe = pd.DataFrame(feature_rows)
    if "eeg_fpz_cz" in modality_data:
        eeg_columns = [col for col in dataframe.columns if col.startswith("eeg_")]
        dataframe = dataframe.drop(columns=eeg_columns, errors="ignore")
    return dataframe


__all__ = ["build_wearable_feature_table", "compute_hrv_features"]
