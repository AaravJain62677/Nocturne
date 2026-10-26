"""PSG-only feature utilities kept out of wearable-model input paths."""

from __future__ import annotations

from collections.abc import Sequence


def compute_eeg_band_stats(
    modality_data: dict[str, Sequence[tuple[str, float]]],
) -> dict[str, float | None]:
    """Compute a minimal EEG summary for evaluation reference use only.

    This module is intentionally restricted to PSG reference data and must not be
    used as a wearable feature source.
    """
    eeg = modality_data.get("eeg_fpz_cz")
    if eeg is None:
        return {"eeg_mean": None, "eeg_std": None}

    values = [float(value) for _, value in eeg]
    if not values:
        return {"eeg_mean": None, "eeg_std": None}

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {"eeg_mean": mean_value, "eeg_std": variance ** 0.5}


__all__ = ["compute_eeg_band_stats"]
