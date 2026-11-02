# Project Status

## Current implementation status

This repository is a Phase 2 validation-and-ingestion milestone for a wearable-versus-PSG sleep-staging project. It remains a guarded research scaffold, not a complete end-to-end experimental pipeline.

### Verified in the repository

- Repository root resolution is derived from the source tree rather than a hardcoded path.
- Dataset registry and sensor-set validation are explicit and enforce supported values.
- Canonical sleep-stage normalization and secondary taxonomy mappings are implemented.
- DREAMT discovery and record parsing are validated against synthetic fixtures and real repository-level CLI checks.
- The dataset-inspection entry point, `python -m src.inspect_data --dataset dreamt ...`, runs successfully on synthetic DREAMT-style records.
- The synthetic wearable preprocessing artifact now serializes deterministically to Parquet (`.parquet`) with a companion metadata JSON file that records schema, counts, exclusions, and signal availability.
- Project hygiene checks pass: `python -m ruff check src tests` and `python -m pytest tests -q`.

### Synthetic artifact contract

The deterministic artifact output uses the pattern:

- `<output_dir>/<dataset_id>_wearable_epochs.parquet`
- `<output_dir>/<dataset_id>_wearable_epochs_metadata.json`

The serialized row schema includes:

- `subject_id`, `recording_id`, `epoch_index`, `epoch_start`, `epoch_end`
- `reference_stage`, `excluded`, `exclusion_reason`, `preprocessing_version`
- wearable feature columns such as `feature_hr_mean`, `feature_hrv_rmssd`, and `feature_accel_magnitude_mean`
- modality flags such as `modality_available_hr`, `modality_available_accel_x`, and `modality_available_eda`
- optional missingness columns for per-epoch quality metadata

The JSON metadata contains:

- `preprocessing_version`
- `source_dataset_id`
- `generation_timestamp_utc`
- `config` with dataset configuration and epoch duration
- `epoch_duration_seconds`
- `feature_schema`
- `counts` for subjects, recordings, epochs, and excluded epochs
- `excluded_epoch_counts`
- `modality_availability`
- `artifact_format`
- `artifact_path`
- `real_data_validation_pending`
- `no_eeg_leakage_guarantee`

### Verified only with synthetic fixtures

The following assumptions are not yet backed by real DREAMT recordings:

- Subject/record directory layout is assumed to be `subject_id / recording_id`.
- Signal files are assumed to be individual CSV/TSV files per modality, e.g. `accel_x.csv`, `hr.csv`, `stage_labels.csv`.
- Timestamps are assumed to be in a per-row numeric or string format compatible with the parser’s direct pass-through.
- Sampling rates are inferred from timestamp deltas rather than read from canonical dataset metadata files.
- Missing wearable/PSG modalities are treated as explicit validation failures, but this has only been checked on synthetic fixtures.
- PSG stage labels are assumed to be reference-only annotations and are not yet validated on a real DREAMT dataset.
- The `combined` sensor-set workflow is intended to keep wearable and PSG signals separate internally; this has not been validated against real recordings.

### Real-data validation status

Real DREAMT validation remains pending because the repository does not currently contain a local DREAMT dataset.

Evidence from the repository:

- `find data -maxdepth 3 -print` shows only placeholder directories: `data/raw`, `data/processed`, and `data/splits`, each containing only `.gitkeep`.
- A search under `/home/aaravjain62677` did not reveal a local DREAMT dataset copy.

Because the actual recordings are absent, the following actions remain unverified:

- Subject discovery on a real dataset.
- Full recording load and alignment checks.
- Real wearable modality inventory and actual sampling rates.
- Real PSG annotation count and label distribution.
- Metadata compatibility with unanticipated file naming conventions or channel naming variations.
- Unsupported or extra modalities inside an actual DREAMT release.

### Blockers for Milestone 3

- No real DREAMT recording files are present locally; therefore no scientific statement about dataset compatibility may be made.
- Access to the upstream DREAMT dataset must be confirmed before any real-data validation or preprocessing can be attempted.
- Dataset-specific filenames, label conventions, and signal metadata must be checked against the actual downloaded release before divergence from the synthetic assumptions is considered safe.

### Scientific guardrails

- PSG labels remain reference-only metadata unless a future analysis explicitly uses them as supervised targets.
- Wearable and PSG signals stay separated in the validation contract; they are not merged by default.
- Timestamp units and reference conventions must be preserved exactly for later alignment logic.
- Missing modalities must produce explicit validation errors, not silent drops.
- Synthetic tests are useful for contract enforcement, but they do not establish real-data compatibility.

## Milestone 3 preprocessing audit

### Confirmed implementations

- Epoch alignment is implemented in `src/preprocessing/epoch_segmentation.py` and enforces an explicit 30-second window contract keyed to the recording time origin.
- Label handling preserves `UNKNOWN`, `UNSCORED`, and `MOVEMENT` as exclusions rather than converting them into valid training targets.
- Duplicate timestamps are counted deterministically and retained in the metadata output instead of being silently folded away.
- Wearable-only feature extraction is implemented in `src/preprocessing/watch_features.py` and keeps EEG/PSG-derived signals out of the feature matrix.
- EEG summary statistics remain in `src/preprocessing/eeg_features.py` and are explicitly restricted to PSG reference use.
- The synthetic preprocessing entry point in `src/preprocess.py` and the package export in `src/preprocessing/__init__.py` return deterministic epoch-level outputs with explicit metadata.

### Mathematical feature definitions

HR features:
- HR mean: $\mu_{HR} = \frac{1}{n}\sum_{i=1}^{n} HR_i$
- HR median: median of valid HR samples in the 30-second window
- HR std: population standard deviation over valid HR samples
- HR min/max: minima and maxima of valid HR samples
- HR valid count: count of non-null samples inside the epoch; missingness equals 1 when no valid HR samples are present

IBI/RR features:
- RMSSD: $\sqrt{\frac{1}{n-1}\sum_{i=1}^{n-1}(RR_{i+1}-RR_i)^2}$
- SDNN: $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(RR_i-\mu_{RR})^2}$
- The implementation computes HRV only from IBI/RR intervals, never from averaged HR values. If fewer than two valid intervals are present, HRV is reported as unavailable.

Accelerometry features:
- axis means and standard deviations are computed over valid samples in the window
- vector magnitude per sample: $\sqrt{x^2 + y^2 + z^2}$
- movement intensity is the mean magnitude over the epoch; accelerometer features are computed from the raw accelerometer axes only and do not import EEG channels

EDA features:
- EDA mean/std/min/max are computed over valid samples in the epoch
- tonic estimate is the median of the EDA window, $T = median(EDA)$
- phasic estimate is the mean of positive deviations from tonic, $\frac{1}{n}\sum_i max(0, EDA_i - T)$
- tonic/phasic summaries are only emitted when there are enough valid EDA samples; otherwise they remain `None`

### Scientific limitations

- This implementation is synthetic-only and has not been validated against real DREAMT recordings.
- No real-data timing, label conventions, or modality naming were checked against an actual DREAMT release.
- EDA tonic/phasic decomposition is implemented as a heuristic summary, not a validated physiology model.
- No normalization or imputation is fitted from the data at this stage, which is correct for a frozen train/validation split design but not a production-ready clinical pipeline.
- The current pipeline is intentionally a preprocessing scaffold; it does not claim model-quality evidence or real-world generalization.

### Outstanding real-data validation

Real DREAMT validation remains pending because the repository does not currently contain a local DREAMT dataset. The pipeline is therefore validated on synthetic fixtures only and must not be treated as scientifically verified on actual data.

### Issues fixed during the audit

- Epoch-start semantics and offset handling were corrected so recording-relative boundaries remain stable and do not silently drift.
- Duplicate timestamps are counted and deterministically handled rather than folded away without traceability.
- Malformed or unsupported stage values are explicitly treated as data-quality issues, while valid unknown labels remain explicit exclusions.
- PSG/EEG features are kept out of wearable predictors and are reported separately.
- The preprocessing output now includes an explicit metadata contract with `preprocessing_version` and configuration details for deterministic synthetic output traceability.

### Final status

The repository is in a stable synthetic validation state for the preprocessing scaffold. It is not yet ready to claim real DREAMT scientific validity or to proceed to model training without an actual dataset and downstream validation checks.
