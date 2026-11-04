# Nocturne architecture overview

This document describes the repository’s current engineering architecture and its scientific boundaries.

## End-to-end flow

Dataset → Ingestion → Epoch segmentation → Wearable feature extraction → Subject split → Training → Evaluation → Results

## Components

### 1. Dataset and ingestion
The dataset layer validates supported identifiers and sensor configurations before any experimental work begins. The repository explicitly accepts the dataset IDs and sensor sets defined in the central configuration and rejects unsupported combinations early.

This layer is intentionally limited to explicit validation. It does not invent a real DREAMT schema or silently coerce files into a format that has not been verified against the actual upstream release.

### 2. Epoch segmentation
The preprocessing stage aligns signals into fixed-length time windows, preserves subject and recording identity, and excludes unsupported or malformed epochs explicitly. It keeps a record of exclusion reasons and does not silently convert unknown or movement labels into valid training targets.

### 3. Wearable feature extraction
The wearable feature path derives signal summaries from HR, HRV, accelerometry, and EDA. Features are created per epoch and stored with provenance fields describing availability, validity, and missingness.

This is intentionally separate from PSG/EEG processing, which remains a reference-only supervision path.

### 4. Subject split
The split layer forms subject-independent train/validation/test partitions. Subject IDs are assigned to partitions as whole entities, never by individual epochs, to reduce leakage risk.

### 5. Training
The model layer consumes the feature matrix defined by the selected feature group. Feature selection is explicit and constrained to the actual artifact schema; no silent fallback to EEG/PSG predictors is allowed in wearable-only mode.

### 6. Evaluation
Evaluation computes classification metrics, per-subject summaries, confusion matrices, and optional ablation results. It preserves taxonomy boundaries and reports undefined metrics explicitly rather than pretending they are valid scientific quantities.

### 7. Results and artifact export
The pipeline writes a coherent output directory containing summary metadata, evaluation metrics, subject-level metrics, confusion matrices, and split manifests. Output artifacts are designed to support reproducibility and auditability without overwriting prior runs silently.

## Scientific boundary

PSG and EEG signals are reference supervision and ground-truth context, not wearable model inputs. The pipeline does not permit the wearable feature gate to consume PSG-derived features unless the explicit contract and the data schema support that workflow.

The repository has verified this contract on synthetic fixtures and code-level validation. Real DREAMT validation remains pending external access to the authorized dataset.
