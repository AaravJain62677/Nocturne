# Nocturne

**Investigating wearable-based sleep staging against polysomnography (PSG) ground truth.**

Nocturne is a research project exploring how accurately sleep stages can be estimated from wearable-available signals and how model choice and sensor combinations affect performance.

The project is designed to keep the research process explicit and reproducible: dataset contracts are validated before unsupported workflows run, sleep-stage labels are handled consistently, and results are not reported unless they have actually been produced.

> **Current status:** Research pipeline under development. The repository contains project configuration, validation contracts, dataset discovery scaffolding, preprocessing components, and automated tests. End-to-end training, real-data evaluation, and research findings should be considered incomplete until they have been run and verified on real recordings.

## Research Questions

- How well can wearable-derived signals predict PSG-scored sleep stages?
- Which sleep stages are most difficult to distinguish from wearable proxies?
- How do sensor combinations and model families affect performance?
- How much performance can be recovered through improved modeling and feature selection?

These are research questions, not established findings. No performance claims are made until experiments have been executed and evaluated.

## Planned Workflow

1. **Dataset ingestion** — inspect and load supported recordings while preserving subject IDs, recording IDs, timestamps, and source metadata.
2. **Preprocessing** — align PSG labels and wearable signals into 30-second epochs, handle missing or unscored data explicitly, and extract eligible wearable-only features.
3. **Subject-independent splitting** — keep subjects separate across training, validation, and test partitions to reduce leakage.
4. **Modeling** — establish simple baselines before evaluating sequence models.
5. **Evaluation** — report accuracy, Cohen's kappa, macro-F1, per-stage metrics, confusion matrices, and uncertainty where supported by the data.
6. **Ablation and reporting** — compare sensor configurations and generate reproducible tables and figures from actual experiment outputs.

## Datasets

The project plans to investigate:

- **DREAMT (PhysioNet):** intended as the primary source for paired PSG labels and wearable signals. Its actual local file layout, available channels, sampling rates, access requirements, and annotation format must be verified before real-data processing.
- **Sleep-EDF (PhysioNet):** a possible source for developing and validating PSG annotation handling. It should not be treated as a wearable-comparison dataset unless suitable paired wearable signals are available.

Dataset access and usage must follow the relevant dataset terms. Raw recordings should remain local and must not be committed to this repository.
## Official DREAMT acquisition and access policy

The official DREAMT release is hosted on PhysioNet and is a restricted-access resource. The project documentation on the official page states the following conditions, which must be satisfied before any files are downloaded or processed:

- The resource is hosted at https://physionet.org/content/dreamt/ and is currently published under the restricted health-data policy for PhysioNet.
- Access is restricted to registered users who complete the required data use agreement and accept the dataset terms.
- The official dataset license is the PhysioNet Restricted Health Data License 1.5.0; the relevant DUA is the PhysioNet Restricted Health Data Use Agreement 1.5.0.
- The dataset is not a public open-benchmark release; it requires authenticated access and formal agreement before the files are made available.

To acquire the dataset lawfully:

1. Create or log in to a PhysioNet account at https://physionet.org/login/.
2. Open the official DREAMT page at https://physionet.org/content/dreamt/.
3. Review the access policy, license, and DUA, then sign the data use agreement that PhysioNet requires.
4. Download the official release files only through the authenticated PhysioNet flow after agreement acceptance.
5. Store files locally under `data/raw/dreamt/` or a project-specific local mirror, never in Git, and never commit credentials.

The repository includes a generic authorized download helper at `src/ingest/dreamt_download.py` for use only after a valid DUA is accepted. It supports resumable downloads and checksum validation, but it intentionally requires the user to provide the official authenticated URL and credentials outside the repository.

## Expected local DREAMT layout

After authorized download, the repository expects a structure like this:

```text
data/
├── raw/
│   └── dreamt/
│       ├── subject_01/
│       │   └── record_001/
│       │       ├── accel_x.csv
│       │       ├── accel_y.csv
│       │       ├── accel_z.csv
│       │       ├── bvp.csv
│       │       ├── eda.csv
│       │       ├── temp.csv
│       │       ├── hr.csv
│       │       ├── ibi.csv
│       │       ├── eeg_fpz_cz.csv
│       │       ├── eeg_pz_oz.csv
│       │       ├── emg.csv
│       │       ├── eog.csv
│       │       └── stage_labels.csv
│       └── ...
```

The exact file naming conventions of the official release must be treated as authoritative. The repository does not silently rewrite timestamps or labels to force incompatible files into the synthetic fixture assumptions.
## Sleep-Stage Labels

The planned taxonomies are:

- **Five-class:** Wake, N1, N2, N3, REM
- **Four-class:** Wake, Light (N1 + N2), Deep (N3), REM

Unknown, movement, and unscored annotations must remain explicit or be excluded through a documented policy. They must not be silently converted to Wake.

## Scientific Safeguards

- Split data by subject, not by epoch.
- Fit learned preprocessing transformations on training data only.
- Keep PSG/EEG signals as reference labels and evaluation data, not as inputs to wearable-only models.
- Compute HRV only when valid beat-to-beat intervals are available.
- Distinguish missing signals from genuine zero-valued measurements.
- Report sample sizes, exclusions, limitations, and uncertainty.
- Clearly distinguish synthetic unit tests from validation on real recordings.
- Do not claim to reproduce proprietary Apple, Fitbit, Garmin, or other commercial sleep-scoring algorithms.

An Empatica E4-based experiment, if supported by the available data, evaluates wearable-signal proxies against PSG. It does not by itself measure the accuracy of commercial smartwatch scores.

## Repository Structure

```text
.
├── data/
│   ├── raw/           # Local source data; not committed
│   ├── processed/     # Generated processed data; not committed
│   └── splits/        # Local split artifacts; not committed
├── notebooks/         # Exploratory or runner notebooks
├── paper/             # Manuscript drafts
├── results/
│   ├── figures/       # Generated figures; not committed by default
│   └── tables/        # Generated tables; not committed by default
├── src/
│   ├── ingest/        # Dataset discovery and loading
│   ├── preprocessing/ # Epoch alignment and feature preparation
│   ├── models/        # Model implementations
│   └── eval/          # Evaluation utilities
└── tests/             # Automated tests
```

The exact implemented functionality may evolve; consult the source code and `PROJECT_STATUS.md` for the current implementation status.

## Setup

The project is developed in Python. From the repository root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your environment uses a Python version that is incompatible with a dependency, use a compatible Python version rather than bypassing dependency errors.

## Quality Checks

Run the current lint and test suite:

```bash
python -m ruff check src tests
python -m pytest tests -q
```

Verified synthetic workflow entry point:

```bash
python -m src.pipeline \
  --dataset dreamt \
  --model random_forest \
  --taxonomy standard \
  --sensor-set wearable \
  --artifact path/to/synthetic_dreamt_wearable_epochs.parquet \
  --output-dir results/pipeline \
  --split-output-dir results/pipeline/splits \
  --feature-group all_wearable \
  --split test
```

The unified pipeline executes the synthetic end-to-end flow: artifact validation, subject split creation, training, checkpoint persistence, and evaluation output generation. It does not claim DREAMT compatibility without a real authorized dataset.

Passing unit tests verifies only the behavior covered by those tests. It does not, by itself, establish that the pipeline has been validated on real DREAMT recordings or that scientific experiments have completed.

## Data and Outputs

- Place downloaded source files under `data/raw/` locally.
- Store processed data under `data/processed/`.
- Store generated experiment outputs under `results/`.
- Keep credentials, local environment files, virtual environments, and large datasets out of Git.
- Record configuration, random seeds, software versions, split assignments, and dataset provenance for reproducible runs.

## Current Limitations

The repository is under active development. Depending on the current code state, the following may still require implementation or real-data validation:

- Verified ingestion of actual DREAMT recordings.
- End-to-end epoch alignment and feature generation.
- Subject-level training and validation workflows.
- Baseline and sequence-model training.
- Evaluation metrics, confidence intervals, and result visualization.
- Sensor-ablation experiments and manuscript conclusions.

Do not interpret planned functionality as completed functionality. Consult `PROJECT_STATUS.md`, the CLI, and test results before running experiments.

## License and Dataset Attribution

An appropriate software license will be added before redistributing this repository.

## Project Goal

Nocturne aims to make the limits of wearable-based sleep staging measurable, transparent, and reproducible—without overstating what wearable proxies can infer compared with PSG.
