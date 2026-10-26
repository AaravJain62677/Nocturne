# Sleep Degradation

A reproducible research repository for quantifying how much sleep-staging performance changes when using wearable-available signals instead of PSG-derived labels, and how much of the gap can be recovered with stronger models and sensor combinations.

## Research goals

- Compare PSG-derived sleep-stage labels against wearable signal predictions.
- Measure the performance difference between full PSG and wearable-only inputs.
- Evaluate whether better modeling and sensor combinations recover a meaningful fraction of the gap.
- Maintain a clear, reproducible pipeline with explicit stage-label handling and deterministic project configuration.

## Project status

This repository is in Phase 1: project foundation, validation, canonical stage mapping, and unsupported-operation guardrails. Dataset download, model implementation, and training workflows are intentionally unavailable until later phases.

## Local setup

### 1. Create the Python environment

```bash
cd /home/aaravjain62677/projects/sleep-degradation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Dependency notes

The project intentionally keeps scientific Python and ML tooling minimal while remaining suitable for sleep-staging research.

- `numpy` and `pandas`: tabular preprocessing, feature generation, and dataset inspection.
- `scikit-learn`: evaluation metrics and validation splits when available.
- `torch`: deep learning runtime used for model experimentation.
- `matplotlib` and `seaborn`: plotting for performance summaries and figures.
- `pytest` and `ruff`: automated validation and linting.

## Activation and usage

```bash
cd /home/aaravjain62677/projects/sleep-degradation
source .venv/bin/activate
python -m src.train --dataset dreamt --model baseline --mode validate --taxonomy standard
```

The CLI validates the selected dataset, model, taxonomy, and execution mode. `--mode validate` is the supported configuration check in Phase 1.

Training and evaluation are intentionally unavailable in this phase and will stop with a clear error message instead of producing misleading output.

## Testing

```bash
source .venv/bin/activate
python -m pytest tests -q
```

Optional linting:

```bash
source .venv/bin/activate
python -m ruff check src tests
```

## Canonical stage handling

The repository defines a canonical stage mapping that treats stage labels explicitly instead of silently converting unknown values to Wake.

- Known labels: `WAKE`, `N1`, `N2`, `N3`, `REM`
- Movement and unscored labels remain distinct: `MOVEMENT`, `UNSCORED`
- Unknown labels are preserved as `UNKNOWN`
- Empty or missing stage labels raise a `ValueError`

This prevents label ambiguity from being silently introduced during preprocessing or evaluation.

## Notable repository conventions

- Raw datasets are stored under `data/raw/` and are not committed to Git.
- Processed data and generated results are under `data/processed/` and `results/` respectively.
- Paths are created from the repository root rather than the current working directory.
- Seeds are configured through the shared utilities for Python, NumPy, and PyTorch.

## Next phase

Phase 2 will add dataset-specific ingestion and modeling workflow definitions, after the repository foundation and validation rules have been reviewed.
