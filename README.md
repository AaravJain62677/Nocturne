# Nocturne

### Reproducible Wearable Sleep-Stage Classification

Nocturne is a reproducible research framework for evaluating **sleep-stage classification from wearable-derived physiological and motion signals** under a **subject-independent experimental protocol**.

The project is designed around a central research question:

> **How effectively can wearable-derived physiological and motion features classify sleep stages under subject-independent evaluation, and which signal modalities contribute most to classification performance?**

Nocturne provides an end-to-end pipeline covering data ingestion, preprocessing, feature extraction, subject-independent splitting, model training, evaluation, ablation studies, and reproducible experiment artifacts.

---

## Research Scope

Sleep-stage classification is commonly performed using polysomnography (PSG), which provides rich physiological signals such as EEG, EOG, and EMG. However, wearable devices offer a more accessible alternative through signals such as:

- Heart rate
- Heart-rate variability (HRV)
- Accelerometer-derived motion
- Other wearable-compatible physiological signals

Nocturne investigates how these wearable-derived signals can be used for automated sleep-stage classification while maintaining a strict separation between subjects during training and evaluation.

The framework supports both:

### Five-class sleep staging

```text
WAKE
N1
N2
N3
REM
```

### Four-class sleep staging

```text
WAKE
LIGHT
DEEP
REM
```

---

## Experimental Design

The primary evaluation protocol uses **subject-independent splits**.

Subjects are assigned to training, validation, and test partitions before model evaluation. This prevents epochs from the same subject appearing across different partitions and reduces the risk of subject-level data leakage.

The primary evaluation metric is:

**Macro-F1**

because sleep-stage datasets can contain substantial class imbalance and macro-averaging gives each sleep stage equal weight.

Additional metrics include:

- Accuracy
- Balanced Accuracy
- Weighted F1
- Cohen's Kappa
- Per-class precision, recall, and F1
- Subject-level metrics
- Confusion matrices

---

## Pipeline

```text
                 ┌─────────────────────┐
                 │     DREAMT Data     │
                 │  / Synthetic Demo   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Data Validation   │
                 │ & Dataset Ingestion │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   30-second Epochs  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Wearable Feature    │
                 │    Extraction       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Subject-Independent │
                 │       Splits        │
                 └──────────┬──────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │        Model Training        │
             │                              │
             │ RF │ HistGB │ GRU │ LSTM    │
             └──────────────┬───────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Evaluation      │
                 │ Metrics / Confusion │
                 │ / Subject Analysis  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Reproducible Result │
                 │     Artifacts       │
                 └─────────────────────┘
```

---

## Models

The framework includes classical machine-learning baselines and sequence models.

### Classical models

- DummyClassifier
- Random Forest
- HistGradientBoosting

### Sequence models

- GRU
- LSTM

The architecture is intentionally modular so additional models can be evaluated using the same preprocessing and evaluation protocol.

---

## Feature Groups

Nocturne supports feature-group ablation to investigate the contribution of different wearable signal modalities.

Examples include:

| Feature Group | Description |
|---|---|
| `hr` | Heart-rate features |
| `hrv` | Heart-rate variability features |
| `acceleration` | Motion / accelerometer features |
| `hr_hrv` | Heart rate + HRV |
| `hr_acceleration` | Heart rate + acceleration |
| `all_wearable` | Complete wearable feature set |

This allows the experimental pipeline to answer questions such as whether physiological or motion-derived features contribute more strongly to sleep-stage discrimination.

---

## Dataset

The intended real-world dataset is **DREAMT**.

DREAMT is access restricted, so the repository does **not** include the real dataset.

The project therefore provides an explicit synthetic demonstration mode for validating the complete pipeline without requiring restricted data.

### Important

The synthetic demonstration is **not a substitute for real DREAMT evaluation**.

Any metrics generated using the synthetic dataset are:

> **Pipeline-validation results only — they are not real DREAMT benchmark results.**

No real-data performance claims are made from the included synthetic results.

---

## Installation

### Requirements

- Python 3.10+
- Git
- A virtual environment is recommended

Clone the repository:

```bash
git clone https://github.com/AaravJain62677/Nocturne.git
cd Nocturne
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Synthetic Demonstration

The synthetic demonstration validates the complete pipeline without requiring access to DREAMT.

Run:

```bash
python -m src.pipeline \
    --dataset dreamt \
    --model random_forest \
    --taxonomy standard \
    --sensor-set wearable \
    --seed 42 \
    --data-dir data \
    --output-dir results/synthetic_demo \
    --split-output-dir results/synthetic_demo/splits \
    --feature-group all_wearable \
    --split test \
    --synthetic-demo
```

The run generates reproducible evaluation artifacts under:

```text
results/synthetic_demo/
```

including:

```text
metrics.json
confusion_matrix.csv
normalized_confusion_matrix.csv
per_subject_metrics.csv
ablation_summary.csv
evaluation_metadata.json
pipeline_summary.json
random_forest_checkpoint.json
random_forest_checkpoint.pkl
```

The synthetic dataset and its provenance metadata are also stored in the results directory.

---

## View Results

Nocturne includes a lightweight CLI for inspecting an evaluation directory.

Run:

```bash
python -m src.cli results results/synthetic_demo
```

The CLI reports:

- Experiment configuration
- Evaluation metrics
- Test-set information
- Subject independence
- Class coverage
- Generated artifacts
- Synthetic-data disclaimer when applicable

Example:

```text
NOCTURNE
Reproducible wearable sleep-stage evaluation

Experiment
──────────
  Dataset:        DREAMT
  Model:          random_forest
  Taxonomy:       standard
  Sensor set:     wearable
  Feature group:  all_wearable
  Seed:            42
  Split:           test

Metrics
───────
  Accuracy             0.2500
  Balanced Accuracy    0.2400
  Macro F1             0.1333
  Weighted F1          0.1389
  Cohen's Kappa        0.0568
```

These values are from the **synthetic demonstration** and must not be interpreted as real DREAMT results.

---

## Using Real DREAMT Data

Real DREAMT experiments require authorized access to the dataset.

Once the dataset has been obtained through the appropriate access process, place the data in the expected data directory:

```text
data/
```

Then run the pipeline **without** the synthetic demonstration flag.

For example:

```bash
python -m src.pipeline \
    --dataset dreamt \
    --model random_forest \
    --taxonomy standard \
    --sensor-set wearable \
    --seed 42 \
    --data-dir data \
    --output-dir results/real_dreamt \
    --split-output-dir results/real_dreamt/splits \
    --feature-group all_wearable \
    --split test
```

The pipeline intentionally does **not** silently replace missing real data with synthetic data.

If DREAMT is unavailable, the pipeline fails explicitly rather than producing results that could be mistaken for real-data experiments.

---

## Reproducibility

Experiments record configuration and evaluation metadata alongside generated results.

Important reproducibility controls include:

- Explicit random seeds
- Subject-independent dataset splits
- Stored split assignments
- Dataset and experiment metadata
- Feature-group configuration
- Model configuration
- Evaluation configuration
- Synthetic-data provenance
- Saved model checkpoints
- Machine-readable metrics

The project also includes automated tests covering research-critical behavior.

---

## Testing

Run the complete test suite:

```bash
pytest -q
```

The current repository passes:

```text
66 passed
```

Run linting:

```bash
ruff check .
```

Expected result:

```text
All checks passed!
```

Some tests intentionally use very small synthetic class distributions, which can produce harmless scikit-learn metric warnings. These warnings do not indicate test failures.

---

## Project Structure

```text
Nocturne/
├── data/
│
├── docs/
│   └── architecture.md
│
├── results/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── config.py
│   ├── contracts.py
│   ├── experiment.py
│   ├── ingest/
│   ├── models/
│   ├── preprocessing/
│   ├── eval/
│   ├── inspect_data.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── train.py
│   └── cli.py
│
├── tests/
│
├── .gitignore
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## Evaluation Philosophy

Nocturne is designed around several principles important for reproducible machine-learning research.

### 1. Subject independence

Sleep epochs from the same subject should not be distributed across training and test partitions.

### 2. Explicit synthetic data

Synthetic data is useful for validating software infrastructure, but synthetic performance is not treated as evidence for real-world model performance.

### 3. Reproducibility

Experiment configuration, splits, metrics, and provenance should be preserved alongside results.

### 4. Modular evaluation

Different models and feature groups should be evaluated under the same experimental protocol.

### 5. No hidden fallback

If real DREAMT data is unavailable, the pipeline does not silently substitute synthetic data.

---

## Current Status

| Component | Status |
|---|---|
| Data ingestion architecture | Complete |
| DREAMT validation | Implemented |
| 30-second epoch processing | Complete |
| Wearable feature extraction | Complete |
| Five-class taxonomy | Complete |
| Four-class taxonomy | Complete |
| Subject-independent splitting | Complete |
| Leakage safeguards | Implemented |
| Random Forest | Complete |
| HistGradientBoosting | Complete |
| GRU | Complete |
| LSTM | Complete |
| Evaluation metrics | Complete |
| Subject-level evaluation | Complete |
| Feature ablation | Complete |
| Synthetic end-to-end validation | Complete |
| Reproducibility metadata | Implemented |
| CLI | Complete |
| Automated tests | Complete |
| Real DREAMT experiments | Pending dataset access |

---

## Limitations

The current repository should be interpreted as a **reproducible experimental framework with synthetic end-to-end validation**, rather than as a completed benchmark of wearable sleep-stage classification.

The principal limitation is the absence of accessible real DREAMT data in the repository.

Consequently:

- Synthetic metrics are not real-world performance estimates.
- No clinical conclusions should be drawn from the synthetic demonstration.
- Real-data model comparisons remain to be performed.
- Real-data confidence intervals and statistical analysis remain future experimental work.

---

## Research Roadmap

The next stage of the study, once authorized DREAMT data is available, is to execute the full experimental matrix:

1. Real DREAMT ingestion and validation
2. Five-class primary evaluation
3. Four-class secondary evaluation
4. Multi-model comparison
5. Feature-group ablations
6. Multiple random seeds
7. Subject-level bootstrap confidence intervals
8. Per-class error analysis
9. Model comparison tables
10. Publication-quality figures
11. Statistical analysis
12. Final reproducibility manifest

---

## Citation

If you use Nocturne in research, please cite the repository and the associated paper when available.

```bibtex
@software{nocturne,
  author = {Aarav Jain},
  title = {Nocturne: Reproducible Wearable Sleep-Stage Classification},
  year = {2026},
  url = {https://github.com/AaravJain62677/Nocturne}
}
```

---

## License

See the repository license for terms of use.

---

## Author

**Aarav Jain**


GitHub: [AaravJain62677](https://github.com/AaravJain62677)

---

> **Nocturne is built to make wearable sleep-stage experiments reproducible, subject-independent, and explicit about the boundary between software validation and scientific evidence.**
