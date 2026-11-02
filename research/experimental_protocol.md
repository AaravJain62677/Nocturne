# Experimental Protocol for Real DREAMT Validation

## Purpose

This protocol defines the step-by-step workflow for validating the Nocturne pipeline on real recordings. It is intended to be executed only after the dataset is available locally and the ingestion contract has been validated. The plan is written to be explicit, reproducible, and conservative about claims.

## Research objective

To estimate the degree to which wearable-derived features can classify sleep stages against PSG ground truth under a standardized subject-independent evaluation design.

## Scope and exclusions

This study will evaluate wearable-only feature sets and will not treat EEG-derived PSG channels as model inputs. PSG labels remain the ground truth for the target labels. This preserves the scientific distinction between the reference annotation and the wearable proxy signal set.

This study will not claim performance for proprietary device algorithms, and it will not infer a commercial product accuracy from an unvalidated local pipeline.

## Data prerequisites

The following conditions must hold before the protocol is executed:

1. The DREAMT dataset is present under the configured local data directory.
2. The dataset layout matches the expected contract.
3. Subject IDs, recording IDs, timestamps, and stage labels can be resolved consistently.
4. The recording set includes valid sleep-stage annotations and wearable-compatible signals required for feature extraction.
5. Any exclusions (missing channels, corrupted files, invalid epochs, and unsupported labels) are recorded.

If these conditions are not met, the study stops at the validation check and does not proceed to training or reporting.

## Experimental design

### Data partitioning

- Split by subject, never by epoch.
- Use a train/validation/test partition that keeps subjects disjoint across splits.
- Record the exact subject assignment and random seed used for reproducibility.
- Preserve subject-specific sample counts in the reporting output.

### Target taxonomies

The primary evaluation will use the standard five-stage taxonomy:

- Wake
- N1
- N2
- N3
- REM

A secondary view will report the collapsed taxonomy:

- Wake
- Light (N1 + N2)
- Deep (N3)
- REM

Unknown, movement, and unscheduled states will not be silently coalesced into wake. They will be treated according to the project’s explicit handling policy.

### Input feature groups

The baseline workflow will compare model performance under wearable feature sets such as:

- PPG-derived heart-rate and HRV features,
- accelerometry-derived activity and motion features,
- combined wearable feature groups,
- optional additional wearable channels if they are available and validated.

This is intentionally designed as an ablation study, not as a claim that all feature groups are equally informative.

### Models

The planned model comparison includes simple baselines and stronger supervised learning approaches supported by the repository, such as:

- deterministic baselines,
- shuffling or dummy baselines,
- tree-based models,
- gradient boosting models,
- sequence models if the processed data supports them.

The final model list will be constrained by the available data, feature schema, and reproducibility requirements.

## Processing pipeline

1. Validate dataset layout and recordings.
2. Load raw signals and metadata.
3. Align timestamps and 30-second epoch boundaries.
4. Map PSG labels into the canonical project taxonomy.
5. Exclude unsupported or ambiguous samples according to a documented policy.
6. Construct wearable-only feature matrices that match the expected model interfaces.
7. Generate subject-independent splits.
8. Fit preprocessing and model parameters on training data only.
9. Evaluate on held-out validation and test sets.
10. Aggregate subject-level outputs and generate summary tables/figures.

## Metrics to report

The study should report at least:

- overall accuracy,
- macro-F1,
- Cohen’s kappa,
- per-stage precision/recall/F1,
- confusion matrices,
- subject-level performance distributions,
- confidence intervals or bootstrap estimates where supported by sample size.

Results should be reported with the sample sizes and exclusions that produced them.

## Ablation analyses

The project’s ablation analysis will compare:

- PPG-only,
- accelerometer-only,
- combined wearable features,
- alternative support for secondary taxonomy.

The purpose is not to search for a “best” feature set by default, but to understand which signal combinations contribute meaningful predictive value.

## Failure handling and stopping rules

The study must stop and report the reason if any of the following occur:

- the dataset is absent or the local layout is invalid,
- stage labels are incompatible with the project taxonomy,
- channel alignment fails,
- subject assignment is ambiguous,
- feature extraction is not possible for the target model schema,
- evaluation outputs cannot be reproduced.

In all stopping cases, the repository must preserve the reason for failure and avoid reporting any experimental result.

## Reporting standard

A complete result report should include:

- dataset provenance,
- participant counts and exclusions,
- split definitions,
- label policy,
- feature set definition,
- model definition,
- evaluation metrics,
- uncertainty measures,
- failure modes and limitations.

The study should remain faithful to the evidence: if the dataset is incomplete or the pipeline has not been executed successfully, the paper should not present results.
