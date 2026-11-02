# Manuscript Outline for the Nocturne Study

## Status note

This document defines the intended structure of the eventual research paper. It is intentionally written as a placeholder for a real-data study and does not include any fabricated results or conclusions.

## 1. Title

A working title should be precise enough to reflect the project’s scope without overstating its novelty. Examples of good direction include:

- Wearable Sleep Staging from PPG and Movement Signals: A Reproducible Subject-Independent Evaluation
- Wearable-Only Sleep Stage Classification Against PSG Ground Truth
- Evaluating Multimodal Wearable Signals for Sleep Staging in a Subject-Independent Framework

The final title should match the actual study design and the dataset used.

## 2. Abstract

The abstract should state:

- the problem of sleep-stage estimation from wearable signals,
- the use of PSG labels as ground truth,
- the focus on reproducible subject-independent evaluation,
- the sensor modality and feature strategy,
- the actual reported results once the real-data study is complete.

The abstract must be written only after execution of the real-data pipeline and should not include unsupported claims.

## 3. Introduction

### Motivation

Describe why wearable sleep staging matters for scalable, low-burden monitoring and why it remains difficult in practice.

### Research questions

- How well do wearable-derived features predict PSG sleep stages?
- Which wearable signal groups matter most?
- Which stage transitions are hardest to separate?
- How much does subject-independent evaluation affect measured performance?

### Contributions

List the actual contributions once validated in the final study. These may include:

- dataset and subject-split protocol,
- wearable feature definitions,
- model and ablation design,
- reproducible evaluation outputs.

## 4. Background and Related Work

Discuss:

- the rise of wearable sleep monitoring,
- PPG and HRV as practical physiological proxies,
- accelerometer information as a motion and activity signal,
- the need for multimodal modeling and careful evaluation,
- the relevance of subject-independent testing and taxonomy definitions.

This section should use verified literature sources and not rely on unverified or fabricated citations.

## 5. Dataset and Ground Truth

Include:

- dataset name and source,
- recording counts and subject counts,
- inclusion/exclusion policies,
- stage label definitions,
- handling of unknown, movement, and unscheduled labels,
- PSG-based ground truth definition.

## 6. Methods

### Sensor and feature design

Describe the wearable feature groups used by the project, with precise definitions for:

- PPG-derived heart rate and HRV features,
- accelerometer features,
- combined wearable feature sets,
- optional additional channels if valid.

### Preprocessing and labeling

Explain:

- epoch alignment,
- handling of missing or invalid values,
- feature extraction and standardization,
- subject-level split generation,
- taxonomy mapping and stage policy.

### Models

Describe the classification framework, including baseline and comparison models, and justify why the selected models are appropriate for the data and feature set.

## 7. Experimental Setup

Report:

- subject-independent split strategy,
- training and validation logic,
- hyperparameter choices,
- optimization procedure,
- evaluation schedule,
- ablation comparisons.

## 8. Results

This section should present only actual outputs from real-data execution, including:

- overall accuracy,
- macro-F1,
- kappa,
- per-stage metrics,
- confusion matrices,
- ablation tables,
- figures showing performance patterns.

No result values should be inserted before the real-data run is complete.

## 9. Discussion

Interpret the findings in light of the literature. Discuss:

- which signals were most informative,
- where the model failed,
- whether the error pattern matches known stage boundaries,
- why subject independence matters,
- the role of data quality and labeling policy.

## 10. Limitations

Explicitly address:

- local data availability constraints,
- possible missingness and annotation issues,
- subject heterogeneity,
- model assumptions,
- generalization beyond the dataset,
- the limits of wearable-only inference relative to PSG.

## 11. Conclusion

Summarize only what the executed study actually showed. Avoid overgeneralization or claims beyond the validated data.

## 12. Appendices

Potential appendices:

- feature list and definitions,
- split assignments,
- environment specification,
- dataset processing notes,
- reproducibility checklist.

## Final editorial principle

The manuscript should be as honest as the code: no real-data result is reportable before the dataset is present, the pipeline is executed, and the outputs are verified.
