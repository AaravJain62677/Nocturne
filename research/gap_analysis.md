# Gap Analysis and Research Positioning

## Executive summary

Nocturne is positioned at the intersection of wearable sensing, physiologic time-series modeling, and subject-independent evaluation for sleep staging. The project is motivated by a clear gap: many wearable sleep studies show encouraging signal quality, but performance often depends on dataset selection, sensor setup, subject independence, and careful labeling policy.

This repository is designed to formalize and test that gap in a reproducible manner. The current codebase is intentionally robust about data contracts and evaluation structure, but it has not yet produced any real-data findings because the real DREAMT dataset is not locally available.

## Scientific gaps motivating the project

### 1. Real wearable sleep staging remains challenging outside controlled settings

Wearable devices can provide rich physiological signals, but their fidelity is lower than PSG-derived signals. PPG-based HRV and accelerometer-derived movement are practical and scalable, yet they are indirect proxies for sleep physiology. The gap is not whether wearable signals contain sleep information, but how much stage information remains usable under realistic subject variability and motion noise.

### 2. A single-sensor setup is rarely sufficient

Many consumer wearable systems rely on heart-rate and movement data, but the literature suggests that these are most useful when modeled jointly rather than treated as independent features. The project therefore emphasizes feature-group comparisons and ablation analysis to distinguish the incremental value of each modality.

### 3. Stage imbalance and taxonomy choices distort performance claims

Sleep stage distribution is uneven. Wake and N2 often dominate, while N1 is sparse and difficult. If not handled carefully, overall accuracy can overstate model quality. This motivates the use of macro-level metrics, confusion matrices, and explicit secondary taxonomy reporting.

### 4. Subject leakage can exaggerate performance

A common methodological mistake in sleep-staging pipelines is to split by epoch instead of by subject. This can leak subject-specific patterns across train and test sets, inflating performance. The project avoids this by design and makes it a core part of the protocol.

### 5. Real-data validation is essential and currently missing

The main gap in the current repository is not algorithmic design alone; it is the absence of verified real-recording execution. Until DREAMT data is available and validated, the project must refrain from claiming actual sleep-staging results.

## Methodological gaps addressed by this repository

### Data contracts and ingestion discipline

The project is designed to validate dataset structure before training begins. This prevents downstream errors in index alignment, subject labeling, and epoch handling. This is important because real wearable datasets often contain heterogeneous metadata and nontrivial preprocessing requirements.

### Explicit stage handling

The configuration layer distinguishes wake, movement, unknown, and unscheduled states instead of silently coalescing everything into one label. This reduces the chance of hidden label corruption and makes evaluation more defensible.

### Reproducible evaluation design

The evaluation utilities and CLI support subject-level aggregation, feature-group comparison, and metric reporting. This creates a reproducible platform for later real-data experiments, rather than a one-off notebook workflow.

### Abstraction over feature groups

The repository supports feature-set ablation; this is important because the true scientific question is not only “does the model work?” but “which sensor combination yields the signal that matters?”

## Research positioning

Nocturne sits in a pragmatic middle ground:

- it does not claim to replace PSG,
- it does not overstate consumer-wearable performance,
- it targets interpretable, feature-driven evaluation against real labels,
- it treats reproducibility and transparency as core scientific infrastructure.

This positioning is appropriate because the project’s contributions are strongest when they are honest about the limits of wearable-only inference.

## Current state of evidence

At present, the evidence in this repository supports only the following claims:

- the engineering scaffolding for synthetic validation is implemented,
- the evaluation and ablation pipeline behaves consistently in the tested synthetic regime,
- real-data execution remains blocked until DREAMT files are available and validated.

No scientific conclusions about wearable sleep staging accuracy should be drawn from this repository until the real-data protocol is executed on actual recordings.
