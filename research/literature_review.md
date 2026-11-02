# Literature Review: Wearable Sleep Staging with PPG, HRV, Accelerometry, and Related Signals

## Scope and intent

This note is a deliberately scoped review of the main research directions relevant to Nocturne. It is not a complete bibliography and it does not claim any specific demonstrated performance from this repository. The purpose is to interpret the broader wearable-sleep literature in terms of what the project is trying to test:

- whether sleep stages can be inferred from wearable-available signals rather than PSG,
- which sensor combinations are informative,
- which stage transitions are easiest or hardest to distinguish,
- how model choice and subject-level splitting affect generalization.

## High-level themes in the wearable sleep-staging literature

### 1. PPG and heart-rate dynamics are informative for sleep staging

A large body of wearable sleep work relies on photoplethysmography (PPG) and on heart-rate-based features. In practice, PPG-derived signals support features such as instantaneous heart rate, heart-rate variability, pulse amplitude, and time-varying cardio-respiratory proxies. These features are often most informative for differentiating wake from sleep and for capturing broad sleep-depth transitions, especially the contrast between deep sleep and lighter stages.

The central challenge is that PPG signals are noisy, motion-sensitive, and dependent on subject-specific physiology. This makes stage classification more difficult than PSG-based sleep scoring, especially when the model is required to generalize across different subjects and recording conditions.

### 2. Accelerometry adds information about movement and restlessness

Accelerometer signals help identify periods of large movement, reduced activity, and behavioral transitions that correlate with sleep. This is especially useful for distinguishing wake and movement epochs from quiescent sleep states. In many wearable settings, accelerometer data is one of the most accessible and robust inputs, but on its own it is not sufficient to resolve all stage boundaries.

A common theme in the literature is that accelerometer information is powerful as a complementary signal, not as a stand-alone complete sleep-scoring solution. Its value is strongest when combined with PPG or other physiological features.

### 3. Multimodal sensing generally outperforms single-sensor approaches

The literature consistently suggests that models trained on multiple wearable channels outperform single-signal models. The gains are typically largest when combining motion and cardiovascular features, and sometimes additional wearable channels such as temperature or skin conductance-related measures when available.

This motivates the project’s sensor-ablation structure. If the Nocturne pipeline is to be scientifically useful, it must determine not just whether a wearable-only model works in aggregate, but where the signal is coming from and which combinations matter most.

### 4. Wake/light/deep/REM boundaries remain difficult

Even in well-performing studies, stage boundaries remain difficult. Wake versus sleep and REM versus non-REM are often easier to distinguish than N1 versus N2, or N2 versus N3, because the underlying physiological signatures are more subtle. This challenge is highly relevant to wearable-only models because the margin between stages may be compressed when using indirect proxies rather than EEG-derived sleep features.

This raises a methodological requirement for Nocturne: evaluation should not only report overall accuracy, but also per-stage recall, precision, confusion matrices, and macro-averaged performance measures.

### 5. Generalization is a central issue

Subject-independent evaluation is a recurring concern in wearable sleep studies. Models can appear strong when epochs are split without controlling for subject identity, because subject-specific physiology and behavior leak across partitions. These leakage patterns are especially risky when using wearable signals with substantial inter-subject variability.

This reinforces the project’s emphasis on subject-level splits, carefully separated train/validation/test partitions, and transparent reporting of subject counts and exclusions.

### 6. Dataset quality and label handling matter as much as model architecture

Many wearable studies rely on PSG-labeled sleep stages as ground truth, but the practical implementation details matter: label normalization, movement states, unknown labels, and epoch alignment can materially change performance. Studies that silently convert uncertain labels to wake or merge incompatible taxonomies risk producing optimistic or non-comparable results.

This project addresses this explicitly by preserving unknown and movement labels as explicit states and by separating taxonomies instead of conflating them.

## Relevance to Nocturne

The literature suggests three core expectations for a wearable sleep-staging project:

1. A multimodal wearable approach is more realistic than a single-sensor one.
2. PPG-derived heart-rate and HRV features are likely to be central predictors, but they must be paired with motion or contextual features to improve stage separation.
3. Subject-independent evaluation and transparent reporting are essential to avoid overestimating real-world performance.

These expectations align closely with the repository’s architecture:

- wearable-only feature groups,
- subject-split logic,
- explicit taxonomy handling,
- model and ablation comparison,
- metric reporting that is sensitive to class imbalance.

## Important limitation of this review

This review is intentionally framed around the literature themes most relevant to the current project. It should not be treated as a final publication-grade literature review. The final manuscript review will need to be expanded with a verified source list and precise citation details once the actual dataset pipeline is running and the final study scope is locked.
