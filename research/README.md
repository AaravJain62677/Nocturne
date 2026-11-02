# Research Foundations

This folder contains the documentation for the planned real-data study behind Nocturne. It is intentionally written as a research preparation set, not as a report of completed findings.

## Current status

The repository has validated the synthetic pipeline, including configuration checks, subject-split logic, evaluation utilities, feature-group abstractions, and CLI entry points. However, no real DREAMT recordings are currently available in the local workspace, and no real-data experiments have been executed or reported here.

The materials in this folder therefore document the research direction, the gap analysis, the planned protocol, and the manuscript outline. They explicitly avoid claiming any empirical results before real recordings are obtained and evaluated.

## Official acquisition route

The official DREAMT access route is the PhysioNet project page at https://physionet.org/content/dreamt/. The dataset is restricted access and requires an authenticated PhysioNet account plus acceptance of the dataset DUA and the applicable health-data license. After authorization, the user may download the official release files through PhysioNet’s authenticated flow only; the repository does not contain any unauthorized or bypassed download endpoint.

The expected local dataset layout remains a local raw mirror under `data/raw/dreamt/`, with subject and recording directories containing the synchronized wearable and PSG files. No dataset files should be committed to Git, and no credentials should be stored inside the repository.

## Documents

- [literature_review.md](literature_review.md) — a structured overview of the wearable sleep-staging literature and the design principles relevant to this project.
- [gap_analysis.md](gap_analysis.md) — the main scientific and methodological gaps that motivate the project and the project’s specific research positioning.
- [experimental_protocol.md](experimental_protocol.md) — the reproducible protocol for actual dataset access, ingestion, feature extraction, training, evaluation, and ablation analysis.
- [paper_outline.md](paper_outline.md) — the manuscript structure for the eventual real-data study.

## Scientific guardrail

The project’s rule is simple: real-data results are only reportable after the dataset is available, its layout is validated, and the pipeline is executed on actual recordings. Until then, this folder captures the research plan, not performance claims.
