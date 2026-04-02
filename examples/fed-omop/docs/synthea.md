# Synthea for Multi-Site Readmission Prediction

## Overview

This project uses **Synthea** as the raw synthetic data generator and publishes a **preprocessed downstream dataset** on Hugging Face for the **30-day readmission** task.

In practice, the Hugging Face dataset should be understood as the **machine learning-ready output** of the pipeline:

- raw synthetic records as input
- engineered `X` and readmission `label` as output

More broadly, this project uses **Synthea** as the upstream data generator for a **multi-site hospital readmission prediction** workflow. In this repository, it serves as a privacy-safe source dataset to develop and test the preprocessing pipeline, feature engineering workflow, and federated learning setup.

## Data Provenance

The raw synthetic records were generated using **Synthea**:

```text
https://github.com/synthetichealth/synthea
```

The current sample metadata corresponds to six state-based runs (Arizona, California, Connecticut, Massachusetts, Ohio, Oregon):

Each run contains **10 years of exported history**, with small sample sizes ranging from **21 to 28 patients** in the uploaded metadata.

A small project-specific version of the resulting dataset is shared on Hugging Face:

```text
https://huggingface.co/datasets/danimanjah/synthea_small
```

The Hugging Face dataset contains the **preprocessed machine learning inputs** derived from the raw Synthea exports. It should therefore be understood as the downstream output of the preprocessing pipeline, rather than as the raw Synthea tables.

It includes:

- a preprocessed feature matrix `X`
- a binary readmission target stored as `label`
- hospital identifiers such as `hospital_id` and `hospital_code`

## Feature Construction

The preprocessing pipeline builds `X` from several groups of patient-level variables.

### Core features

- patient demographics
- comorbidity information
- previous encounters in the last 180 days

### Additional features currently enabled

- BMI-related features
- anxiety-related features
- medication-related features

The final exported `X` includes:

- numeric variables that are cleaned and imputed
- categorical variables retained and filled with `"UNK"` where needed

The preprocessing pipeline is available in the original GitHub repository:

```text
https://github.com/manjahdani/fedomop
```

## Multi-Site Usage

This dataset is used in a **multi-site configuration**, where each hospital or site represents a distinct local data distribution.

By default, the workflow uses the `iid` setting, which pools all available datasets and then applies an IID split across nodes.

If you prefer to preserve the original hospital-level separation, use:

```text
partitioner = "natural"
```

Note that with the natural split, the number of nodes must not exceed the number of hospitals available in the dataset. Otherwise, the run will fail.

This setup is suitable for evaluating:

- local model training
- centralized baselines
- federated learning strategies
- the effect of site heterogeneity on performance

## Limitations

This dataset should be used with the following limitations in mind:

- it is **synthetic**, not real-world clinical data
- it is intended for **prototyping and experimentation**, not clinical validation
- the target and features depend on the assumptions encoded in the preprocessing pipeline
- some variables are derived aggregates rather than native raw EHR fields