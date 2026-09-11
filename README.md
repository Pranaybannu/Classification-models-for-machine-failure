# Industrial Machine Failure Prediction System

An end-to-end machine-learning project that predicts whether industrial equipment will fail and, for predicted failures, classifies the failure type. The project evaluates multiple classifiers, addresses severe class imbalance, and includes a Streamlit prototype for single-record and batch inference.

## Project Objective

The workflow has two stages:

1. **Stage 1 - Failure Screening:** Predict machine failure (`0` = no failure, `1` = failure).
2. **Stage 2 - Failure-Type Classification:** Classify predicted failures as Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), or Overstrain Failure (OSF).

## Repository Contents

```text
data/                 Training and test CSV files
Notebook/TSC.ipynb    Data analysis, modelling, evaluation, and interpretation
models/               Saved scaler, SMOTE object, and stage-wise Random Forest models
streamlit app/        Streamlit prediction prototype and requirements file
