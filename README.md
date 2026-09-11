Industrial Machine Failure Prediction System

An end-to-end machine-learning project that predicts whether industrial equipment will fail and, for predicted failures, classifies the failure type. The project evaluates multiple classifiers, addresses severe class imbalance, and includes a Streamlit prototype for single-record and batch inference.

Project objective

The workflow has two stages:

Stage 1 - failure screening: predict machine failure (0 = no failure, 1 = failure).

Stage 2 - failure-type classification: classify predicted failures as Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), or Overstrain Failure (OSF).

Repository contents

data/                 Training and test CSV files
Notebook/TSC.ipynb    Data analysis, modelling, evaluation, and interpretation
models/               Saved scaler, SMOTE object, and stage-wise Random Forest models
streamlit app/        Streamlit prediction prototype and requirements file

Data and preparation

The repository contains separate training and test files with more than 227K sensor records in total. The training subset used in the notebook has 136,429 rows and 14 columns.

Key input variables include air temperature, process temperature, rotational speed, torque, tool wear, and product quality type. ID and Product ID are removed because they are identifiers rather than predictive inputs.

The preprocessing workflow includes:

manual mapping of the categorical Type feature;

StandardScaler for numerical features;

retention of extreme torque and rotational-speed values because they can be associated with overstrain and power failures;

two derived predictors:

temp_diff(k) = process temperature - air temperature;

power[kw] = torque x rotational speed / 9548.8;

SMOTE, class weighting, and threshold tuning to address the minority failure class.

Exploratory and statistical analysis

The notebook examines variable distributions, class imbalance, feature relationships, and failure-pattern differences with histograms, count plots, scatter plots, box plots, and a correlation heatmap.

Three hypothesis tests validate selected relationships:

Mann-Whitney U test for tool-wear differences between failure and non-failure records;

Welch's t-test for torque differences;

Welch's t-test for process-temperature differences.

Modelling and evaluation

The following Stage 1 classifiers are compared:

Logistic Regression

XGBoost

Random Forest

Support Vector Machine (SVM)

Models are assessed with classification reports, confusion matrices, precision-recall curves, F1-scores, cross-validation, GridSearchCV, and threshold analysis. Random Forest with threshold tuning was selected as the Stage 1 model because it provided the strongest precision-recall-F1 balance among the evaluated classifiers.

Reported Stage 1 Random Forest test results after tuning and thresholding:

Class

Precision

Recall

F1-score

No failure

0.99

0.99

0.99

Failure

0.44

0.46

0.45

For Stage 2, the Random Forest classifier reported 74% accuracy, 71% macro F1-score, and an 86% F1-score for Heat Dissipation Failure.

Feature-importance analysis is used to interpret the drivers of model predictions. The notebook highlights rotational speed, torque, tool wear, and derived power as leading candidate predictors.

Streamlit prototype

The streamlit app/app.py prototype supports:

single-record failure screening and conditional failure-type prediction;

batch CSV upload, feature derivation, and prediction display.

The application loads the saved Stage 1 model, Stage 2 model, and scaler from models/. Its current model paths are Windows absolute paths, so update them to local relative paths before running the app.

cd "streamlit app"
pip install -r requirements.txt
streamlit run app.py

Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, XGBoost, Joblib, and Streamlit.

Notes

This is a portfolio project based on the repository datasets. It does not use proprietary operational data or represent a deployment for an industrial company.
