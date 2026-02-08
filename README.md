# Breast Cancer Classification (Logistic Regression Baseline)

This project trains a logistic regression model on the Wisconsin Breast Cancer dataset to establish a baseline machine learning pipeline.

## Features
- Data preprocessing and label encoding
- Stratified train/test split
- Standard scaling to avoid feature magnitude bias
- Logistic regression with different regularization strengths
- Evaluation using ROC-AUC and classification report

## How to Run

```bash
python python-ml/train.py

## Dataset

This project uses the **Wisconsin Breast Cancer dataset**,  
which contains 569 samples with features computed from digitized images of breast masses.

- Binary classification: malignant (1) vs benign (0)
- 30 numerical input features
- Common benchmark dataset for ML classification

The dataset file should be placed at:
data/breast_cancer.csv


## Future Work

Planned improvements to move beyond the baseline:

- Cross-validation for more reliable evaluation
- Hyperparameter tuning using grid search
- Model persistence for reproducibility
- Comparison with nonlinear models (e.g., SVM, Random Forest)
