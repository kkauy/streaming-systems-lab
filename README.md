# Breast Cancer Diagnosis – Logistic Regression Study

## Overview
This project investigates the robustness and generalization ability of a
logistic regression model for breast cancer diagnosis using structured
machine learning evaluation.

The workflow follows a **research-grade evaluation protocol**:
- Stratified train/test split
- Hyperparameter selection via stratified cross-validation
- Final unbiased evaluation on a held-out test set
- Reproducible preprocessing without data leakage

---

## Dataset
- Source: Breast Cancer Wisconsin Dataset
- Samples: 569
- Target: Malignant (1) vs Benign (0)

Class distribution is preserved using **stratified splitting**.

---

## Methodology

### 1. Preprocessing
- Feature scaling using **StandardScaler**
- Scaler fitted only on training data to prevent **data leakage**

### 2. Model
- Logistic Regression with **L2 regularization**
- Hyperparameter **C** selected via **5-fold Stratified Cross-Validation**

### 3. Evaluation Protocol
- Cross-validation reports:
  - **Mean ROC-AUC**
  - **Standard deviation (stability)**
- Final model evaluated **once** on held-out test set  
  → ensures **unbiased performance estimate**

---

## Results

### Cross-Validation (Training Set Only)

| C | Train ROC-AUC | Val ROC-AUC |
|---|---------------|-------------|
| 1.0 | 0.9977 ± 0.0008 | **0.9958 ± 0.0047** |
| 0.1 | 0.9958 ± 0.0012 | 0.9949 ± 0.0056 |
| 0.01 | 0.9928 ± 0.0014 | 0.9911 ± 0.0090 |

**Selected C = 1.0**  
(best validation ROC with stable variance)

---

### Final Held-Out Test Performance

- **Train ROC-AUC:** 0.9976  
- **Test ROC-AUC:** 0.9960  

Indicates:
- High discriminative ability  
- Minimal overfitting  
- Strong generalization stability

---

## Reproducibility
- Fixed random seeds (`random_state=42`)
- Stratified splitting across all stages
- No leakage between train / validation / test

This ensures **deterministic and reproducible results**.

---

## Tech Stack
- Python
- Pandas / NumPy
- scikit-learn

---

## Research Significance
Demonstrates a **reproducible machine learning evaluation pipeline**
suitable for:

- Medical risk prediction studies  
- Academic research prototyping  
- Reliable model selection under limited data  

---

## Author
Anthony Au Yeung  
B.S. Computer Science, WGU (2026)
