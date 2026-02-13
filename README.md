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

## Evaluation Pipeline
```
Raw Data (569 samples)
    ↓
[Stratified Split: 80/20]
    ↓
Training Set (455) ────→ 5-Fold CV → Select C=1.0
    ↓                         ↓
    ↓                    Val ROC: 0.9958±0.0047
    ↓
[Train final model with C=1.0]
    ↓
Test Set (114) ────→ Final Evaluation
                         ↓
                    Test ROC: 0.9960
```

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

### Classification Report (Test Set)

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Benign    | 0.982     | 0.986  | 0.984    | 71      |
| Malignant | 0.977     | 0.971  | 0.974    | 43      |
| **Accuracy** |        |        | **0.982** | **114** |

**Key Insight:** High recall (97.1%) for malignant cases is critical in cancer screening to minimize false negatives.

## Reproducibility
- Fixed random seeds (`random_state=42`)
- Stratified splitting across all stages
- No leakage between train / validation / test

This ensures **deterministic and reproducible results**.

---

## Data Leakage Prevention

Medical ML evaluation is highly sensitive to leakage.  
This project explicitly prevents leakage by:

- Fitting preprocessing **only on training folds**
- Performing cross-validation **before** final testing
- Using the held-out test set **exactly once**

This guarantees an **unbiased estimate of clinical generalization**.

---

## Clinical Interpretation

### Why ROC-AUC Matters for Cancer Detection
- **Class imbalance resilient:** Unlike accuracy, ROC-AUC evaluates performance across all classification thresholds
- **Risk stratification:** Probability outputs enable physicians to adjust sensitivity/specificity based on clinical context

### Model Performance Context
- Test ROC-AUC of **0.9960** indicates excellent discriminative ability
- Small train-test gap (0.9976 → 0.9960) suggests minimal overfitting
- For clinical deployment, external validation on different patient populations would be required
## Tech Stack
- **Python 3.x**
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** scikit-learn (LogisticRegression, StratifiedKFold, StandardScaler)
- **Evaluation:** ROC-AUC, Classification Report
- **Persistence:** PostgreSQL (experiment tracking via custom `model_runs_repo`)

---

## Experiment Tracking and Research Integrity

All training runs are persisted into a PostgreSQL database to ensure:

- **Reproducibility** of experimental outcomes  
- **Auditability** of hyperparameter selection  
- **Transactional integrity** of stored research metrics  

This mirrors real-world **ML research and MLOps experiment tracking practices**.

---

## Key Contributions

This project demonstrates:

1. **Proper ML evaluation protocol** preventing common pitfalls:
   - Data leakage via correct train/test scaling
   - Selection bias via nested CV for hyperparameter tuning
   - Overfitting via single test set evaluation

2. **Reproducible research workflow** including:
   - Deterministic random seeds
   - Experiment tracking (PostgreSQL logging)
   - Documented preprocessing steps

3. **Production-ready considerations**:
   - Stratified sampling for imbalanced medical data
   - ROC-AUC metric appropriate for diagnostic tasks
   - Model versioning for audit trails

**Limitations:**
- Small dataset (n=569) limits generalizability
- Single-center data may not represent diverse populations
- Feature interpretability not explored (future work: SHAP values)

---

## Future Research Directions

Potential extensions toward clinical research deployment include:

- **Model interpretability** via SHAP feature attribution  
- **External validation** on independent patient cohorts  
- **Comparison with nonlinear learners** (Random Forest, Gradient Boosting)  
- **Probability calibration** for clinical risk estimation  

These directions move the study closer to **translational medical AI research**.

---

## Author
Anthony Au Yeung  
B.S. Computer Science, WGU (2026)
