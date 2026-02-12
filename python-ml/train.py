# python-ml/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold


from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "breast_cancer.csv"


def train_and_eval(X_train, y_train, X_test, y_test, C_value: float):
    # Scaling (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression (L2 regularization by default)
    model = LogisticRegression(max_iter=2000, C=C_value)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # --- Train ROC-AUC (how well it fits seen data) ---
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    train_roc = roc_auc_score(y_train, y_train_prob)

    # Metrics
    roc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, digits=3)
    test_roc = roc_auc_score(y_test, y_prob)
    return train_roc, test_roc, report

# cross validation
def cv_roc_for_C(X_train, y_train, C_value, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    X_np = X_train.to_numpy()
    y_np = y_train.to_numpy()

    train_rocs = []
    val_rocs = []

    for train_idx, val_idx in skf.split(X_np, y_np):
        X_tr, X_val = X_np[train_idx], X_np[val_idx]
        y_tr, y_val = y_np[train_idx], y_np[val_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(max_iter=2000, C=C_value)
        model.fit(X_tr_scaled, y_tr)

        train_prob = model.predict_proba(X_tr_scaled)[:, 1]
        val_prob = model.predict_proba(X_val_scaled)[:, 1]

        train_rocs.append(roc_auc_score(y_tr, train_prob))
        val_rocs.append(roc_auc_score(y_val, val_prob))

    return np.mean(train_rocs), np.std(train_rocs), np.mean(val_rocs), np.std(val_rocs)


def main():
    # 1) read csv
    df = pd.read_csv(DATA_PATH)

    # drop empty column (some dataset versions have it)
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    # 2) Label encode：M/B -> 1/0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # 3) features + target
    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"]

    # 4) train/test split (stratify keeps class ratio stable)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # quick data sanity check
    print("Train class ratio (mean y):", float(y_train.mean()))
    print("Test  class ratio (mean y):", float(y_test.mean()))
    print(df[["radius_mean", "area_mean"]].describe())

    print("\n" + "=" * 60)
    print("Cross-validation selection for C (TRAIN only)")

    best_C = None
    best_val_mean = -1
    best_val_std = None

    for C_value in [1.0, 0.1, 0.01]:
        tr_m, tr_s, val_m, val_s = cv_roc_for_C(X_train, y_train, C_value)
        print(f"C={C_value:<4}  Train ROC={tr_m:.4f}±{tr_s:.4f}  Val ROC={val_m:.4f}±{val_s:.4f}")

        # pick best by highest val mean; tie-breaker by lower std
        if (val_m > best_val_mean) or (val_m == best_val_mean and (best_val_std is None or val_s < best_val_std)):
            best_val_mean = val_m
            best_val_std = val_s
            best_C = C_value

    print(f"\nSelected C = {best_C} (best CV Val ROC mean, stable std)")

    print("\n" + "=" * 60)
    print("Final evaluation on held-out TEST set (used once)")

    train_roc, test_roc, report = train_and_eval(X_train, y_train, X_test, y_test, best_C)
    print(f"FINAL C={best_C}")
    print(f"Train ROC-AUC: {train_roc:.4f}")
    print(f"Test  ROC-AUC: {test_roc:.4f}")
    print(report)




if __name__ == "__main__":
    main()
