# python-ml/train.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path
import pandas as pd

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

    # Metrics
    roc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, digits=3)
    return roc, report


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

    # 5) Regularization experiment (C smaller => stronger regularization)
    for C_value in [1.0, 0.1, 0.01]:
        roc, report = train_and_eval(X_train, y_train, X_test, y_test, C_value)
        print("\n" + "=" * 60)
        print(f"C = {C_value}  (smaller C => stronger regularization)")
        print(report)
        print("ROC-AUC:", roc)


if __name__ == "__main__":
    main()
