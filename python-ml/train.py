from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

from model_runs_repo import insert_model_run

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "breast_cancer.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    return df


def make_train_test_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df.drop(columns=["id", "diagnosis"])
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

# cross validation
def cv_roc_for_C(X_train, y_train, C_value, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    X_np = X_train.to_numpy()
    y_np = y_train.to_numpy()

    train_rocs = []
    val_rocs = []

    for train_idx, val_idx in skf.split(X_np, y_np):
        X_tr, X_val = X_np[train_idx], X_np[val_idx]
        y_tr, y_val = y_np[train_idx], y_np[val_idx]

        # fold : fit scaler on fold - train only
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        # probability Soft core for ROC -AOC
        model = LogisticRegression(max_iter=2000, C=C_value)
        model.fit(X_tr_scaled, y_tr)

        # probability of class 1 (maglignant)
        train_prob = model.predict_proba(X_tr_scaled)[:, 1]
        val_prob = model.predict_proba(X_val_scaled)[:, 1]

        train_rocs.append(roc_auc_score(y_tr, train_prob))
        val_rocs.append(roc_auc_score(y_val, val_prob))

    return float(np.mean(train_rocs)), float(np.std(train_rocs)), float(np.mean(val_rocs)), float(np.std(val_rocs))


def select_best_c_by_cv(X_train, y_train, C_candidates=(1.0, 0.1, 0.01), n_splits=5):
    best_C = None
    best_val_mean = -1.0
    best_val_std = None

    for C_value in C_candidates:
        tr_m, tr_s, val_m, val_s = cv_roc_for_C(X_train, y_train, C_value, n_splits=n_splits)
        print(f"C={C_value:<4}  Train ROC={tr_m:.4f}±{tr_s:.4f}  Val ROC={val_m:.4f}±{val_s:.4f}")

        # pick best by highest val mean
        if (val_m > best_val_mean) or (val_m == best_val_mean and (best_val_std is None or val_s < best_val_std)):
            best_val_mean = val_m
            best_val_std = val_s
            best_C = C_value

    return best_C, best_val_mean, best_val_std


def train_final_model(X_train, y_train, C_value):
    # Scaling (fit only on train)
    scaler = StandardScaler()
    # fit on train
    X_train_scaled = scaler.fit_transform(X_train)

    # Logistic regression (L2 regularization)
    model = LogisticRegression(max_iter=2000, C=C_value)
    model.fit(X_train_scaled, y_train)
    return scaler, model


def evaluate_model(scaler, model, X_train, y_train, X_test, y_test):

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train ROC- AOC
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    # hard label: precision, recall and F1
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Soft core: ROC-AOC use (ranking)
    y_test_pred = model.predict(X_test_scaled)

    train_roc = roc_auc_score(y_train, y_train_prob)
    test_roc = roc_auc_score(y_test, y_test_prob)
    report = classification_report(y_test, y_test_pred, digits=3)

    return float(train_roc), float(test_roc), report


def save_artifacts(scaler, model, feature_cols):
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "feature_cols.json").write_text(json.dumps(list(feature_cols)))
    print(f"Saved artifacts to {ARTIFACT_DIR}/")


def persist_results(best_C, best_val_mean, best_val_std, test_roc, n_splits=5, random_state=42):
    insert_model_run(
        model_name="logreg_l2",
        dataset_name="breast_cancer_wisconsin",
        n_splits=n_splits,
        best_c=best_C,
        cv_val_mean_roc=best_val_mean,
        cv_val_std_roc=best_val_std,
        test_roc=test_roc,
        random_state=random_state
    )
    print("Saved model run to Postgres (model_runs).")

def main():
    # read csv
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = make_train_test_split(df)

    print("Train class ratio (mean y):", float(y_train.mean()))
    print("Test  class ratio (mean y):", float(y_test.mean()))
    print(df[["radius_mean", "area_mean"]].describe())

    print("\n" + "=" * 60)
    print("Cross-validation selection for C (TRAIN only)")
    best_C, best_val_mean, best_val_std = select_best_c_by_cv(X_train, y_train)

    print(f"\nSelected C = {best_C} (best CV Val ROC mean, stable std)")

    print("\n" + "=" * 60)
    print("Final evaluation on held-out TEST set (used once)")
    scaler, model = train_final_model(X_train, y_train, best_C)
    train_roc, test_roc, report = evaluate_model(scaler, model, X_train, y_train, X_test, y_test)

    print(f"FINAL C={best_C}")
    print(f"Train ROC-AUC: {train_roc:.4f}")
    print(f"Test  ROC-AUC: {test_roc:.4f}")
    print(report)

    persist_results(best_C, best_val_mean, best_val_std, test_roc)
    save_artifacts(scaler, model, X_train.columns)

if __name__ == "__main__":
    main()
