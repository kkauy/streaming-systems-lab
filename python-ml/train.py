# python-ml/train.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report



def main():
    # ----------------------------
    # 1) read csv
    # ----------------------------

    df = pd.read_csv("data/breast_cancer.csv")

    # print data
    print(df.head())
    print(df.columns)

    # drop empty column
    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    # ----------------------------
    # 2) Label encode：Convert M/B to 1 and 0
    # ----------------------------
    # diagnosis ：M=malignant, B=benign
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

   # 3) feature and target
    X = df.drop(columns=["id", "diagnosis"])
    y= df["diagnosis"]

    #print(df.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )


    print(df.describe())
    print(df[["radius_mean", "area_mean"]].describe())


    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #  Logistic Regression
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)


    #ROC-AUC
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


if __name__ == "__main__":
    main()
