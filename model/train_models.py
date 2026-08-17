"""
train_models.py
----------------
Trains 5 classification models on the UCI Online Shoppers Purchasing
Intention dataset, evaluates each on a held-out test split, and saves:
  - one fitted scikit-learn Pipeline (preprocessing + model) per algorithm,
    as a .pkl file inside model/saved_models/
  - test_data.csv (the raw held-out test rows, unprocessed) at the project
    root, used by the Streamlit app and for grading
  - metrics_summary.csv with the 6 evaluation metrics for every model

Run with:  python model/train_models.py
"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "online_shoppers_intention.csv")
SAVED_MODELS_DIR = os.path.join(HERE, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

TARGET = "Revenue"


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Weekend/Revenue arrive as booleans; keep Weekend as a numeric 0/1
    # feature and encode Revenue (the target) separately below.
    df["Weekend"] = df["Weekend"].astype(int)
    return df


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_data()
    y = df[TARGET].astype(int)  # True/False -> 1/0
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    results = {}

    for name, model in get_models().items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        results[name] = metrics

        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(pipe, os.path.join(SAVED_MODELS_DIR, filename))
        print(f"{name:28s} -> {metrics}")

    # Save metrics table
    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "ML Model Name"
    metrics_df = metrics_df.round(4)
    metrics_df.to_csv(os.path.join(PROJECT_ROOT, "model", "metrics_summary.csv"))
    print("\nSaved metrics_summary.csv:\n", metrics_df)

    # Save the held-out test split (raw, unprocessed columns + target) for the
    # Streamlit app's "upload test data" feature and for grading.
    test_out = X_test.copy()
    test_out[TARGET] = y_test.map({1: True, 0: False}).values
    test_out.to_csv(os.path.join(PROJECT_ROOT, "test_data.csv"), index=False)
    print(f"\nSaved test_data.csv with {len(test_out)} rows.")


if __name__ == "__main__":
    main()
