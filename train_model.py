import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Helpers --------------------------------------------------------------

def drop_constant_and_id(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    """Drop id-like columns and columns with a single value."""
    to_drop = []
    for c in id_cols:
        if c in df.columns:
            to_drop.append(c)
    for c in df.columns:
        if df[c].nunique(dropna=False) <= 1:
            to_drop.append(c)
    to_drop = list(sorted(set(to_drop)))
    if to_drop:
        print(f"Dropping ID/constant columns: {to_drop}")
    return df.drop(columns=to_drop, errors="ignore")


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Drop columns with missing ratio > threshold."""
    miss_ratio = df.isna().mean()
    keep_cols = [c for c in df.columns if miss_ratio.get(c, 0.0) <= threshold]
    dropped = [c for c in df.columns if c not in keep_cols]
    if dropped:
        print(f"Dropping high-missing cols (> {threshold*100:.0f}%): {dropped}")
    return df[keep_cols]


def cap_outliers_iqr(X: pd.DataFrame, num_cols: List[str], iqr_factor: float = 1.5) -> pd.DataFrame:
    """Cap numeric columns at [Q1 - k*IQR, Q3 + k*IQR] to reduce extreme outliers."""
    Xc = X.copy()
    for col in num_cols:
        if col not in Xc.columns:
            continue
        s = pd.to_numeric(Xc[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        low = q1 - iqr_factor * iqr
        high = q3 + iqr_factor * iqr
        s = s.clip(lower=low, upper=high)
        Xc[col] = s
    return Xc


def make_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer that imputes, caps outliers, and encodes categoricals."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor, num_cols, cat_cols

# --- Main ----------------------------------------------------------------

def main(args):
    data_path = Path(args.data_path)
    assert data_path.exists(), f"Dataset not found at {data_path}"

    df = pd.read_csv(data_path)
    df = df.sample(n=200_000, random_state=42)  # use 200k rows instead of 2.1M
    print(f"Training on a subset of {len(df):,} rows for speed.")

    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Basic cleanup: drop IDs/constants & duplicates
    df = drop_constant_and_id(df, id_cols=["user_id", "USER_ID", "id", "ID"])
    df = df.drop_duplicates()

    # Target
    target_col = "CHURN"
    assert target_col in df.columns, f"Target column '{target_col}' not found."
    y = df[target_col].astype(int)

    # Features
    X = df.drop(columns=[target_col])
    X = drop_high_missing(X, threshold=0.8)

    #  profiling
    if args.profile:
        try:
            from ydata_profiling import ProfileReport
            os.makedirs("reports", exist_ok=True)
            prof = ProfileReport(pd.concat([X, y.rename(target_col)], axis=1),
                                 title="Expresso Churn Profiling", minimal=True)
            prof.to_file("reports/profile_report.html")
            print("Wrote reports/profile_report.html")
        except Exception as e:
            print("Profiling failed (check ydata-profiling install).", e)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor, num_cols, cat_cols = make_preprocessor(X_train)

    # Classifier (robust baseline for large tabular data)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    # Full pipeline = preprocessing + model
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ])

    if args.fast:
        print("FAST mode enabled: skipping cross-validation for speed.")
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            auc_scores = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=1)
            print(f"CV ROC-AUC: mean={auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
        except Exception as e:
            print("CV failed:", e)

    # Fit & evaluate
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"Test ROC-AUC: {auc:.4f}")
        except Exception:
            pass

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred, digits=4))

    # Save pipeline + metadata
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipe, "artifacts/model.joblib")

    meta = {
        "features_after_drop": X.columns.tolist(),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "target": target_col
    }
    import json
    with open("artifacts/metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved artifacts/model.joblib and artifacts/metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip cross-validation for faster training"
    )

    parser.add_argument("--data-path", type=str, default="data/Expresso_churn_dataset.csv")
    parser.add_argument("--profile", action="store_true", help="Generate ydata-profiling HTML report")
    args = parser.parse_args()
    main(args)
