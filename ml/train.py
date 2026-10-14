import json
import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
csv_path = ROOT / config["csv_path"]
out_dir = ROOT / "artifacts"
out_dir.mkdir(exist_ok=True)

if not csv_path.exists():
    raise SystemExit(
        f"Dataset not found: {csv_path}\n"
        "Add your authorized competition CSV, then update ml/config.json with the real target column names."
    )

df = pd.read_csv(csv_path)
target_map = config["target_columns"]
missing_targets = [name for name in target_map.values() if name not in df.columns]
if missing_targets:
    raise SystemExit(f"Configured target columns missing from CSV: {missing_targets}. Edit config.json.")

# Required leakage guard from the Track A brief.
excluded = set(config.get("leakage_columns", [])) | set(target_map.values())
feature_cols = [c for c in df.columns if c not in excluded]
if not feature_cols:
    raise SystemExit("No feature columns remain after leakage exclusions.")

X = df[feature_cols].copy()
# Remove columns that are completely empty.
X = X.dropna(axis=1, how="all")
if X.empty or X.shape[1] == 0:
    raise SystemExit("No usable features after removing empty columns.")

numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
    ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
])

def normalize_target(series):
    # Map common binary encodings to 0/1; stop rather than silently guess multi-class labels.
    s = series.dropna()
    unique = list(pd.unique(s))
    if len(unique) != 2:
        raise ValueError(f"Expected a binary target, got values {unique[:20]}")
    mapping = {}
    for value in unique:
        key = str(value).strip().lower()
        if key in {"1", "yes", "true", "positive", "present", "cad", "disease"}:
            mapping[value] = 1
        elif key in {"0", "no", "false", "negative", "absent", "normal", "no cad", "no disease"}:
            mapping[value] = 0
    if len(mapping) != 2 or len(set(mapping.values())) != 2:
        # If dataset uses numeric 1/2 labels, map the larger numeric value to positive and record it.
        try:
            vals = sorted(float(v) for v in unique)
            if len(vals) == 2:
                mapping = {unique[0]: int(float(unique[0]) == vals[1]), unique[1]: int(float(unique[1]) == vals[1])}
            else:
                raise ValueError
        except Exception as exc:
            raise ValueError(f"Unrecognized target coding {unique}; explicitly map it in normalize_target().") from exc
    return series.map(mapping), {str(k): int(v) for k, v in mapping.items()}

metrics = {
    "dataset_rows": int(len(df)),
    "features_used": list(X.columns),
    "excluded_from_features": sorted(excluded),
    "random_state": int(config.get("random_state", 42)),
    "test_size": float(config.get("test_size", 0.25)),
    "targets": {},
    "warning": "Small-dataset test estimates may be unstable; not clinical validation."
}

for task, target_col in target_map.items():
    y, label_mapping = normalize_target(df[target_col])
    valid = y.notna()
    Xi, yi = X.loc[valid], y.loc[valid].astype(int)
    if yi.nunique() != 2 or yi.value_counts().min() < 2:
        warnings.warn(f"Skipping {task}: insufficient examples in one class.")
        continue
    X_train, X_test, y_train, y_test = train_test_split(
        Xi, yi, test_size=config.get("test_size", 0.25),
        random_state=config.get("random_state", 42), stratify=yi
    )
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    result = {
        "target_column": target_col,
        "label_mapping": label_mapping,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_balance_full": {str(k): int(v) for k, v in yi.value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() == 2 else None
    }
    metrics["targets"][task] = result
    joblib.dump(model, out_dir / f"model_{task}.joblib")

(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
print(f"\nSaved model artifacts and metrics to {out_dir}")
