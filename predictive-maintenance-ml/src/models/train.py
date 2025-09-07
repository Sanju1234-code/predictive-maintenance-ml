"""Train a classifier to predict failure in the next hour."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

DATA = Path("data/processed/features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def main():
    df = pd.read_csv(DATA)
    # features
    y = df["failure_next_1h"].astype(int)
    X = df.drop(columns=["failure_next_1h"])  # keep other columns including original 'failure'
    # select numeric columns for pipeline
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # drop non-numeric that model can't use
    drop_cols = [c for c in X.columns if c not in num_cols]
    X = X[num_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

    pre = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False))  # sparse-safe
    ])

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=7,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    pipe = Pipeline([
        ("prep", pre),
        ("clf", clf)
    ])

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "report": classification_report(y_test, y_pred, output_dict=False)
    }

    # Save model and metrics
    joblib.dump({"model": pipe, "feature_names": X.columns.tolist()}, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Feature importances (from RF inside pipeline)
    importances = pipe.named_steps["clf"].feature_importances_
    fi = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
    fi.to_csv(MODEL_DIR / "feature_importances.csv", index=False)

    print("Training complete. Metrics:\n", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
