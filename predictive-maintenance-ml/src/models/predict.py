"""Batch predict from CSV using trained model."""
import argparse
from pathlib import Path
import joblib
import pandas as pd

MODEL = Path("models/model.joblib")

def main(input_csv: str, output_csv: str):
    bundle = joblib.load(MODEL)
    model = bundle["model"]
    feat_names = bundle["feature_names"]
    df = pd.read_csv(input_csv)
    X = df[feat_names]
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    out = df.copy()
    out["pred_proba"] = proba
    out["pred_label"] = preds
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote predictions to {output_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to features CSV")
    ap.add_argument("--output", required=True, help="Path to write predictions CSV")
    args = ap.parse_args()
    main(args.input, args.output)
