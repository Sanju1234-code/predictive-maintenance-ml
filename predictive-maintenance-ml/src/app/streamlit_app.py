import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
st.title("🔧 Predictive Maintenance (Next-Hour Failure Prediction)")
st.write("Upload a features CSV or use the included sample to get predictions.")

# --- make paths work on Streamlit Cloud (repo root has a subfolder) ---
ROOT = Path(__file__).resolve().parents[2]          # .../<repo>/predictive-maintenance-ml
bundle = joblib.load(ROOT / "models" / "model.joblib")
model = bundle["model"]
feature_names = bundle["feature_names"]
sample_path = ROOT / "data" / "processed" / "features.csv"

tab1, tab2 = st.tabs(["Batch CSV", "Single Row"])

with tab1:
    uploaded = st.file_uploader("Upload features CSV", type=["csv"])
    if st.button("Predict on Sample (first 500 rows)"):
        df = pd.read_csv(sample_path).head(500)
        X = df[feature_names]
        proba = model.predict_proba(X)[:, 1]
        df_out = df.assign(pred_proba=proba, pred_label=(proba >= 0.5).astype(int))
        st.dataframe(df_out.head(50))
        st.download_button("Download Predictions", data=df_out.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")
    elif uploaded is not None:
        df = pd.read_csv(uploaded)
        X = df[feature_names]
        proba = model.predict_proba(X)[:, 1]
        df_out = df.assign(pred_proba=proba, pred_label=(proba >= 0.5).astype(int))
        st.dataframe(df_out.head(50))
        st.download_button("Download Predictions", data=df_out.to_csv(index=False),
                           file_name="predictions.csv", mime="text/csv")

with tab2:
    st.write("Edit a single row of features and predict:")
    df = pd.read_csv(sample_path, nrows=1)
    X = df[feature_names]
    defaults = X.iloc[0].to_dict()
    inputs = {}
    shown = 0
    for k, v in defaults.items():
        if shown < 15:
            inputs[k] = st.number_input(k, value=float(v))
            shown += 1
    if st.button("Predict single row"):
        full = [inputs.get(name, float(defaults[name])) for name in feature_names]
        proba = float(model.predict_proba([full])[0, 1])
        st.success(f"Predicted failure probability: {proba:.3f}")
