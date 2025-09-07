import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")

st.title("🔧 Predictive Maintenance (Next‑Hour Failure Prediction)")
st.write("Upload a **features CSV** or use the included sample to get predictions.")

bundle = joblib.load(Path("models/model.joblib"))
model = bundle["model"]
feature_names = bundle["feature_names"]

sample_path = Path("data/processed/features.csv")

tab1, tab2 = st.tabs(["Batch CSV", "Single Row"])

with tab1:
    uploaded = st.file_uploader("Upload features CSV", type=["csv"])    
    if st.button("Predict on Sample (first 500 rows)"):
        df = pd.read_csv(sample_path).head(500)
        X = df[feature_names]
        proba = model.predict_proba(X)[:,1]
        df_out = df.assign(pred_proba=proba, pred_label=(proba>=0.5).astype(int))
        st.dataframe(df_out.head(50))
        st.download_button("Download Predictions", data=df_out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    elif uploaded is not None:
        df = pd.read_csv(uploaded)
        X = df[feature_names]
        proba = model.predict_proba(X)[:,1]
        df_out = df.assign(pred_proba=proba, pred_label=(proba>=0.5).astype(int))
        st.dataframe(df_out.head(50))
        st.download_button("Download Predictions", data=df_out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

with tab2:
    st.write("Edit a single row of features and predict:")
    df = pd.read_csv(sample_path, nrows=1)
    X = df[feature_names]
    defaults = X.iloc[0].to_dict()
    cols = {}
    for k, v in defaults.items():
        # show only a subset to keep UI manageable
        if len(cols) < 15:
            cols[k] = st.number_input(k, value=float(v))
    if st.button("Predict single row"):
        import numpy as np
        # align to full feature vector (fill missing with defaults)
        full = [cols.get(name, float(defaults[name])) for name in feature_names]
        proba = float(model.predict_proba([full])[0,1])
        st.success(f"Predicted failure probability: {proba:.3f}")
