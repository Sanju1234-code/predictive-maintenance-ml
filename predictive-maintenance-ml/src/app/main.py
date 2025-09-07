from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

bundle = joblib.load(Path("models/model.joblib"))
model = bundle["model"]
feature_names = bundle["feature_names"]

class FeatureVector(BaseModel):
    values: list[float]

@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /predict with FeatureVector"}

@app.post("/predict")
def predict(payload: FeatureVector):
    x = np.array(payload.values).reshape(1, -1)
    proba = float(model.predict_proba(x)[0,1])
    return {"pred_proba": proba, "pred_label": int(proba >= 0.5), "feature_count": len(feature_names)}
