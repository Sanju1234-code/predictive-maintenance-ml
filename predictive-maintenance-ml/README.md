# Predictive Maintenance (End‑to‑End ML Project)

A production‑minded ML project suitable for a **3‑year experience** portfolio. It predicts whether a machine will **fail in the next hour** from sensor telemetry.

## Highlights (resume‑friendly)
- Clean **src/** layout with modular scripts (data → features → train → predict).
- **Reproducible**: `Makefile` and simple commands.
- **FastAPI** inference API and **Streamlit** demo app.
- **Unit tests** with `pytest`.
- **Dockerfile** for containerized serve.
- **GitHub Actions CI** (runs tests on push).
- Small synthetic dataset included; no downloads required.

## Quickstart

### 1) Create and activate a virtual environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux (bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) (Optional) Regenerate data
```bash
python src/data/generate_data.py
python src/features/build_features.py
```

### 4) Train
```bash
python src/models/train.py
```

### 5) Batch predict (CSV → predictions)
```bash
python src/models/predict.py --input data/processed/features.csv --output data/predictions/preds.csv
```

### 6) Serve the model via API
```bash
uvicorn src.app.main:app --reload --port 8000
# Then POST to http://127.0.0.1:8000/predict
```

### 7) Streamlit demo UI
```bash
streamlit run src/app/streamlit_app.py
```

## Project Structure
```
predictive-maintenance-ml/
├─ data/
│  ├─ raw/events.csv
│  ├─ processed/features.csv
│  └─ predictions/
├─ models/                # trained artifacts (.joblib, metrics)
├─ src/
│  ├─ data/generate_data.py
│  ├─ features/build_features.py
│  ├─ models/train.py
│  ├─ models/predict.py
│  └─ app/
│     ├─ main.py          # FastAPI
│     └─ streamlit_app.py
├─ tests/test_basic.py
├─ requirements.txt
├─ Dockerfile
├─ Makefile
└─ .github/workflows/ci.yml
```

## What to talk about in interviews
- Problem framing (early failure warning) and **label definition** (failure in next hour using forward shift).
- **Feature engineering**: rolling means/std, deltas, recent context windows.
- **Imbalanced classification** strategy (class_weight='balanced', evaluation with ROC‑AUC/PR‑AUC).
- **Pipeline** design, versioned artifacts, API + UI demo, and basic CI.

## License
MIT
