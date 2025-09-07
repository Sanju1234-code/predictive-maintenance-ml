@echo off
python -m venv .venv
call .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python src/models/train.py
streamlit run src/app/streamlit_app.py
