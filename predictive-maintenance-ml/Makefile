# Convenience commands
.PHONY: data features train predict api app test

data:
	python src/data/generate_data.py

features:
	python src/features/build_features.py

train:
	python src/models/train.py

predict:
	python src/models/predict.py --input data/processed/features.csv --output data/predictions/preds.csv

api:
	uvicorn src.app.main:app --reload --port 8000

app:
	streamlit run src/app/streamlit_app.py

test:
	pytest -q
