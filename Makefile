.PHONY: setup data train train-hpsearch evaluate export serve docker-build docker-serve test lint clean

setup:
	pip install -e ".[dev]"

data:
	python -m src.data.download --config configs/default.yaml

train:
	python -m src.training.trainer --config configs/default.yaml

train-hpsearch:
	python -m src.training.trainer --config configs/default.yaml --hpsearch

evaluate:
	python -m src.evaluation.metrics --config configs/default.yaml

export:
	python -m src.serving.export_onnx --config configs/default.yaml

serve:
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose build

docker-serve:
	docker-compose up serve mlflow

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

clean:
	rm -rf mlruns/ outputs/ checkpoints/ __pycache__ .pytest_cache
