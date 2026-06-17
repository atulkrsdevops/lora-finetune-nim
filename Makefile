.PHONY: setup preprocess train evaluate test lint

setup:
	pip install -r requirements.txt

setup-train:
	pip install -r requirements-train.txt

preprocess:
	python -m src.preprocess

train:
	python -m src.train

evaluate:
	python -m src.evaluate

test:
	pytest -q

lint:
	ruff check .
