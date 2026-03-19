# Makefile for Recipe ML Project

.PHONY: help install install-dev test test-cov lint format clean docker-build docker-up docker-down download-data pipeline app

help:
	@echo "Available commands:"
	@echo "  make install          - Install project dependencies"
	@echo "  make install-dev      - Install dev dependencies"
	@echo "  make test             - Run tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make lint             - Run linting"
	@echo "  make format           - Format code"
	@echo "  make clean            - Clean generated files"
	@echo "  make download-data    - Download Kaggle dataset"
	@echo "  make pipeline         - Run full ML pipeline"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-up        - Start Docker containers"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test,sentiment]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

download-data:
	python scripts/download_kaggle_data.py

pipeline: download-data
	python scripts/clean_data.py
	python scripts/run_recipe_pipeline.py
	python scripts/run_nutrition_pipeline.py
	python main_nutrition_tagging.py
	python main_time_prediction.py
	python train_sentiment_model.py

app:
	streamlit run streamlit_app_final.py

docker-build:
	docker build -t recipe-ml:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
