.PHONY: install test test-ci lint lint-fix clean run-api run-dashboard docker-build docker-run docker-compose-up docker-compose-down

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-ci:
	pytest tests/ -v --cov=src --cov-report=xml

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix:
	ruff check --fix src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache htmlcov

run-api:
	uvicorn src.api.app:app --reload --port 8000

run-dashboard:
	python -m src.dashboard.app

docker-build:
	docker build -t sentiment-dashboard .

docker-run:
	docker run -p 8000:8000 -p 8050:8050 -v $(PWD)/data:/app/data sentiment-dashboard

docker-compose-up:
	docker-compose up --build

docker-compose-down:
	docker-compose down
