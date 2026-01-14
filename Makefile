.PHONY: help install install-dev test lint format docs clean build release

help:
	@echo "TransitKit Development Commands:"
	@echo ""
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run code quality checks"
	@echo "  format       Format code with black/isort"
	@echo "  docs         Build documentation"
	@echo "  docs-serve   Serve documentation locally"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  release      Create release"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,test]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=transitkit --cov-report=html --cov-report=term-missing

test-all:
	tox

lint:
	flake8 src/transitkit tests/
	black --check src/transitkit tests/
	isort --check-only src/transitkit tests/
	mypy src/transitkit

format:
	black src/transitkit tests/
	isort src/transitkit tests/

docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

build:
	python -m build

release: clean build
	twine upload dist/*

docker-build:
	docker build -t transitkit:latest .

docker-run:
	docker run -p 8501:8501 transitkit:latest

docker-dev:
	docker-compose up --build

pre-commit:
	pre-commit run --all-files

benchmark:
	python benchmarks/benchmark_fitting.py
	python benchmarks/benchmark_loading.py