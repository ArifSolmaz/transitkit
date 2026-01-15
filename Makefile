.PHONY: install dev test lint format clean build

install:
	pip install -e .

dev:
	pip install -e ".[dev,full]"

test:
	pytest tests/ -v --cov=src/transitkit

lint:
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

build: clean
	python -m build
