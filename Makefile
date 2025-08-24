# Makefile for pyins development

.PHONY: help install install-dev test test-cov test-fast lint format type-check security clean build docs serve-docs

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install the package in production mode
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run all tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage report
	pytest tests/ \
		--cov=pyins \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=xml

test-fast:  ## Run tests in parallel for speed
	pytest tests/ -n auto --dist loadscope

test-unit:  ## Run only unit tests
	pytest tests/ -m unit -v

test-integration:  ## Run only integration tests
	pytest tests/ -m integration -v

lint:  ## Run linting checks
	ruff check pyins tests
	ruff format --check pyins tests

format:  ## Auto-format code
	ruff check --fix pyins tests
	ruff format pyins tests

type-check:  ## Run type checking with mypy
	mypy pyins --ignore-missing-imports

security:  ## Run security checks
	bandit -r pyins -ll
	safety check

clean:  ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

build:  ## Build distribution packages
	python -m build

docs:  ## Build documentation
	cd docs && make html

serve-docs:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server

pre-commit:  ## Run pre-commit on all files
	pre-commit run --all-files

bump-patch:  ## Bump patch version (x.x.X)
	bump2version patch

bump-minor:  ## Bump minor version (x.X.x)
	bump2version minor

bump-major:  ## Bump major version (X.x.x)
	bump2version major

# Development workflow shortcuts
dev-test: format lint type-check test  ## Run full development test suite

ci-test: lint test-cov security  ## Run CI test suite locally

release-check: clean build test-cov  ## Check before release

# Installation shortcuts
reinstall: clean install-dev  ## Clean and reinstall with dev dependencies