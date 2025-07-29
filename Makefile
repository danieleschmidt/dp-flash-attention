.PHONY: help install install-dev test test-gpu test-privacy lint format clean build docs
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in development mode
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev,test,docs,cuda]"
	pre-commit install

test: ## Run unit tests
	pytest tests/unit/ -v

test-gpu: ## Run GPU-specific tests (requires CUDA)
	pytest tests/ -m gpu -v

test-privacy: ## Run privacy-specific tests
	pytest tests/privacy/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-all: ## Run all tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=dp_flash_attention --cov-report=html --cov-report=term

benchmark: ## Run performance benchmarks
	pytest tests/benchmarks/ -v --benchmark-only

lint: ## Run linting checks
	ruff check src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	ruff check --fix src/ tests/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build package
	python -m build

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

setup-cuda: ## Set up CUDA environment
	@echo "Setting up CUDA development environment..."
	scripts/setup_cuda.sh

setup-dev: ## Set up development environment
	@echo "Setting up development environment..."
	scripts/setup_dev.sh

privacy-audit: ## Run privacy audit
	dp-privacy-audit --config privacy_config.yaml

security-scan: ## Run security scanning
	bandit -r src/
	safety check