# Smart Document Assistant - Makefile
# Unified commands for development and testing

.PHONY: help install install-test test test-all test-coverage test-clean lint format clean docker-build docker-run

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Smart Document Assistant - Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install       Install production dependencies with uv"
	@echo "  make install-test  Install test dependencies with uv"
	@echo "  make install-dev   Install development dependencies with uv"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test          Run unit tests (fast)"
	@echo "  make test-all      Run all tests including slow ones"
	@echo "  make test-coverage Run tests with coverage report"
	@echo "  make test-category CATEGORY=document  Run specific category"
	@echo ""
	@echo "$(GREEN)Test Categories:$(NC)"
	@echo "  make test-document  Document parsing tests"
	@echo "  make test-vector    Vector operation tests"
	@echo "  make test-embedding Embedding tests"
	@echo "  make test-storage   Storage tests"
	@echo "  make test-retrieval Retrieval tests"
	@echo "  make test-prompt    Prompt generation tests"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make format        Format code (black)"
	@echo "  make type-check    Run type checker (mypy)"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean         Clean cache and temp files"
	@echo "  make clean-all     Clean everything including venv"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"

# Setup commands
install: ## Install production dependencies
	@echo "$(YELLOW)📦 Installing production dependencies with uv...$(NC)"
	cd backend && uv pip install -e .

install-test: ## Install test dependencies
	@echo "$(YELLOW)📦 Installing test dependencies with uv...$(NC)"
	cd backend && uv pip install -e ".[test]"

install-dev: ## Install development dependencies
	@echo "$(YELLOW)📦 Installing development dependencies with uv...$(NC)"
	cd backend && uv pip install -e ".[dev]"

# Test commands
test: ## Run unit tests (excludes slow tests)
	@echo "$(YELLOW)🧪 Running unit tests...$(NC)"
	cd backend && uv run pytest tests/unit -v --tb=short -m "not slow"

test-all: ## Run all tests including slow tests
	@echo "$(YELLOW)🧪 Running all tests...$(NC)"
	cd backend && uv run pytest tests/unit -v --tb=short

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)📊 Running tests with coverage...$(NC)"
	cd backend && uv run pytest tests/unit -v --tb=short \
		--cov=app --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✅ Coverage report: backend/htmlcov/index.html$(NC)"

test-category: ## Run tests by category (usage: make test-category CATEGORY=document)
	@echo "$(YELLOW)🧪 Running $(CATEGORY) tests...$(NC)"
	cd backend && uv run pytest tests/unit -v --tb=short -m "$(CATEGORY)"

# Test category shortcuts
test-document: ## Run document parsing tests
	@$(MAKE) test-category CATEGORY=document

test-vector: ## Run vector operation tests
	@$(MAKE) test-category CATEGORY=vector

test-embedding: ## Run embedding tests
	@$(MAKE) test-category CATEGORY=embedding

test-storage: ## Run storage tests
	@$(MAKE) test-category CATEGORY=storage

test-retrieval: ## Run retrieval tests
	@$(MAKE) test-category CATEGORY=retrieval

test-prompt: ## Run prompt generation tests
	@$(MAKE) test-category CATEGORY=prompt

test-clean: ## Clean test cache
	@echo "$(YELLOW)🧹 Cleaning test cache...$(NC)"
	cd backend && rm -rf .pytest_cache htmlcov .coverage
	find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Test cache cleaned!$(NC)"

# Code quality commands
lint: ## Run linter
	@echo "$(YELLOW)🔍 Running linter...$(NC)"
	cd backend && uv run ruff check app tests

format: ## Format code
	@echo "$(YELLOW)📝 Formatting code...$(NC)"
	cd backend && uv run black app tests
	cd backend && uv run ruff format app tests

type-check: ## Run type checker
	@echo "$(YELLOW)🔍 Running type checker...$(NC)"
	cd backend && uv run mypy app

# Maintenance commands
clean: ## Clean cache and temp files
	@echo "$(YELLOW)🧹 Cleaning cache files...$(NC)"
	cd backend && rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find backend -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Cache cleaned!$(NC)"

clean-all: ## Clean everything including virtual environment
	@echo "$(RED)🧹 Cleaning everything...$(NC)"
	cd backend && rm -rf .venv .venv-test __pycache__ .pytest_cache htmlcov .coverage
	find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find backend -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Everything cleaned!$(NC)"

# Docker commands
docker-build: ## Build Docker image
	@echo "$(YELLOW)🐳 Building Docker image...$(NC)"
	docker build -t smart-document-assistant:latest .

docker-run: ## Run Docker container
	@echo "$(YELLOW)🐳 Running Docker container...$(NC)"
	docker run -p 8000:8000 -v $(PWD)/uploads:/app/uploads smart-document-assistant:latest

# CI/CD helper
ci-test: ## Run tests for CI (no colors, xml output)
	@echo "Running tests for CI..."
	cd backend && uv run pytest tests/unit -v --tb=short \
		--cov=app --cov-report=xml --cov-report=term \
		-m "not slow"

# Development server
run: ## Run development server
	@echo "$(YELLOW)🚀 Starting development server...$(NC)"
	cd backend && uv run python main.py

run-dev: ## Run with auto-reload
	@echo "$(YELLOW)🚀 Starting development server with reload...$(NC)"
	cd backend && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
