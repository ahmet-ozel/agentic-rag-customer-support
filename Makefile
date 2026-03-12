.PHONY: help dev test test-unit test-integration docker-up docker-down docker-gpu lint clean

help:
	@echo "AgentDesk — Available commands:"
	@echo "  make dev          Run FastAPI server locally (hot-reload)"
	@echo "  make ui           Run Gradio UI"
	@echo "  make test         Run all tests"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make test-int     Run integration tests only"
	@echo "  make docker-up    Start all services (CPU mode)"
	@echo "  make docker-gpu   Start all services including vLLM (GPU mode)"
	@echo "  make docker-down  Stop all services"
	@echo "  make lint         Run ruff linter"
	@echo "  make clean        Remove __pycache__ and .pytest_cache"

dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

ui:
	python frontend/gradio_app.py

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit -v

test-int:
	pytest tests/integration -v

docker-up:
	docker compose up -d

docker-gpu:
	docker compose --profile gpu up -d

docker-down:
	docker compose down

lint:
	ruff check src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
