.DEFAULT_GOAL := help
.PHONY: help setup data check format figures clean

UV := $(shell command -v uv 2>/dev/null)

help: ## Show this list
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

setup: ## Create .venv and install pinned deps from uv.lock
ifndef UV
	$(error uv is not installed. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh)
endif
	uv sync --locked

data: ## Download raw corpora, verify checksums, build data/processed
	@echo "make data: not implemented yet."
	@echo "  manifest + checksum check -> #17, corpora -> #21, extraction -> #23"

check: ## Lint, format check and tests (run before every PR)
	uv run ruff check .
	uv run ruff format --check .
	@# type checker goes here (#16)
	uv run pytest

format: ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

figures: ## Regenerate every figure from data/processed
	@echo "make figures: not implemented yet (no results to plot)."

clean: ## Remove caches and build artifacts (keeps data/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
