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

data: ## Download raw corpora and verify checksums (OFFLINE=1: verify only)
	@test -x .venv/bin/python || { echo "make data: no .venv yet. Run 'make setup' once while online."; exit 1; }
	@# --no-sync: checking files only needs pyyaml + pydantic, so a pull that changes the
	@# dependencies or .python-version can't make this need the network or rebuild .venv
	uv run --no-sync python -m sandhi_probe.data.manifest $(if $(filter 1 yes true,$(OFFLINE)),--offline)
	@# extraction into data/processed goes here (#23)

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
