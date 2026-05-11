ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
else
    PYTHON := .venv/bin/python
endif

.PHONY: install-dev lint format typecheck test security train eval benchmark thesis check demo-check full-pipeline clean help

help: ## Show this help message
	@$(PYTHON) -c "import re, sys; lines = open('Makefile').readlines(); targets = [(m.group(1), m.group(2)) for l in lines if (m := re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', l))]; [print(f'  \033[36m{t[0]:<24}\033[0m {t[1]}') for t in sorted(targets)]"

install-dev: ## Install package in editable mode with all dependencies
	$(PYTHON) -m pip install -e . -r requirements.txt

lint: ## Check code style with ruff
	$(PYTHON) -m ruff check .

format: ## Verify formatting with ruff (does not modify files)
	$(PYTHON) -m ruff format --check .

typecheck: ## Run mypy static type checker
	$(PYTHON) -m mypy

test: ## Run pytest suite (coverage gate ≥80%)
	$(PYTHON) -m pytest

security: ## Scan source for security issues with bandit
	$(PYTHON) -m bandit -q -r src scripts -s B101 -x tests

train: ## Train model end-to-end (use DATA_PATH=... to override CSV)
	$(PYTHON) scripts/train.py $(if $(DATA_PATH),--data-path $(DATA_PATH),) $(if $(MAX_ROWS),--max-rows $(MAX_ROWS),)

eval: ## Run post-training verification on existing artifacts
	$(PYTHON) scripts/train.py --verify-only

benchmark: ## Generate 16 benchmark CSV tables in reports/benchmarks/
	$(PYTHON) scripts/benchmark.py

thesis: ## Full thesis analysis including SHAP and Optuna
	$(PYTHON) scripts/train.py --thesis

check: ## Run all quality gates (artifacts, metrics, sync, fairness)
	$(PYTHON) scripts/check.py all

demo-check: ## Pre-demo readiness check (artifacts, model load, predictions, live server)
	$(PYTHON) scripts/demo_check.py

full-pipeline: train eval benchmark check ## Full refresh: train → eval → benchmark → check
	@echo "full-pipeline complete — all checks passed."

clean: ## Remove all caches and build artifacts
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(d, ignore_errors=True) for d in ['.mypy_cache','.pytest_cache','.ruff_cache','htmlcov','build','dist'] + [str(p) for p in pathlib.Path('.').glob('*.egg-info')]]; pathlib.Path('.coverage').unlink(missing_ok=True)"
