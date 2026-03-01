ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
else
    PYTHON := .venv/bin/python
endif

.PHONY: install-dev lint format typecheck test security deps-audit train eval benchmark repro-check artifact-check metrics-gate sync-check full-pipeline run-notebooks fairness-check thesis-analysis thesis-analysis-fast clean coverage-html help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}' \
	  | sort

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

deps-audit: ## Audit dependencies for known vulnerabilities
	$(PYTHON) -m pip_audit -r requirements.txt --no-deps --disable-pip

train: ## Train model end-to-end (use DATA_PATH=... to override CSV)
	$(PYTHON) scripts/train.py $(if $(DATA_PATH),--data-path $(DATA_PATH),) $(if $(MAX_ROWS),--max-rows $(MAX_ROWS),)

eval: ## Run post-training verification report
	$(PYTHON) scripts/verify.py

benchmark: ## Generate 16 benchmark CSV tables in reports/benchmarks/
	$(PYTHON) scripts/benchmark.py

repro-check: ## Reproduce 5k-row subset and verify hash stability
	$(PYTHON) scripts/repro.py --max-rows 5000

artifact-check: ## Validate artifacts integrity, hashes, and smoke prediction
	$(PYTHON) scripts/check_artifacts.py

metrics-gate: ## Enforce metric quality gates (fails if model is below bar)
	$(PYTHON) scripts/check_metrics.py

sync-check: ## Verify thresholds are consistent across artifacts and reports
	$(PYTHON) scripts/sync_check.py

full-pipeline: train eval benchmark artifact-check sync-check ## Full refresh: train → eval → benchmark → artifact-check → sync-check
	@echo "full-pipeline complete — all checks passed."

thesis-analysis: train ## Full thesis analysis including SHAP and Optuna
	$(PYTHON) scripts/thesis.py

thesis-analysis-fast: train ## Thesis analysis skipping Optuna and SHAP (faster)
	$(PYTHON) scripts/thesis.py --skip-tuning --skip-shap

run-notebooks: ## Execute all 8 notebooks headlessly and validate outputs
	$(PYTHON) scripts/notebooks.py

fairness-check: ## Run hyperparameter fairness audit (LightGBM vs XGBoost budget check)
	$(PYTHON) scripts/fairness_check.py

clean: ## Remove all caches and build artifacts
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov *.egg-info build dist

coverage-html: ## Generate HTML coverage report (open htmlcov/index.html)
	$(PYTHON) -m pytest --cov-report=html
