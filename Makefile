FIXTURE := tests/fixtures
SMOKE := .cache/smoke

.PHONY: install install-all test lint smoke fixture

install:
	uv sync

install-all:
	uv sync --all-extras

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

# Offline end-to-end run on the synthetic fixture, then the regression gate. Same as CI.
smoke:
	uv run triage build --config $(FIXTURE)/repos.yaml --raw-dir $(FIXTURE)/raw --out-dir $(SMOKE)/processed
	uv run triage eval --config $(FIXTURE)/repos.yaml --processed-dir $(SMOKE)/processed \
		--results $(SMOKE)/latest.json --systems tfidf,difflib,bm25 --sample 60
	uv run triage gate --results $(SMOKE)/latest.json --thresholds configs/smoke_gate.yaml

fixture:
	uv run python $(FIXTURE)/make_fixture.py
