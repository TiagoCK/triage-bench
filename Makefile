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
		--results $(SMOKE)/latest.json --systems tfidf,difflib,bm25 --sample 60 \
		--min-test-duplicates 5
	uv run triage gate --results $(SMOKE)/latest.json --thresholds configs/smoke_gate.yaml

fixture:
	uv run python $(FIXTURE)/make_fixture.py

# --- live service -------------------------------------------------------------
SERVICE_TEST_DB := postgresql://triage:triage@localhost:5433/postgres

.PHONY: service-db service-test up down

# Start only Postgres (for tests and local runs).
service-db:
	docker compose up -d --wait db

# Service tests against the Docker Postgres; each run uses a throwaway database.
service-test: service-db
	TRIAGE_TEST_DATABASE_URL=$(SERVICE_TEST_DB) uv run --extra service pytest -q tests/service

up:
	docker compose up --build

down:
	docker compose down
