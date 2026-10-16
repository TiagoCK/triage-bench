# syntax=docker/dockerfile:1
# Image for the live service (api and worker). Only the `service` extra is installed: no
# torch, no LiteLLM, so the image stays small.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md LICENSE ./
# uv's download cache lives in a BuildKit cache mount, not in the image layers.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra service --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra service

RUN useradd --create-home --uid 1000 app
USER app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["triage", "service", "api"]
