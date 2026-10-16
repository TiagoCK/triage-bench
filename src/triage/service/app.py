"""The HTTP side of the service: the GitHub webhook receiver and a health check.

The receiver does as little as possible: verify the signature, drop repeat deliveries,
enqueue a job, and answer 202. GitHub expects a reply within 10 seconds, so all real work
happens in workers.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool

from triage.service import queue
from triage.service.db import make_pool, migrate
from triage.service.settings import Settings, get_settings
from triage.service.webhooks import job_for_event, verify_signature

log = logging.getLogger(__name__)


async def raw_body(request: Request) -> bytes:
    """The unparsed body: the signature covers these exact bytes."""
    return await request.body()


RawBody = Annotated[bytes, Depends(raw_body)]


def create_app(settings: Settings | None = None, pool: ConnectionPool | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = pool is None
        app.state.pool = pool or make_pool(settings.database_url)
        with app.state.pool.connection() as conn:
            migrate(conn)
        yield
        if owned:
            app.state.pool.close()

    app = FastAPI(title="triage-bench service", lifespan=lifespan)
    app.state.settings = settings

    @app.post("/webhooks/github", status_code=202)
    def github_webhook(request: Request, body: RawBody) -> JSONResponse:
        secret = settings.webhook_secret
        if secret is None:
            raise HTTPException(503, "webhook secret not configured")
        if len(body) > settings.max_webhook_bytes:
            raise HTTPException(413, "payload too large")
        if not verify_signature(
            secret.get_secret_value(), body, request.headers.get("X-Hub-Signature-256")
        ):
            raise HTTPException(401, "invalid signature")
        delivery = request.headers.get("X-GitHub-Delivery")
        event = request.headers.get("X-GitHub-Event")
        if not delivery or not event:
            raise HTTPException(400, "missing GitHub delivery headers")
        if event == "ping":
            return JSONResponse({"ok": True, "event": "ping"}, status_code=200)

        payload = _json(body)
        job = job_for_event(event, payload)
        with app.state.pool.connection() as conn, conn.transaction():
            # Recording the delivery and enqueueing happen together, so a crash can't leave
            # a delivery marked as seen with no job behind it.
            first_time = conn.execute(
                "INSERT INTO webhook_deliveries (delivery_id, event, action)"
                " VALUES (%s, %s, %s) ON CONFLICT DO NOTHING RETURNING delivery_id",
                (delivery, event, payload.get("action")),
            ).fetchone()
            if not first_time:
                return JSONResponse({"duplicate": True}, status_code=200)
            if job is None:
                return JSONResponse({"ignored": True}, status_code=200)
            kind, job_payload = job
            job_payload["source"] = "webhook"
            job_id = queue.enqueue(
                conn, kind, job_payload, settings.job_max_attempts, dedupe_key=delivery
            )
        log.info(
            "queued job %s for %s/%s delivery %s", job_id, event, payload.get("action"), delivery
        )
        return JSONResponse({"queued": job_id}, status_code=202)

    @app.get("/healthz")
    def healthz() -> dict:
        with app.state.pool.connection() as conn:
            conn.execute("SELECT 1")
            jobs = queue.counts(conn)
        return {"status": "ok", "mode": settings.mode, "jobs": jobs}

    return app


def _json(body: bytes) -> dict:
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise HTTPException(400, "body is not JSON") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "body is not a JSON object")
    return data
