-- Webhook deliveries already seen. GitHub retries deliveries, so the delivery ID is the
-- idempotency key: a repeat is acknowledged and dropped.
CREATE TABLE webhook_deliveries (
    delivery_id text PRIMARY KEY,
    event       text NOT NULL,
    action      text,
    received_at timestamptz NOT NULL DEFAULT now()
);

-- The work queue. Workers claim jobs with FOR UPDATE SKIP LOCKED, so several workers can
-- run without handing out the same job twice.
CREATE TABLE jobs (
    id           bigserial PRIMARY KEY,
    kind         text NOT NULL,
    payload      jsonb NOT NULL,
    status       text NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'done', 'dead')),
    attempts     integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL,
    run_after    timestamptz NOT NULL DEFAULT now(),
    locked_at    timestamptz,
    locked_by    text,
    last_error   text,
    dedupe_key   text UNIQUE,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX jobs_ready ON jobs (run_after, id) WHERE status = 'queued';

-- Issues as last seen. Later milestones add embeddings, predictions and outcomes.
CREATE TABLE issues (
    repo        text NOT NULL,
    number      integer NOT NULL,
    title       text NOT NULL,
    body        text NOT NULL DEFAULT '',
    state       text NOT NULL,
    labels      jsonb NOT NULL DEFAULT '[]',
    author      text,
    url         text,
    created_at  timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL,
    source      text NOT NULL,           -- 'webhook' or 'poller'
    first_seen  timestamptz NOT NULL DEFAULT now(),
    last_seen   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, number)
);
