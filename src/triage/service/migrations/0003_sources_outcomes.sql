-- How far the poller has read each watched repo. It only advances in the same transaction
-- that enqueues the new issues, so a failed run never skips an issue.
CREATE TABLE poll_cursors (
    repo       text NOT NULL,
    kind       text NOT NULL,
    cursor     timestamptz NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, kind)
);

-- Which GitHub App installation covers which repo, learned from webhook payloads.
CREATE TABLE installations (
    repo            text PRIMARY KEY,
    installation_id bigint NOT NULL,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- What maintainers later did with each predicted issue, as of the last check. Week 4's
-- live metrics compare predictions against these rows.
CREATE TABLE outcomes (
    repo         text NOT NULL,
    number       integer NOT NULL,
    checked_at   timestamptz NOT NULL DEFAULT now(),
    state        text NOT NULL,           -- OPEN, CLOSED, or MISSING (deleted/transferred)
    state_reason text,
    closed_at    timestamptz,
    labels       jsonb NOT NULL DEFAULT '[]',
    label_events jsonb NOT NULL DEFAULT '[]',
    dup_links    jsonb NOT NULL DEFAULT '[]',
    dup_mentions jsonb NOT NULL DEFAULT '[]',
    duplicate_of integer,                 -- resolved with the benchmark's orientation rules
    oriented_by  text,
    PRIMARY KEY (repo, number),
    FOREIGN KEY (repo, number) REFERENCES issues ON DELETE CASCADE
);
