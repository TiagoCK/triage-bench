CREATE EXTENSION IF NOT EXISTS vector;

-- One embedding per issue, of its latest text. text_hash lets the worker skip re-embedding
-- when an event (a label change, a comment) didn't change the text.
CREATE TABLE issue_embeddings (
    repo       text NOT NULL,
    number     integer NOT NULL,
    model      text NOT NULL,
    text_hash  text NOT NULL,
    embedding  vector(384) NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, number),
    FOREIGN KEY (repo, number) REFERENCES issues ON DELETE CASCADE
);

-- What the bot would have said when the issue was opened. One row per issue and kind, so
-- retried jobs and repeat deliveries can't double-count. Versions make every prediction
-- traceable to the model and code that produced it.
CREATE TABLE predictions (
    id               bigserial PRIMARY KEY,
    repo             text NOT NULL,
    number           integer NOT NULL,
    kind             text NOT NULL DEFAULT 'open',
    mode             text NOT NULL,
    made_at          timestamptz NOT NULL DEFAULT now(),
    issue_created_at timestamptz NOT NULL,
    candidates       jsonb NOT NULL,      -- [{"number": n, "similarity": s}], best first
    labels           jsonb,               -- suggested labels, or NULL without a labeler
    label_scores     jsonb,               -- {"label": probability}
    embed_model      text NOT NULL,
    labeler_version  text,
    code_version     text NOT NULL,
    timings_ms       jsonb NOT NULL,
    UNIQUE (repo, number, kind),
    FOREIGN KEY (repo, number) REFERENCES issues ON DELETE CASCADE
);
