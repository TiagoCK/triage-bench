"""The live triage service (Phase 2).

It receives GitHub issue events, queues them in Postgres, and processes them in workers. It
runs in shadow mode: it records what it would do and never posts to GitHub.
"""
