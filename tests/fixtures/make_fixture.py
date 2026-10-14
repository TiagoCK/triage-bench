"""Generate the small synthetic repository used by unit tests and the CI smoke eval.

It is not meant to be realistic text; it exists so the full pipeline (build -> eval -> gate)
runs offline in seconds with known structure: topical labels, and duplicates that are
rephrasings of an earlier issue. Re-run with `uv run python tests/fixtures/make_fixture.py`.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

OUT = Path(__file__).parent / "raw"
REPO = "acme/widgets"

TOPICS = {
    "area: install": [
        "pip install fails on {os} with wheel build error",
        "cannot install package behind corporate proxy on {os}",
        "installer hangs resolving dependencies for python {py}",
    ],
    "area: cli": [
        "cli flag --{flag} is ignored when config file present",
        "command line help text for {flag} is wrong",
        "cli crashes with traceback when {flag} is empty",
    ],
    "area: cache": [
        "cache directory grows without bound on {os}",
        "stale cache entries returned after upgrade to python {py}",
        "cache lock timeout when running two processes on {os}",
    ],
    "area: docs": [
        "docs page for {flag} has broken example",
        "missing documentation for environment variable {flag}",
    ],
}
KIND = {"bug": 0.6, "enhancement": 0.25, "question": 0.15}
SLOTS = {
    "os": ["windows", "macos", "linux", "alpine", "wsl"],
    "py": ["3.10", "3.11", "3.12", "3.13"],
    "flag": ["verbose", "offline", "index-url", "no-cache", "target", "python", "quiet"],
}
FILLER = [
    "Steps to reproduce are below.",
    "This started after the latest release.",
    "I tried clearing everything and reinstalling.",
    "Logs attached.",
    "Expected it to just work.",
]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(**{k: rng.choice(v) for k, v in SLOTS.items()})


def main() -> None:
    rng = random.Random(7)
    # Separate stream for link placement so the issue text does not change with it.
    link_rng = random.Random(11)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    issues: list[dict] = []
    for i in range(300):
        number = 1000 + i
        created = start + timedelta(hours=12 * i + rng.randint(0, 5))
        is_dup = i > 20 and rng.random() < 0.15
        if is_dup:
            canon = rng.choice(issues[:-1])
            topic = canon["_topic"]
            title = canon["title"].replace(" on ", " under ").replace("fails", "is failing")
            body = canon["body"].split("\n\n")[0] + "\n\nSame here. " + rng.choice(FILLER)
            kind = canon["_kind"]
        else:
            canon = None
            topic = rng.choice(list(TOPICS))
            title = _fill(rng.choice(TOPICS[topic]), rng)
            body = f"<!-- Please fill in the template -->\n{title}.\n\n" + " ".join(
                rng.sample(FILLER, 2)
            )
            kind = rng.choices(list(KIND), weights=list(KIND.values()))[0]

        labels = [topic, kind]
        dup_links = []
        state_reason = "COMPLETED" if rng.random() < 0.7 else None
        if canon is not None:
            labels.append("duplicate")
            state_reason = "DUPLICATE"
            # Mirror real GitHub data: the timeline link may sit only on the original issue,
            # only on the duplicate, or come from a "Duplicate of #N" comment.
            where = link_rng.choice(["canonical", "duplicate", "comment"])
            if where == "canonical":
                canon["dup_links"].append(
                    {"other": number, "source": "timeline", "directed": False}
                )
            elif where == "duplicate":
                dup_links.append(
                    {"other": canon["number"], "source": "timeline", "directed": False}
                )
            else:
                dup_links.append({"other": canon["number"], "source": "comment", "directed": True})
        issues.append(
            {
                "repo": REPO,
                "number": number,
                "title": title,
                "body": body,
                "url": f"https://github.com/{REPO}/issues/{number}",
                "created_at": created.isoformat().replace("+00:00", "Z"),
                "closed_at": None,
                "state": "CLOSED" if state_reason else "OPEN",
                "state_reason": state_reason,
                "author": f"user{rng.randint(1, 80)}",
                "labels": labels,
                "dup_links": dup_links,
                "dup_mentions": [],
                # Applied by a maintainer two hours in, so none count as template labels.
                "label_events": [
                    {
                        "label": lbl,
                        "at": (created + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                        "actor": "maintainer",
                    }
                    for lbl in labels
                ],
                "_canon": canon["number"] if canon else None,
                "_topic": topic,
                "_kind": kind,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "acme__widgets.jsonl").open("w") as f:
        for rec in issues:
            f.write(json.dumps({k: v for k, v in rec.items() if not k.startswith("_")}) + "\n")
    label_defs = [{"name": t, "description": f"Issues about {t.split(': ')[1]}"} for t in TOPICS]
    label_defs += [
        {"name": "bug", "description": "Something is not working"},
        {"name": "enhancement", "description": "New feature or request"},
        {"name": "question", "description": "Usage question"},
        {"name": "duplicate", "description": "This issue already exists"},
    ]
    (OUT / "acme__widgets.labels.json").write_text(json.dumps(label_defs, indent=2))
    print(f"wrote {len(issues)} issues, {sum(r['_canon'] is not None for r in issues)} dups")


if __name__ == "__main__":
    main()
