"""Duplicate retrieval systems.

Every retriever indexes all issues of a repository once, then answers `search(i, k)` for the
issue at position `i` using only issues created strictly before it. That time mask is the
core leakage guard: a system can never "find" a canonical issue from the future.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from triage.config import ModelConfig
from triage.llm import CallRecord, LLMError, complete_json

log = logging.getLogger(__name__)

Hit = tuple[int, float]  # (issue number, score)


class IssueDoc(NamedTuple):
    """The parts of an issue the duplicate judge sees. Built from a DataFrame row in the
    benchmark and from Postgres in the live service, so both send identical prompts."""

    number: int
    title: str
    body_clean: str


def embedding_text(title: str, body_clean: str) -> str:
    """The exact text embedded for an issue, shared by the benchmark and the service."""
    return title + "\n\n" + body_clean[:1500]


class Retriever:
    name = "base"

    def index(self, issues: pd.DataFrame) -> None:
        issues = issues.sort_values("created_at", kind="stable").reset_index(drop=True)
        self.issues = issues
        self.numbers = issues["number"].to_numpy()
        self.created = issues["created_at"].dt.tz_convert(None).to_numpy()
        self.position = {int(n): i for i, n in enumerate(self.numbers)}
        self._build(issues)

    def search(self, number: int, k: int = 10) -> list[Hit]:
        i = self.position[number]
        # Issues created strictly before this one (ties excluded on purpose).
        n_before = int(np.searchsorted(self.created, self.created[i], side="left"))
        if n_before == 0:
            return []
        scores = self._scores(i, n_before)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(self.numbers[j]), float(scores[j])) for j in top]

    def similarity(self, a: int, b: int) -> float | None:
        """Embedding cosine between two issues, if this retriever has embeddings."""
        return None

    def _build(self, issues: pd.DataFrame) -> None:
        raise NotImplementedError

    def _scores(self, i: int, n_before: int) -> np.ndarray:
        """Scores for issues[0:n_before] against issue i."""
        raise NotImplementedError


class DifflibRetriever(Retriever):
    """Title string similarity against recent issues: what most hobby triage Actions do."""

    name = "difflib"

    def __init__(self, window: int = 2000) -> None:
        self.window = window

    def _build(self, issues: pd.DataFrame) -> None:
        self.titles = [t.lower() for t in issues["title"]]

    def _scores(self, i: int, n_before: int) -> np.ndarray:
        scores = np.full(n_before, -1.0)
        sm = SequenceMatcher(autojunk=False)
        sm.set_seq2(self.titles[i])
        for j in range(max(0, n_before - self.window), n_before):
            sm.set_seq1(self.titles[j])
            if sm.real_quick_ratio() > 0.3 and sm.quick_ratio() > 0.3:
                scores[j] = sm.ratio()
        return scores


class BM25Retriever(Retriever):
    """Sparse BM25 over title (counted twice) + body.

    Note: IDF is computed over the whole window, including future issues. That leaks corpus
    statistics (not labels or links); the effect is small and is documented in the README.
    """

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def _build(self, issues: pd.DataFrame) -> None:
        docs = (issues["title"] + " " + issues["title"] + " " + issues["body_clean"]).tolist()
        self.cv = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b\w[\w.\-]+\b")
        tf = self.cv.fit_transform(docs).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * lengths / lengths.mean())
        # BM25 term weight, computed only on non-zero entries of the sparse matrix.
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.weights = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape)
        # Query side: which terms appear in each issue (binary).
        self.query_terms = (tf > 0).astype(np.float32).tocsr()

    def _scores(self, i: int, n_before: int) -> np.ndarray:
        q = self.query_terms[i].T
        return np.asarray((self.weights[:n_before] @ q).todense()).ravel()


class EmbeddingRetriever(Retriever):
    """Dense cosine similarity with a sentence-transformers model (cached on disk)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir=Path(".cache/emb")):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.name = "embed-" + model_name.split("/")[-1]

    def _build(self, issues: pd.DataFrame) -> None:
        texts = [
            embedding_text(t, b) for t, b in zip(issues["title"], issues["body_clean"], strict=True)
        ]
        self.emb = encode_cached(self.model_name, texts, self.cache_dir)

    def _scores(self, i: int, n_before: int) -> np.ndarray:
        return self.emb[:n_before] @ self.emb[i]

    def similarity(self, a: int, b: int) -> float | None:
        return float(self.emb[self.position[a]] @ self.emb[self.position[b]])


class HybridRetriever(Retriever):
    """Reciprocal rank fusion of BM25 and dense retrieval."""

    def __init__(self, dense: EmbeddingRetriever, k_rrf: int = 60, depth: int = 100) -> None:
        self.bm25 = BM25Retriever()
        self.dense = dense
        self.k_rrf = k_rrf
        self.depth = depth
        self.name = "hybrid-bm25+" + dense.name.removeprefix("embed-")

    def _build(self, issues: pd.DataFrame) -> None:
        self.bm25.index(issues)
        self.dense.index(issues)

    def similarity(self, a: int, b: int) -> float | None:
        return self.dense.similarity(a, b)

    def _scores(self, i: int, n_before: int) -> np.ndarray:
        fused = np.zeros(n_before)
        for part in (self.bm25, self.dense):
            s = part._scores(i, n_before)
            top = np.argsort(-s, kind="stable")[: self.depth]
            fused[top] += 1.0 / (self.k_rrf + np.arange(1, len(top) + 1))
        return fused


class RerankedRetriever(Retriever):
    """First-stage retriever followed by a cross-encoder over the top candidates."""

    def __init__(
        self, first: Retriever, model_name: str = "BAAI/bge-reranker-base", depth: int = 20
    ) -> None:
        self.first = first
        self.model_name = model_name
        self.depth = depth
        self.name = f"{first.name}+rerank"

    def _build(self, issues: pd.DataFrame) -> None:
        from sentence_transformers import CrossEncoder

        self.first.index(issues)
        self.ce = CrossEncoder(self.model_name, max_length=512)
        self.texts = (issues["title"] + "\n\n" + issues["body_clean"].str[:1000]).tolist()

    def similarity(self, a: int, b: int) -> float | None:
        return self.first.similarity(a, b)

    def search(self, number: int, k: int = 10) -> list[Hit]:
        candidates = self.first.search(number, self.depth)
        if not candidates:
            return []
        q = self.texts[self.position[number]]
        pairs = [(q, self.texts[self.position[n]]) for n, _ in candidates]
        scores = self.ce.predict(pairs, show_progress_bar=False)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(candidates[j][0], float(scores[j])) for j in order]


def encode_cached(model_name: str, texts: list[str], cache_dir: Path) -> np.ndarray:
    key = hashlib.sha256(("\x00".join([model_name, *texts])).encode()).hexdigest()[:24]
    path = cache_dir / f"{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


DUP_SYSTEM_PROMPT = """You help maintainers of the {repo} GitHub repository find duplicate issues.
You get a new issue and a few earlier candidate issues found by search.
An issue is a duplicate only if it reports the same underlying bug or requests the same change,
not merely the same area or the same error message with a different cause.

Reply with only a JSON object:
{{"duplicate_of": <candidate issue number or null>, "confidence": <0.0 to 1.0>}}"""


class LLMDuplicateJudge:
    """Two-stage: a retriever proposes candidates, the LLM confirms one or rejects all."""

    def __init__(
        self, retriever: Retriever, model: ModelConfig, repo: str, n_candidates: int = 5
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.repo = repo
        self.n = n_candidates
        self.name = f"{retriever.name}+llm"
        self.records: list[CallRecord] = []
        self.failures = 0

    def judge_many(self, numbers: list[int]) -> list[tuple[list[Hit], int | None, float]]:
        with ThreadPoolExecutor(max_workers=self.model.concurrency) as pool:
            return list(pool.map(self.judge, numbers))

    def judge(self, number: int) -> tuple[list[Hit], int | None, float]:
        """Returns (re-ordered hits, chosen duplicate or None, confidence)."""
        hits = self.retriever.search(number, self.n)
        if not hits:
            return hits, None, 0.0
        issues = self.retriever.issues
        pos = self.retriever.position

        def doc(n: int) -> IssueDoc:
            row = issues.iloc[pos[n]]
            return IssueDoc(n, row["title"], row["body_clean"])

        system, user = build_judge_prompt(self.repo, doc(number), [doc(n) for n, _ in hits])
        try:
            reply, record = complete_json(self.model, system, user)
        except LLMError as exc:
            log.warning("duplicate call failed: %s", exc)
            self.failures += 1
            return hits, None, 0.0
        self.records.append(record)

        chosen, conf = parse_judge_reply(reply, [n for n, _ in hits])
        if chosen is not None:
            hits = [h for h in hits if h[0] == chosen] + [h for h in hits if h[0] != chosen]
        return hits, chosen, conf


def build_judge_prompt(repo: str, new: IssueDoc, candidates: list[IssueDoc]) -> tuple[str, str]:
    """(system, user) messages for the duplicate judge. The live service calls this too, so
    any change here changes both, and the LLM cache shows whether the bytes moved."""
    parts = [f"New issue #{new.number}:\n{_fmt(new)}\n", "Candidates:"]
    for c in candidates:
        parts.append(f"\n#{c.number}:\n{_fmt(c, 800)}")
    return DUP_SYSTEM_PROMPT.format(repo=repo), "\n".join(parts)


def parse_judge_reply(reply: dict[str, Any], candidates: list[int]) -> tuple[int | None, float]:
    """(chosen issue or None, confidence). A pick outside the shown candidates is dropped,
    and confidence is 0 when nothing is picked."""
    chosen = reply.get("duplicate_of")
    try:
        chosen = int(chosen) if chosen is not None else None
    except (TypeError, ValueError):
        chosen = None
    if chosen not in set(candidates):
        chosen = None  # the model named an issue it was not shown
    try:
        conf = float(reply.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return chosen, conf if chosen is not None else 0.0


def _fmt(doc: IssueDoc, body_chars: int = 1500) -> str:
    body = doc.body_clean
    if len(body) > body_chars:
        body = body[:body_chars] + " ..."
    return f"Title: {doc.title}\n{body}"
