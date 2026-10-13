"""Label prediction systems.

Each system is fit on train issues that carry at least one vocabulary label and predicts a
list of labels per test issue. LLM systems also return per-call records for cost/latency.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from triage.config import ModelConfig
from triage.llm import CallRecord, LLMError, complete_json

log = logging.getLogger(__name__)


class TfidfLabeler:
    """TF-IDF + one-vs-rest logistic regression. The classic, cheap baseline."""

    name = "tfidf-lr"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.records: list[CallRecord] = []

    def fit(self, train: pd.DataFrame, vocab: list[dict]) -> None:
        self.vocab = [v["name"] for v in vocab]
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True
        )
        x = self.vec.fit_transform(train["text"])
        self.mlb = MultiLabelBinarizer(classes=self.vocab)
        y = self.mlb.fit_transform(train["target_labels"])
        keep = y.sum(axis=0) > 0  # a label with no train positives cannot be learned
        self.classes = np.array(self.vocab)[keep]
        self.clf = OneVsRestClassifier(
            LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)
        )
        self.clf.fit(x, y[:, keep])

    def predict(self, test: pd.DataFrame) -> list[list[str]]:
        proba = self.clf.predict_proba(self.vec.transform(test["text"]))
        out = []
        for row in proba:
            picked = list(self.classes[row >= self.threshold])
            # Every evaluated issue has at least one gold label, so never predict nothing.
            out.append(picked or [self.classes[int(row.argmax())]])
        return out


LABEL_SYSTEM_PROMPT = """You triage GitHub issues for the {repo} repository.
Pick the labels a maintainer would apply to the issue, using only labels from this list:

{label_list}

Rules:
- Choose between 1 and 3 labels. Prefer precision: only pick a label you are confident about.
- Use the exact label names from the list.
- Reply with only a JSON object: {{"labels": ["<label>", ...]}}"""

EXAMPLES_HEADER = "Here are past issues from this repository with their labels:\n\n"


class LLMLabeler:
    """Zero-shot (label names + descriptions) or few-shot (k nearest train issues)."""

    def __init__(self, model: ModelConfig, repo: str, few_shot_k: int = 0) -> None:
        self.model = model
        self.repo = repo
        self.k = few_shot_k
        self.name = f"llm-{'fewshot' if few_shot_k else 'zeroshot'}"
        self.records: list[CallRecord] = []
        self.failures = 0

    def fit(self, train: pd.DataFrame, vocab: list[dict]) -> None:
        self.vocab = [v["name"] for v in vocab]
        lines = []
        for v in vocab:
            desc = f": {v['description']}" if v.get("description") else ""
            lines.append(f"- {v['name']}{desc}")
        self.system = LABEL_SYSTEM_PROMPT.format(repo=self.repo, label_list="\n".join(lines))
        if self.k:
            # Nearest neighbours by TF-IDF cosine; cheap and needs no embedding model.
            self.train = train.reset_index(drop=True)
            self.vec = TfidfVectorizer(sublinear_tf=True, min_df=2)
            self.train_x = self.vec.fit_transform(self.train["text"])

    def predict(self, test: pd.DataFrame) -> list[list[str]]:
        texts = list(test["text"])
        examples = self._examples(texts) if self.k else [""] * len(texts)
        with ThreadPoolExecutor(max_workers=self.model.concurrency) as pool:
            return list(pool.map(self._predict_one, texts, examples))

    def _predict_one(self, text: str, examples: str) -> list[str]:
        user = f"{examples}New issue:\n\n{_truncate(text, 3000)}"
        try:
            reply, record = complete_json(self.model, self.system, user)
        except LLMError as exc:
            log.warning("label call failed: %s", exc)
            self.failures += 1
            return []
        self.records.append(record)
        labels = reply.get("labels", [])
        if not isinstance(labels, list):
            return []
        return [x for x in labels if x in self.vocab]

    def _examples(self, texts: list[str]) -> list[str]:
        sims = (self.vec.transform(texts) @ self.train_x.T).toarray()
        out = []
        for row in sims:
            top = np.argsort(-row)[: self.k]
            parts = []
            for i in top:
                ex = self.train.iloc[i]
                parts.append(
                    f"Issue:\n{_truncate(ex['text'], 600)}\n"
                    f"Labels: {json.dumps(list(ex['target_labels']))}\n"
                )
            out.append(EXAMPLES_HEADER + "\n".join(parts) + "\n")
        return out


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " ..."
