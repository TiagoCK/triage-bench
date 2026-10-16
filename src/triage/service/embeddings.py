"""Text embeddings for the live service, without torch.

The default model is BAAI's own full-precision ONNX export of bge-small-en-v1.5, loaded
through fastembed. On 300 vscode issues it matched the benchmark's sentence-transformers
embeddings exactly (cosine 1.00000, identical top-5 neighbours). fastembed's built-in
bge-small is a separate quantized port, kept as an option under its own name.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

DIM = 384
FP32_MODEL = "triage/bge-small-en-v1.5-fp32"
QUANTIZED_MODEL = "BAAI/bge-small-en-v1.5"  # fastembed's built-in (quantized) port


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def to_pgvector(vec: np.ndarray) -> str:
    """A pgvector literal ('[0.1,0.2,...]'), passed as text and cast with ::vector."""
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


_registered = False


def _register_fp32() -> None:
    global _registered
    if _registered:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    if FP32_MODEL not in {m["model"] for m in TextEmbedding.list_supported_models()}:
        TextEmbedding.add_custom_model(
            model=FP32_MODEL,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf="BAAI/bge-small-en-v1.5"),
            dim=DIM,
            model_file="onnx/model.onnx",
        )
    _registered = True


class FastEmbedder:
    def __init__(self, model: str = FP32_MODEL, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding

        if model == FP32_MODEL:
            _register_fp32()
        self.name = model
        self.dim = DIM
        self._model = TextEmbedding(model, cache_dir=str(cache_dir) if cache_dir else None)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embeds in length-sorted batches: each batch pads to its longest text, so mixing a
        one-line issue with a 1,500-character one wastes work. Output is in input order and
        identical to unsorted batching (checked: max difference 1e-7)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.array(
            list(self._model.embed([texts[i] for i in order], batch_size=32)), dtype=np.float32
        )
        result = np.empty_like(out)
        result[order] = out
        return result
