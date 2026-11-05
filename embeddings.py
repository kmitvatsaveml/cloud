"""Embedding backends.

Primary: `BAAI/bge-small-en-v1.5` run locally through fastembed (ONNX, no torch).
Fallback: a deterministic hashed bag-of-words embedder, used automatically when
the model cannot be downloaded (e.g. an offline sandbox). The fallback keeps the
whole pipeline runnable; retrieval quality is lower.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from config import get_settings

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 and the fallback embedder."""
    return TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """bge-small via fastembed."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


class HashEmbedder:
    """Deterministic hashed unigram+bigram embedder (offline fallback)."""

    name = "hash-bow-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            toks = tokenize(text)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            vec = [0.0] * self.dim
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return bge-small if loadable, else the hashing fallback."""
    import os

    if os.getenv("EMBED_BACKEND", "auto") == "hash" or os.getenv("HACKAGENT_OFFLINE") == "1":
        return HashEmbedder()
    try:
        emb = FastEmbedder(get_settings().embed_model)
        emb.embed(["warmup"])
        return emb
    except Exception as exc:
        print(f"[embeddings] {get_settings().embed_model} unavailable ({type(exc).__name__}); using HashEmbedder fallback")
        return HashEmbedder()
