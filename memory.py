"""Part A.3: experience memory - `consult(question, k=6)`.

Pipeline: multi-query rewrite -> hybrid retrieval (Chroma vector + BM25) fused
with Reciprocal Rank Fusion -> FlashRank rerank -> top-k chunks with citations.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

from config import get_settings
from embeddings import HashEmbedder, get_embedder, tokenize
from llm import complete, extract_json

RRF_K = 60
_STOP = set("a an the to of and or in on for with how what would i you we it is are be do does should this that".split())


def rewrite_queries(question: str, n: int = 3) -> list[str]:
    """Multi-query rewrite. Uses the LLM if available, else keyword heuristics."""
    text, _ = complete(
        "You rewrite search queries for a transcript search engine. Reply with a JSON list of strings only.",
        f"Write {n} diverse rephrasings of this question, using vocabulary a teacher would say out loud "
        f"in a lecture:\n{question}",
    )
    rewrites = extract_json(text)
    if isinstance(rewrites, list) and rewrites:
        return [question] + [str(r) for r in rewrites[:n]]
    kw = [t for t in tokenize(question) if t not in _STOP]
    variants = [question, " ".join(kw)]
    if any(w in kw for w in ("bug", "debug", "error", "fail", "failing", "broken", "fix")):
        variants.append("common bug check print values hypothesis test tiny input")
    if any(w in kw for w in ("scope", "time", "hackathon", "mvp", "prioritize", "deadline", "plan")):
        variants.append("start simple baseline end to end scope ruthlessly works and demos well")
    return variants[: n + 1]


@lru_cache(maxsize=8)
def _load(persona: str) -> tuple[Any, dict[str, Any], Any]:
    """Load (chroma collection, bm25 data, BM25Okapi) for a persona."""
    import chromadb
    from rank_bm25 import BM25Okapi

    s = get_settings()
    col = chromadb.PersistentClient(path=str(s.chroma_dir)).get_collection(persona)
    with open(s.bm25_dir / f"{persona}.pkl", "rb") as fh:
        data = pickle.load(fh)
    return col, data, BM25Okapi(data["tokens"])


@lru_cache(maxsize=1)
def _reranker() -> Any | None:
    import os

    if os.getenv("HACKAGENT_OFFLINE") == "1":
        return None
    try:
        from flashrank import Ranker

        return Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(get_settings().data_dir / "flashrank"))
    except Exception as exc:
        print(f"[memory] FlashRank unavailable ({type(exc).__name__}); using lexical rerank fallback")
        return None


def rerank(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rerank candidates with FlashRank (or token-overlap fallback)."""
    ranker = _reranker()
    if ranker is not None:
        from flashrank import RerankRequest

        res = ranker.rerank(RerankRequest(query=question, passages=[{"id": c["id"], "text": c["text"]} for c in chunks]))
        scores = {r["id"]: float(r["score"]) for r in res}
    else:
        q = {t for t in tokenize(question) if t not in _STOP}
        scores = {c["id"]: len(q & set(tokenize(c["text"]))) / (len(q) or 1) + c["rrf"] for c in chunks}
    for c in chunks:
        c["score"] = round(scores.get(c["id"], 0.0), 4)
    return sorted(chunks, key=lambda c: c["score"], reverse=True)


def consult(question: str, k: int = 6, persona: str | None = None, candidates: int = 20) -> list[dict[str, Any]]:
    """Ask the expert memory. Returns chunks with `text`, `title`, `timestamp`, `url`, `citation`."""
    persona = persona or get_settings().default_persona
    try:
        col, data, bm25 = _load(persona)
    except Exception as exc:
        print(f"[memory] persona '{persona}' not indexed: {exc}")
        return []
    by_id = {c["id"]: c for c in data["chunks"]}
    embedder = get_embedder()
    if (col.metadata or {}).get("embedder") == HashEmbedder.name and embedder.name != HashEmbedder.name:
        embedder = HashEmbedder()  # query with the same embedder the index was built with
    fused: dict[str, float] = {}
    for q in rewrite_queries(question):
        vec = col.query(query_embeddings=embedder.embed([q]), n_results=min(candidates, col.count()))
        ranked_lists = [vec["ids"][0]]
        scores = bm25.get_scores(tokenize(q))
        ranked_lists.append([data["chunks"][i]["id"] for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:candidates]])
        for ranked in ranked_lists:
            for rank, cid in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    top = sorted(fused, key=fused.get, reverse=True)[:candidates]
    cands = [{**by_id[c], "rrf": fused[c]} for c in top]
    out = rerank(question, cands)[:k]
    for c in out:
        c["citation"] = f"[{c['title']} @ {c['timestamp']}]({c['url']})" if c["url"] else f"[{c['title']} @ {c['timestamp']}]"
    return out


def format_context(chunks: list[dict[str, Any]], max_chars: int = 3500) -> str:
    """Render chunks for inclusion in an LLM prompt."""
    out, n = [], 0
    for c in chunks:
        line = f"- {c['text']} (source: {c['citation']})"
        if n + len(line) > max_chars:
            break
        out.append(line)
        n += len(line)
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    for c in consult(" ".join(sys.argv[1:]) or "how do I debug a model that is not learning?"):
        print(f"{c['score']:.3f} {c['citation']}\n    {c['text'][:160]}")
