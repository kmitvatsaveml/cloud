"""Part A.1: playlist -> transcripts -> chunks -> Chroma + BM25.

Usage:
    python ingest.py --playlist <URL> --persona karpathy-zero-to-hero
    python ingest.py --samples --persona karpathy-zero-to-hero   # offline test data
    python ingest.py --files a.json b.txt --persona my-expert      # uploaded transcripts
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

from config import get_settings
from embeddings import get_embedder, tokenize

CHUNK_MIN_S, CHUNK_MAX_S = 45.0, 60.0


# ---------------------------------------------------------------- fetching
def list_playlist(url: str) -> list[dict[str, str]]:
    """Return [{video_id, title, url}] using yt-dlp in metadata-only mode."""
    import yt_dlp

    opts = {"quiet": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") or [info]
    return [
        {"video_id": e["id"], "title": e.get("title") or e["id"], "url": f"https://www.youtube.com/watch?v={e['id']}",
         "playlist": info.get("title", "")}
        for e in entries if e and e.get("id")
    ]


def fetch_transcript(video: dict[str, str], cache_dir: Path) -> dict[str, Any] | None:
    """Fetch one transcript (cached as JSON). Returns None on failure."""
    cache = cache_dir / f"{video['video_id']}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        try:  # youtube-transcript-api >= 1.0
            fetched = YouTubeTranscriptApi().fetch(video["video_id"], languages=["en", "en-US", "en-GB"])
            segs = [{"start": s.start, "duration": s.duration, "text": s.text} for s in fetched]
        except AttributeError:  # older API
            raw = YouTubeTranscriptApi.get_transcript(video["video_id"], languages=["en", "en-US", "en-GB"])
            segs = [{"start": s["start"], "duration": s["duration"], "text": s["text"]} for s in raw]
    except Exception as exc:
        print(f"[ingest] SKIP {video['video_id']} ({video.get('title')}): {type(exc).__name__}: {exc}"[:300])
        return None
    doc = {**video, "segments": segs}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(doc))
    return doc


def load_uploaded(path: Path) -> dict[str, Any]:
    """Load an uploaded transcript: our JSON format, or plain text / .srt / .vtt."""
    if path.suffix == ".json":
        return json.loads(path.read_text())
    text = path.read_text(errors="ignore")
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and "-->" not in l and not l.strip().isdigit() and l.strip() != "WEBVTT"]
    segs = [{"start": i * 8.0, "duration": 8.0, "text": l} for i, l in enumerate(lines)]
    vid = re.sub(r"[^a-zA-Z0-9_-]", "_", path.stem)
    return {"video_id": vid, "title": path.stem, "url": "", "segments": segs}


# ---------------------------------------------------------------- chunking
def ts_url(url: str, seconds: float) -> str:
    """YouTube URL that jumps to a timestamp."""
    return f"{url}&t={int(seconds)}s" if url else ""


def chunk_transcript(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Group segments into ~45-60 s chunks with citation metadata."""
    chunks, buf, start = [], [], None
    for seg in doc["segments"]:
        if start is None:
            start = seg["start"]
        buf.append(seg["text"].replace("\n", " "))
        span = seg["start"] + seg.get("duration", 0) - start
        if span >= CHUNK_MIN_S:  # segments are short, so chunks land in ~45-60 s
            chunks.append((start, span, " ".join(buf)))
            buf, start = [], None
    if buf:
        chunks.append((start, 0.0, " ".join(buf)))
    return [
        {
            "id": f"{doc['video_id']}-{i:04d}",
            "text": text,
            "video_id": doc["video_id"],
            "title": doc.get("title", ""),
            "start": float(st),
            "timestamp": f"{int(st)//60:02d}:{int(st)%60:02d}",
            "url": ts_url(doc.get("url", ""), st),
        }
        for i, (st, _span, text) in enumerate(chunks)
    ]


# ---------------------------------------------------------------- indexing
def build_index(persona: str, docs: list[dict[str, Any]]) -> int:
    """(Re)build the Chroma collection and BM25 pickle for a persona."""
    import chromadb

    s = get_settings()
    chunks = [c for d in docs for c in chunk_transcript(d)]
    if not chunks:
        raise RuntimeError("No chunks to index - all transcripts failed?")
    embedder = get_embedder()
    client = chromadb.PersistentClient(path=str(s.chroma_dir))
    try:
        client.delete_collection(persona)
    except Exception:
        pass
    col = client.create_collection(persona, metadata={"hnsw:space": "cosine", "embedder": embedder.name})
    for i in range(0, len(chunks), 64):
        batch = chunks[i : i + 64]
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embedder.embed([c["text"] for c in batch]),
            metadatas=[{k: v for k, v in c.items() if k not in ("id", "text")} for c in batch],
        )
    s.bm25_dir.mkdir(parents=True, exist_ok=True)
    with open(s.bm25_dir / f"{persona}.pkl", "wb") as fh:
        pickle.dump({"chunks": chunks, "tokens": [tokenize(c["text"]) for c in chunks]}, fh)
    # keep the raw transcripts next to the persona for persona.py
    pdir = s.personas_dir / persona / "transcripts"
    pdir.mkdir(parents=True, exist_ok=True)
    for d in docs:
        (pdir / f"{d['video_id']}.json").write_text(json.dumps(d))
    print(f"[ingest] persona={persona} videos={len(docs)} chunks={len(chunks)} embedder={embedder.name}")
    return len(chunks)


def ingest(persona: str, playlist: str | None = None, samples: bool = False, files: list[str] | None = None) -> int:
    """Entry point used by CLI and UI. Returns number of chunks indexed."""
    s = get_settings()
    docs: list[dict[str, Any]] = []
    if samples:
        docs += [json.loads(p.read_text()) for p in sorted(s.samples_dir.glob("*.json"))]
    for f in files or []:
        docs.append(load_uploaded(Path(f)))
    if playlist:
        failed = []
        for v in list_playlist(playlist):
            d = fetch_transcript(v, s.transcripts_dir)
            (docs.append(d) if d else failed.append(v["video_id"]))
        if failed:
            print(f"[ingest] skipped {len(failed)} videos: {failed}")
    return build_index(persona, docs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--playlist")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--persona", default=get_settings().default_persona)
    a = ap.parse_args()
    if not (a.playlist or a.samples or a.files):
        ap.error("give --playlist, --samples or --files")
    ingest(a.persona, a.playlist, a.samples, a.files)
