"""Part A.2: map-reduce transcripts into `personas/{slug}/persona.md`.

Map: per video, extract tagged bullets for each section.
Reduce: merge + deduplicate into one persona document, every bullet tagged
with its source video (and timestamp).
Uses Claude when available; otherwise an extractive keyword classifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import get_settings
from ingest import chunk_transcript
from llm import complete, extract_json

SECTIONS: dict[str, list[str]] = {
    "Core principles": ["understand", "from scratch", "philosophy", "intuition", "simplicity", "first principles", "become one"],
    "Building philosophy": ["build", "start with", "simplest", "baseline", "end to end", "working", "library", "organization"],
    "Debugging playbook": ["bug", "check", "print", "shape", "gradient", "test", "verify", "suspicious", "sanity", "overfit"],
    "Scoping & prioritizing under time pressure": ["scope", "time pressure", "first", "then", "demo", "pretrained", "one component", "prematurely"],
    "Things they warn against": ["don't", "do not", "never", "warn", "careful", "bite", "classic bug", "wrong"],
    "Do / Don't rules": ["do:", "don't:", "always", "never"],
}


def _tag(c: dict[str, Any]) -> str:
    return f"[{c['title']} @ {c['timestamp']}]({c['url']})" if c.get("url") else f"[{c['title']} @ {c['timestamp']}]"


def map_video(doc: dict[str, Any]) -> dict[str, list[str]]:
    """Extract section bullets from one transcript."""
    chunks = chunk_transcript(doc)
    numbered = "\n".join(f"[{i}] ({c['timestamp']}) {c['text']}" for i, c in enumerate(chunks))
    text, _ = complete(
        "You distill an expert's lecture transcript into reusable engineering heuristics.",
        f"Transcript of '{doc.get('title')}' split into numbered chunks:\n{numbered[:60000]}\n\n"
        f"Return JSON: {{section: [{{\"bullet\": str, \"chunk\": int}}]}} with sections exactly {list(SECTIONS)}. "
        "Bullets are short, imperative, in the expert's spirit. 2-6 bullets per section. Only use what the transcript supports.",
    )
    data = extract_json(text)
    out: dict[str, list[str]] = {s: [] for s in SECTIONS}
    if isinstance(data, dict):
        for sec, items in data.items():
            if sec in out and isinstance(items, list):
                for it in items:
                    try:
                        c = chunks[int(it.get("chunk", 0))]
                        out[sec].append(f"{it['bullet']} - {_tag(c)}")
                    except Exception:
                        continue
        if any(out.values()):
            return out
    # offline extractive fallback: classify sentences by keyword
    for seg in doc["segments"]:
        sent = seg["text"].strip()
        low = sent.lower()
        c = {"title": doc.get("title", ""), "timestamp": f"{int(seg['start'])//60:02d}:{int(seg['start'])%60:02d}",
             "url": f"{doc['url']}&t={int(seg['start'])}s" if doc.get("url") else ""}
        best = max(SECTIONS, key=lambda s: sum(k in low for k in SECTIONS[s]))
        if sum(k in low for k in SECTIONS[best]) > 0:
            out[best].append(f"{sent} - {_tag(c)}")
    return out


def reduce_persona(name: str, maps: list[dict[str, list[str]]]) -> str:
    """Merge per-video bullets into persona markdown."""
    merged = {s: [b for m in maps for b in m.get(s, [])] for s in SECTIONS}
    text, _ = complete(
        "You merge distilled heuristics into a concise persona document. Keep every source tag exactly as given.",
        "Merge and deduplicate these bullets per section (keep the source link tag at the end of each bullet; "
        "if two bullets merge, keep both tags). Return JSON {section: [bullet]}.\n" + json.dumps(merged)[:80000],
    )
    data = extract_json(text)
    if isinstance(data, dict) and any(data.values()):
        merged = {s: [str(b) for b in data.get(s, merged[s])] for s in SECTIONS}
    lines = [f"# Persona: {name}", "",
             "_Distilled from the expert's video transcripts. Every bullet links to its source video and timestamp._", ""]
    for sec, bullets in merged.items():
        lines += [f"## {sec}", ""] + [f"- {b}" for b in dict.fromkeys(bullets)] + [""]
    return "\n".join(lines)


def build_persona(slug: str, display_name: str | None = None) -> Path:
    """Generate persona.md from the transcripts saved by ingest."""
    s = get_settings()
    tdir = s.personas_dir / slug / "transcripts"
    docs = [json.loads(p.read_text()) for p in sorted(tdir.glob("*.json"))]
    if not docs:
        raise RuntimeError(f"No transcripts for persona '{slug}'. Run ingest first.")
    md = reduce_persona(display_name or slug, [map_video(d) for d in docs])
    if any("SYNTHETIC SAMPLE" in d.get("note", "") for d in docs):
        md = md.replace("\n\n", "\n\n> Built from SYNTHETIC sample transcripts (offline test data), not the real videos.\n\n", 1)
    path = s.personas_dir / slug / "persona.md"
    path.write_text(md)
    print(f"[persona] wrote {path} ({len(md)} chars)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default=get_settings().default_persona)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    build_persona(a.persona, a.name)
