"""Per-course study notes: one Markdown file per course (playlist).

`courses/{slug}/COURSE_NOTES.md` has a section per video: a summary, key
takeaways with timestamp links, and a glossary. Uses Claude when available;
otherwise it is extractive (picks the most information-dense sentences).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from config import get_settings
from embeddings import tokenize
from llm import complete, extract_json

STOP = set("the a an to of and or in on for with is are be it this that i you we so if do not just from have what your will then when they them there their about would".split())


def _ts(sec: float) -> str:
    return f"{int(sec)//60:02d}:{int(sec)%60:02d}"


def _link(doc: dict[str, Any], sec: float) -> str:
    return f"[{_ts(sec)}]({doc['url']}&t={int(sec)}s)" if doc.get("url") else f"[{_ts(sec)}]"


def video_notes(doc: dict[str, Any]) -> str:
    """Notes for one video (LLM or extractive)."""
    segs = doc["segments"]
    text = "\n".join(f"({_ts(s['start'])}) {s['text']}" for s in segs)
    out, _ = complete("You write excellent, concise study notes from lecture transcripts.",
                      f"Lecture: {doc.get('title')}\nTranscript:\n{text[:60000]}\n\nReturn JSON {{\"summary\": str, "
                      "\"takeaways\": [{\"point\": str, \"seconds\": number}], \"glossary\": [{\"term\": str, \"definition\": str}]}")
    data = extract_json(out)
    if not isinstance(data, dict):
        freq = Counter(t for s in segs for t in tokenize(s["text"]) if t not in STOP and len(t) > 3)
        scored = sorted(segs, key=lambda s: -sum(freq[t] for t in set(tokenize(s["text"]))) / (len(s["text"]) ** 0.5 + 1))
        top = sorted(scored[: min(8, len(segs))], key=lambda s: s["start"])
        data = {"summary": " ".join(s["text"] for s in segs[:2]),
                "takeaways": [{"point": s["text"], "seconds": s["start"]} for s in top],
                "glossary": [{"term": w, "definition": f"mentioned {n}x"} for w, n in freq.most_common(6)]}
    lines = [f"## {doc.get('title', doc['video_id'])}", ""]
    if doc.get("url"):
        lines += [f"Video: {doc['url']}", ""]
    lines += ["**Summary.** " + str(data.get("summary", "")), "", "**Key takeaways**", ""]
    lines += [f"- {t['point']} {_link(doc, float(t.get('seconds', 0)))}" for t in data.get("takeaways", [])]
    if data.get("glossary"):
        lines += ["", "**Glossary**", ""] + [f"- **{g['term']}**: {g['definition']}" for g in data["glossary"]]
    return "\n".join(lines) + "\n"


def build_course_notes(slug: str, title: str | None = None) -> Path:
    """Write courses/{slug}/COURSE_NOTES.md from the transcripts ingested for that course."""
    s = get_settings()
    tdir = s.personas_dir / slug / "transcripts"
    docs = [json.loads(p.read_text()) for p in sorted(tdir.glob("*.json"))]
    if not docs:
        raise RuntimeError(f"No transcripts for course '{slug}'. Ingest it first.")
    head = [f"# Course notes: {title or slug}", "", f"{len(docs)} lectures. Timestamps link to the video.", ""]
    if any("SYNTHETIC" in d.get("note", "") for d in docs):
        head += ["> Built from SYNTHETIC sample transcripts (offline test data).", ""]
    toc = ["## Contents", ""] + [f"{i+1}. {d.get('title', d['video_id'])}" for i, d in enumerate(docs)] + [""]
    body = "\n".join(video_notes(d) for d in docs)
    out = s.root / "courses" / slug / "COURSE_NOTES.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + toc) + "\n" + body)
    print(f"[notes] wrote {out}")
    return out


def list_courses() -> list[str]:
    """Slugs of courses that have transcripts."""
    d = get_settings().personas_dir
    return sorted(p.name for p in d.iterdir() if (p / "transcripts").exists()) if d.exists() else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", default=get_settings().default_persona)
    ap.add_argument("--title")
    ap.add_argument("--all", action="store_true", help="build notes for every ingested course")
    a = ap.parse_args()
    for c in (list_courses() if a.all else [a.course]):
        build_course_notes(c, a.title)
