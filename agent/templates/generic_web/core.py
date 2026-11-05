"""Core logic for the generated MVP: store submissions and analyze them with keyword scoring."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

STOP = set("a an the to of and or in on for with is are be it this that i you we".split())


@dataclass
class Store:
    """In-memory store; swap for SQLite when persistence matters."""
    items: list[dict] = field(default_factory=list)

    def add(self, text: str) -> dict:
        item = {"id": len(self.items) + 1, "text": text, "analysis": analyze(text)}
        self.items.append(item)
        return item


def analyze(text: str) -> dict:
    """Return word count, top keywords and a short summary."""
    words = [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in STOP]
    top = [w for w, _ in Counter(words).most_common(5)]
    first = re.split(r"(?<=[.!?])\s+", text.strip())[0] if text.strip() else ""
    return {"words": len(words), "keywords": top, "summary": first[:200]}
