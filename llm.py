"""Thin LLM wrapper.

Uses Claude through `langchain-anthropic` when `ANTHROPIC_API_KEY` is set.
Otherwise every call returns `None`, and callers fall back to deterministic
heuristics or templates (the "offline mode"). Token usage is tracked so the
agent can enforce a token budget.
"""
from __future__ import annotations

import json
import re
from typing import Any

from config import get_settings

_LLM_CACHE: dict[str, Any] = {}


def get_chat_model() -> Any | None:
    """Return a cached `ChatAnthropic` instance, or None in offline mode."""
    s = get_settings()
    if s.offline:
        return None
    if s.model not in _LLM_CACHE:
        from langchain_anthropic import ChatAnthropic

        _LLM_CACHE[s.model] = ChatAnthropic(model=s.model, max_tokens=8000, temperature=0.2, timeout=180)
    return _LLM_CACHE[s.model]


def image_blocks(images: list[Any] | None) -> list[dict[str, Any]]:
    """Turn image paths / bytes / {media_type,data} dicts into Anthropic image content blocks."""
    import base64
    import mimetypes
    from pathlib import Path

    blocks = []
    for img in images or []:
        if isinstance(img, dict) and "data" in img:
            mt, data = img.get("media_type", "image/png"), img["data"]
        elif isinstance(img, (bytes, bytearray)):
            mt, data = "image/png", base64.b64encode(img).decode()
        else:
            p = Path(str(img))
            if not p.exists():
                continue
            mt, data = mimetypes.guess_type(p.name)[0] or "image/png", base64.b64encode(p.read_bytes()).decode()
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
    return blocks


def complete(system: str, prompt: str, images: list[Any] | None = None) -> tuple[str | None, int]:
    """Run a single (optionally multimodal) chat completion.

    `images` may be file paths, raw bytes or base64 dicts; they are sent to
    Claude's vision input alongside the prompt.
    Returns (text, tokens_used). text is None in offline mode or on error.
    """
    model = get_chat_model()
    if model is None:
        return None, 0
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        content: Any = [*image_blocks(images), {"type": "text", "text": prompt}] if images else prompt
        msg = model.invoke([SystemMessage(content=system), HumanMessage(content=content)])
    except Exception as exc:  # network / rate-limit: let caller fall back
        print(f"[llm] error: {exc}")
        return None, 0
    usage = getattr(msg, "usage_metadata", None) or {}
    tokens = int(usage.get("total_tokens", 0))
    content = msg.content if isinstance(msg.content, str) else "".join(
        part.get("text", "") for part in msg.content if isinstance(part, dict)
    )
    return content, tokens


def extract_json(text: str | None) -> Any | None:
    """Extract the first JSON object/array from an LLM response."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    for cand in candidates:
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = cand.find(opener), cand.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(cand[start : end + 1])
                except json.JSONDecodeError:
                    continue
    return None


FILE_BLOCK_RE = re.compile(r'<file\s+path="([^"]+)">\n?(.*?)</file>', re.S)


def extract_files(text: str | None) -> dict[str, str]:
    """Parse `<file path="...">content</file>` blocks into a dict."""
    if not text:
        return {}
    return {path.strip(): body for path, body in FILE_BLOCK_RE.findall(text)}
