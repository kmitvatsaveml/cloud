"""Central configuration loaded from `.env`.

All tunables (budgets, model names, paths) live here so every module reads the
same values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
# Only trace to LangSmith when a key is configured (avoids noisy connection errors).
if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    samples_dir: Path = ROOT / "data" / "samples"
    transcripts_dir: Path = ROOT / "data" / "transcripts"
    chroma_dir: Path = ROOT / "data" / "chroma"
    bm25_dir: Path = ROOT / "data" / "bm25"
    personas_dir: Path = ROOT / "personas"
    runs_dir: Path = ROOT / "runs"
    checkpoint_db: Path = ROOT / "checkpoints.sqlite"

    anthropic_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    force_offline: bool = field(default_factory=lambda: os.getenv("HACKAGENT_OFFLINE", "0") == "1")
    tavily_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    max_iterations: int = field(default_factory=lambda: _int("MAX_ITERATIONS", 25))
    max_debug_loops: int = field(default_factory=lambda: _int("MAX_DEBUG_LOOPS", 4))
    max_wall_minutes: int = field(default_factory=lambda: _int("MAX_WALL_MINUTES", 20))
    max_tokens_budget: int = field(default_factory=lambda: _int("MAX_TOKENS_BUDGET", 400_000))
    cmd_timeout: int = field(default_factory=lambda: _int("CMD_TIMEOUT_SECONDS", 600))

    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    default_persona: str = field(default_factory=lambda: os.getenv("DEFAULT_PERSONA", "karpathy-zero-to-hero"))

    @property
    def offline(self) -> bool:
        """True when no LLM is available (no key, or forced offline)."""
        return self.force_offline or not self.anthropic_key


def get_settings() -> Settings:
    """Return fresh settings (re-reads env so tests can monkeypatch)."""
    return Settings()
