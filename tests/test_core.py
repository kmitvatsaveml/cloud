"""Core tests: chunking, memory, sandbox safety, debug heuristics, budgets, graph."""
import json
import time
from pathlib import Path

import pytest

from agent import sandbox as sb
from agent.graph import budget_exceeded
from agent.nodes import debug, parse_constraints
from ingest import chunk_transcript, ingest
from memory import consult

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_chunks_are_45_to_70s():
    doc = json.loads(next(SAMPLES.glob("*.json")).read_text())
    chunks = chunk_transcript(doc)
    assert chunks and all("&t=" in c["url"] for c in chunks)
    starts = [c["start"] for c in chunks]
    assert all(45 <= b - a <= 75 for a, b in zip(starts, starts[1:]))


def test_consult_returns_citations():
    ingest("test-persona", samples=True)
    out = consult("how do I debug a bug in my gradients", k=3, persona="test-persona")
    assert len(out) == 3 and all("youtube.com" in c["citation"] for c in out)


def test_sandbox_blocks_escape():
    with pytest.raises(sb.SandboxViolation):
        sb.safe_path("t-run", "../../etc/passwd")
    sb.project_dir("t-run").mkdir(parents=True, exist_ok=True)
    with pytest.raises(sb.SandboxViolation):
        sb.run_cmd("t-run", ["rm", "-rf", "/tmp/whatever"])


def test_debug_adds_missing_module():
    sb.write_file("t-dbg", "requirements.txt", "pytest\n")
    state = {"run_id": "t-dbg", "current_task": 0, "debug_loops": 0, "persona": "test-persona",
             "test_results": {"steps": [{"ok": False, "cmd": "pytest", "stdout": "", "stderr": "ModuleNotFoundError: No module named 'flask'"}]}}
    out = debug(state)
    assert "flask" in sb.read_file("t-dbg", "requirements.txt") and out["debug_loops"] == 1


def test_budgets():
    assert budget_exceeded({"iteration": 999, "budget_used": {"started_at": time.time()}})
    assert budget_exceeded({"iteration": 0, "budget_used": {"started_at": time.time() - 99999}})
    assert budget_exceeded({"iteration": 0, "budget_used": {"started_at": time.time()}}) is None


def test_parse_constraints():
    assert parse_constraints("A 24-hour hackathon using Flask")["duration_hours"] == 24
