"""Part B: the autonomous LangGraph StateGraph.

analyze -> research -> plan -> code -> execute -> (debug <-> execute)* -> advance -> code ... -> review -> (code | package)

No interrupts, no human in the loop. Hard budgets route straight to `package`.
State is checkpointed in SQLite, so `resume(run_id)` continues a crashed run.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.sandbox import run_dir
from agent.state import AgentState
from config import get_settings


def budget_exceeded(state: dict) -> str | None:
    """Return a reason string if any hard budget is hit."""
    s = get_settings()
    b = state.get("budget_used") or {}
    if state.get("iteration", 0) >= s.max_iterations:
        return f"MAX_ITERATIONS={s.max_iterations} reached"
    if time.time() - b.get("started_at", time.time()) > s.max_wall_minutes * 60:
        return f"MAX_WALL_MINUTES={s.max_wall_minutes} reached"
    if b.get("tokens", 0) > s.max_tokens_budget:
        return f"token budget {s.max_tokens_budget} reached"
    return None


def _guard(name: str, fn):
    """Wrap a node: skip work and flag stop_reason when a budget is hit."""
    def wrapped(state: dict) -> dict:
        reason = budget_exceeded(state)
        if reason and not state.get("stop_reason"):
            return {"stop_reason": reason, "log": [nodes._entry(name, "stop", f"budget hit: {reason} -> package")]}
        if state.get("stop_reason"):
            return {}
        return fn(state)
    wrapped.__name__ = name
    return wrapped


def _stopped(state: dict) -> bool:
    return bool(state.get("stop_reason"))


def after_execute(state: dict) -> str:
    if _stopped(state):
        return "package"
    if state["test_results"].get("ok"):
        return "advance"
    return "debug" if state.get("debug_loops", 0) < get_settings().max_debug_loops else "give_up"


def after_advance(state: dict) -> str:
    if _stopped(state):
        return "package"
    return "code" if state.get("current_task", 0) < len(state.get("plan", [])) else "review"


def after_review(state: dict) -> str:
    if _stopped(state):
        return "package"
    more = state.get("current_task", 0) < len(state.get("plan", []))
    return "code" if more and state.get("review_rounds", 0) < 3 else "package"


def build_graph(checkpointer: Any | None = None):
    """Compile the HackAgent graph."""
    g = StateGraph(AgentState)
    for name in ("analyze", "research", "plan", "code", "execute", "debug", "advance", "review"):
        g.add_node(name, _guard(name, getattr(nodes, name)))
    g.add_node("give_up", _guard("give_up", nodes.give_up_task))
    g.add_node("package", nodes.package)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", lambda s: "package" if _stopped(s) else "research", ["research", "package"])
    g.add_conditional_edges("research", lambda s: "package" if _stopped(s) else "plan", ["plan", "package"])
    g.add_conditional_edges("plan", lambda s: "package" if _stopped(s) else "code", ["code", "package"])
    g.add_conditional_edges("code", lambda s: "package" if _stopped(s) else "execute", ["execute", "package"])
    g.add_conditional_edges("execute", after_execute, ["advance", "debug", "give_up", "package"])
    g.add_conditional_edges("debug", lambda s: "package" if _stopped(s) else "execute", ["execute", "package"])
    g.add_conditional_edges("advance", after_advance, ["code", "review", "package"])
    g.add_conditional_edges("give_up", after_advance, ["code", "review", "package"])
    g.add_conditional_edges("review", after_review, ["code", "package"])
    g.add_edge("package", END)
    return g.compile(checkpointer=checkpointer)


def _checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(str(get_settings().checkpoint_db), check_same_thread=False)
    return SqliteSaver(conn)


def _persona_md(slug: str) -> str:
    p = get_settings().personas_dir / slug / "persona.md"
    return p.read_text()[:6000] if p.exists() else ""


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def run_agent(problem: str, persona: str | None = None, constraints: dict[str, Any] | None = None,
              run_id: str | None = None) -> Iterator[dict[str, Any]]:
    """Run the agent end to end, yielding {node, log, state} events for live UIs."""
    s = get_settings()
    persona = persona or s.default_persona
    run_id = run_id or new_run_id()
    run_dir(run_id).joinpath("project").mkdir(parents=True, exist_ok=True)
    (run_dir(run_id) / "PROBLEM.txt").write_text(problem)
    init = {"run_id": run_id, "problem": problem, "constraints": constraints or {}, "persona": persona,
            "persona_md": _persona_md(persona), "iteration": 0, "review_rounds": 0,
            "budget_used": {"tokens": 0, "started_at": time.time()}, "log": [], "assumptions": [], "errors": [],
            "citations": [], "limitations": []}
    yield from _stream(init, run_id)


def resume(run_id: str) -> Iterator[dict[str, Any]]:
    """Resume a checkpointed run after a crash."""
    yield from _stream(None, run_id)


def _stream(inp: dict | None, run_id: str) -> Iterator[dict[str, Any]]:
    graph = build_graph(_checkpointer())
    cfg = {"configurable": {"thread_id": run_id}, "recursion_limit": 200, "run_name": f"hackagent-{run_id}",
           "tags": ["hackagent"], "metadata": {"run_id": run_id}}
    for update in graph.stream(inp, cfg, stream_mode="updates"):
        for node, delta in update.items():
            for line in (delta or {}).get("log", []):
                yield {"run_id": run_id, "node": node, "log": line}
    yield {"run_id": run_id, "node": "__end__", "log": "done", "state": graph.get_state(cfg).values}


if __name__ == "__main__":
    import sys

    prob = " ".join(sys.argv[1:]) or "Build a web app that detects plant disease from a leaf photo and suggests treatment. 24-hour hackathon."
    for ev in run_agent(prob):
        print(ev["log"])
