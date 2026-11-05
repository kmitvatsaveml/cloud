"""LangGraph state for a HackAgent run."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state. List fields with `operator.add` are append-only across nodes."""

    run_id: str
    problem: str
    constraints: dict[str, Any]
    persona: str            # persona slug
    persona_md: str         # persona text (truncated) injected into prompts
    analysis: dict[str, Any]
    mode: str               # 'llm' or 'template' (offline)
    research: list[dict[str, Any]]
    spec: list[str]         # requirement checklist
    file_tree: list[str]
    plan: list[dict[str, Any]]   # ordered tasks: {id, title, files, check, status}
    current_task: int
    files: list[str]        # relative paths written into project/
    test_results: dict[str, Any]
    review: dict[str, Any]
    review_rounds: int
    debug_loops: int        # loops for the current task
    iteration: int
    budget_used: dict[str, Any]   # {tokens, started_at}
    stop_reason: str
    errors: Annotated[list[str], operator.add]
    log: Annotated[list[str], operator.add]
    assumptions: Annotated[list[str], operator.add]
    citations: Annotated[list[dict[str, str]], operator.add]
    limitations: Annotated[list[str], operator.add]
