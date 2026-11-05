"""Writes README.md, PITCH.md, ASSUMPTIONS.md and RUN_REPORT.md for a run."""
from __future__ import annotations

import json

from agent import sandbox as sb
from llm import complete


def _cites(state: dict, n: int = 8) -> list[str]:
    seen, out = set(), []
    for c in state.get("citations", []):
        if c["citation"] not in seen:
            seen.add(c["citation"])
            out.append(f"- {c['citation']}: \"{c['text'][:140]}...\"")
    return out[:n]


def _mermaid(state: dict) -> str:
    files = [f for f in sb.list_files(state["run_id"]) if f.endswith((".py", ".html")) and not f.startswith("tests/")]
    lines = ["```mermaid", "flowchart LR", "  U[User / Browser] --> A[app.py]"]
    for f in files:
        if f != "app.py":
            node = f.replace("/", "_").replace(".", "_")
            lines.append(f"  A --> {node}[{f}]")
    lines += ["  T[tests + smoke_test.py] -.verify.-> A", "```"]
    return "\n".join(lines)


def write_docs(state: dict) -> int:
    """Generate all run documents. Returns tokens used."""
    run_id, a = state["run_id"], state.get("analysis", {})
    rd = sb.run_dir(run_id)
    tr = state.get("test_results", {})
    review = state.get("review", {}).get("items", [])
    title = a.get("title") or (a.get("template") and {"leaf_doctor": "LeafDoctor"}.get(a["template"])) or "HackAgent Project"
    cites = _cites(state)
    files = sb.list_files(run_id)
    tokens = 0

    readme = f"""# {title}

> Generated autonomously by **HackAgent** (run `{run_id}`).

**Problem:** {state['problem']}

**MVP:** {a.get('mvp', '')}

## Setup
```bash
cd project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage
```bash
python app.py            # then open http://localhost:5000
pytest -q                # unit tests
python smoke_test.py     # end-to-end smoke check
```

## Architecture
{_mermaid(state)}

## Files
{chr(10).join(f'- `{f}`' for f in files)}

## Expert influence
{a.get('expert_influence', '')}

{chr(10).join(cites[:4])}
"""
    text, t = complete("You write crisp hackathon pitches.", (
        f"Write PITCH.md (markdown) for this hackathon project. Sections: Problem, Solution, Demo script (numbered, 2 minutes), "
        f"Why it's technically impressive, What's next.\nProblem: {state['problem']}\nAnalysis: {json.dumps(a)[:3000]}\n"
        f"Files: {files}\nTests passed: {tr.get('passed', 0)}"))
    tokens += t
    pitch = text or f"""# {title}: Pitch

## Problem
{a.get('core_problem', state['problem'])}

## Solution
{a.get('mvp', '')}

## Demo script (2 min)
1. Open the app and state the problem in one sentence.
2. Upload a sample input and show the result plus the actionable recommendation.
3. Upload a bad input to show graceful error handling.
4. Show `pytest` passing live ({tr.get('passed', 0)} tests) as proof that it works.
5. Close with the roadmap: {', '.join(a.get('stretch_goals', [])[:2])}.

## Why it's technically impressive
- Built end to end by an autonomous agent: analyze, plan, code, execute, debug, review, with zero human input.
- Interpretable baseline first, following the expert's advice to start simple and verify every step.
- Verified with automated tests and an end-to-end smoke test.

## What's next
{chr(10).join(f'- {s}' for s in a.get('stretch_goals', []))}
"""
    assumptions = "# Assumptions\n\nDecisions HackAgent made on its own:\n\n" + "\n".join(
        f"- {x}" for x in dict.fromkeys(state.get("assumptions", []))) + "\n"

    report = [f"# Run report: `{run_id}`", "", f"**Stop reason:** {state.get('stop_reason') or 'completed normally'}",
              f"**Iterations:** {state.get('iteration', 0)} · **Tokens:** {state.get('budget_used', {}).get('tokens', 0)}",
              f"**Tests:** {'PASS' if tr.get('ok') else 'FAIL'} ({tr.get('passed', 0)} passed)", "",
              "## What was built", *[f"- [{t.get('status')}] {t['title']}" for t in state.get("plan", [])], "",
              "## Spec review", *[f"- {'✅' if r.get('met') else '❌'} {r['requirement']} ({r.get('evidence', '')})" for r in review], "",
              "## Known limitations", *([f"- {l}" for l in state.get("limitations", [])] or ["- None recorded."]),
              *[f"- Stretch goal (not built): {s}" for s in a.get("stretch_goals", [])], "",
              "## Expert citations (where the persona shaped decisions)", *(cites or ["- (no persona memory indexed)"]), "",
              "## Errors encountered", *([f"- `{e[:200]}`" for e in state.get("errors", [])] or ["- None."]), "",
              "## Agent step log", "```", *state.get("log", []), "```"]
    (rd / "README.md").write_text(readme)
    (rd / "PITCH.md").write_text(pitch)
    (rd / "ASSUMPTIONS.md").write_text(assumptions)
    (rd / "RUN_REPORT.md").write_text("\n".join(report) + "\n")
    sb.write_file(run_id, "README.md", readme)
    return tokens
