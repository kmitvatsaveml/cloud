"""Graph nodes: analyze, research, plan, code, execute, debug, review, package.

Each node returns a partial state update and appends one line to `log`.
With an LLM, nodes prompt Claude; offline, they fall back to templates and heuristics.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from agent import sandbox as sb
from agent.templates import TEMPLATES, pick_template, render_file
from config import get_settings
from llm import complete, extract_files, extract_json
from memory import consult, format_context

SYSTEM = ("You are HackAgent, an autonomous senior engineer winning a hackathon. You think like this expert:\n"
          "{persona}\nBe concrete. Prefer the simplest stack that demos well and runs locally with no paid services.")


# ---------------------------------------------------------------- helpers
def _sys(state: dict) -> str:
    return SYSTEM.format(persona=state.get("persona_md", "")[:4000])


def _ask(state: dict, prompt: str) -> tuple[str | None, int]:
    return complete(_sys(state), prompt, images=state.get("constraints", {}).get("images"))


def _consult(state: dict, q: str, k: int = 4) -> tuple[str, list[dict[str, str]]]:
    chunks = consult(q, k=k, persona=state.get("persona"))
    cites = [{"question": q, "citation": c["citation"], "text": c["text"][:200]} for c in chunks]
    return format_context(chunks), cites


def _tokens(state: dict, n: int) -> dict:
    b = dict(state.get("budget_used") or {})
    b["tokens"] = b.get("tokens", 0) + n
    return b


def _entry(node: str, status: str, detail: str) -> str:
    return f"{time.strftime('%H:%M:%S')} | {node:<8} | {status:<4} | {detail}"


def parse_constraints(problem: str) -> dict[str, Any]:
    """Extract duration / stack / theme hints from free text."""
    c: dict[str, Any] = {}
    m = re.search(r"(\d+)\s*-?\s*(hour|hr|h)\b", problem, re.I)
    if m:
        c["duration_hours"] = int(m.group(1))
    for stack in ("python", "flask", "fastapi", "streamlit", "react", "node", "next.js"):
        if stack in problem.lower():
            c.setdefault("stack_hints", []).append(stack)
    return c


# ---------------------------------------------------------------- nodes
def analyze(state: dict) -> dict:
    """Parse the problem and consult the expert on how to approach it."""
    problem = state["problem"]
    constraints = {**parse_constraints(problem), **(state.get("constraints") or {})}
    ctx, cites = _consult(state, f"How should I scope and approach under time pressure: {problem}")
    ctx2, cites2 = _consult(state, "start simple baseline first, what to avoid, do and don't rules")
    text, tok = _ask(state, (
        f"Problem: {problem}\nConstraints: {json.dumps({k: v for k, v in constraints.items() if k != 'images'})}\n"
        f"Expert memory:\n{ctx}\n{ctx2}\n\nReturn JSON with keys: core_problem, mvp, stretch_goals (list), "
        "risks (list), libraries (list of libs/APIs worth researching), assumptions (list), "
        "expert_influence (one sentence on how the expert's advice shaped this)."))
    a = extract_json(text)
    tpl = pick_template(problem)
    if not isinstance(a, dict):
        t = TEMPLATES[tpl]
        a = {"core_problem": problem.split(".")[0], "mvp": t["mvp"], "stretch_goals": t["stretch"], "risks": t["risks"],
             "libraries": [], "expert_influence": "Start with the simplest baseline that works end to end, then iterate.",
             "assumptions": [f"Offline mode: no LLM available, so I used the '{tpl}' project template ({t['stack']}).",
                             f"Time box: {constraints.get('duration_hours', 24)} hours; the MVP is scoped to fit a demo."]}
    a["template"] = tpl
    return {"analysis": a, "constraints": constraints, "budget_used": _tokens(state, tok),
            "assumptions": list(a.get("assumptions", [])), "citations": cites + cites2,
            "log": [_entry("analyze", "ok", f"MVP: {str(a.get('mvp'))[:110]}")]}


def research(state: dict) -> dict:
    """Optional Tavily lookups. Skipped silently without TAVILY_API_KEY."""
    s = get_settings()
    libs = (state.get("analysis") or {}).get("libraries") or []
    if not s.tavily_key:
        return {"research": [], "log": [_entry("research", "skip", "no TAVILY_API_KEY")]}
    out = []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=s.tavily_key)
        for q in (libs[:3] or [state["problem"][:200] + " best open-source library python"]):
            r = client.search(str(q), max_results=3)
            out += [{"query": q, "title": x["title"], "url": x["url"], "content": x["content"][:400]} for x in r["results"]]
    except Exception as exc:
        return {"research": [], "log": [_entry("research", "fail", f"{type(exc).__name__}: continuing without")]}
    return {"research": out, "log": [_entry("research", "ok", f"{len(out)} results for {len(libs[:3]) or 1} queries")]}


def plan(state: dict) -> dict:
    """Write spec checklist, file tree and ordered task list."""
    a = state["analysis"]
    ctx, cites = _consult(state, "how to plan and order the work: get an end-to-end pipeline working first")
    research_txt = "\n".join(f"- {r['title']}: {r['content'][:200]} ({r['url']})" for r in state.get("research", []))
    text, tok = _ask(state, (
        f"Problem: {state['problem']}\nAnalysis: {json.dumps(a)}\nResearch:\n{research_txt}\nExpert memory:\n{ctx}\n\n"
        "Plan a Python project. Rules: requirements.txt; tests in tests/ runnable with pytest; a smoke_test.py "
        "that exits 0 when the app works (no network, no servers left running, use test clients); no paid APIs; "
        "no large model downloads in tests. Return JSON: {\"spec\": [requirement], \"file_tree\": [path], "
        "\"tasks\": [{\"title\": str, \"files\": [path], \"check\": str}]} with 3-6 ordered tasks, the smoke test last."))
    p = extract_json(text)
    if not (isinstance(p, dict) and p.get("tasks")):
        t = TEMPLATES[a["template"]]
        p = {"spec": t["spec"], "tasks": [dict(x) for x in t["tasks"]],
             "file_tree": sorted({f for x in t["tasks"] for f in x["files"]}), "offline_template": True}
    tasks = [{**t, "id": i, "status": "todo"} for i, t in enumerate(p["tasks"])]
    return {"mode": "template" if p.get("offline_template") else "llm", "spec": p["spec"], "file_tree": p.get("file_tree", []), "plan": tasks, "current_task": 0,
            "debug_loops": 0, "citations": cites, "budget_used": _tokens(state, tok),
            "log": [_entry("plan", "ok", f"{len(p['spec'])} requirements, {len(tasks)} tasks")]}


def code(state: dict) -> dict:
    """Implement the current task by writing real files into project/."""
    run_id, tasks = state["run_id"], [dict(t) for t in state["plan"]]
    i = state.get("current_task", 0)
    task = tasks[i]
    tok, written = 0, []
    if state.get("mode") == "template":
        for f in task["files"]:
            sb.write_file(run_id, f, render_file(state["analysis"]["template"], f, state["problem"]))
            written.append(f)
    else:
        existing = "\n".join(f'<file path="{f}">\n{sb.read_file(run_id, f)[:4000]}</file>' for f in sb.list_files(run_id))
        text, tok = _ask(state, (
            f"Problem: {state['problem']}\nSpec: {json.dumps(state['spec'])}\nFull plan: {json.dumps(tasks)}\n"
            f"Existing files:\n{existing[:30000]}\n\nImplement task #{i}: {task['title']} (files: {task['files']}; "
            f"check: {task['check']}). Include a minimal test for it. Output complete file contents only, as "
            '<file path="relative/path">...</file> blocks. Keep requirements.txt cumulative.'))
        for f, body in extract_files(text).items():
            sb.write_file(run_id, f, body)
            written.append(f)
    task["status"] = "coded"
    tasks[i] = task
    return {"plan": tasks, "files": sb.list_files(run_id), "debug_loops": 0, "budget_used": _tokens(state, tok),
            "log": [_entry("code", "ok" if written else "warn", f"task {i}: {task['title']} -> {', '.join(written) or 'no files'}")]}



def execute(state: dict) -> dict:
    """Install deps in the run venv, then run pytest and the smoke test."""
    run_id = state["run_id"]
    results: dict[str, Any] = {"steps": []}
    env = sb.ensure_env(run_id)
    results["steps"].append(env)
    py = str(sb.venv_python(run_id))
    if env["ok"]:
        files = sb.list_files(run_id)
        if any(f.startswith("tests/") and f.endswith(".py") for f in files):
            r = sb.run_cmd(run_id, [py, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"], timeout=300)
            r["ok"] = r["ok"] or r["code"] == 5  # 5 = no tests collected yet
            results["steps"].append(r)
        if "smoke_test.py" in files and all(s["ok"] for s in results["steps"]):
            results["steps"].append(sb.run_cmd(run_id, [py, "smoke_test.py"], timeout=120))
    results["ok"] = all(s["ok"] for s in results["steps"])
    failed = next((s for s in results["steps"] if not s["ok"]), None)
    summary = "all checks passed" if results["ok"] else f"FAILED `{failed['cmd'][:60]}`: {(failed['stderr'] or failed['stdout']).strip()[-140:]}"
    passed = re.search(r"(\d+) passed", " ".join(s.get("stdout", "") for s in results["steps"]))
    results["passed"] = int(passed.group(1)) if passed else 0
    return {"test_results": results, "iteration": state.get("iteration", 0) + 1,
            "log": [_entry("execute", "ok" if results["ok"] else "fail", summary.replace("\n", " "))]}


def debug(state: dict) -> dict:
    """Diagnose from logs with the expert's debugging playbook; patch minimal files."""
    run_id = state["run_id"]
    loops = state.get("debug_loops", 0) + 1
    steps = state["test_results"]["steps"]
    failed = next(s for s in steps if not s["ok"])
    logs = f"$ {failed['cmd']}\nSTDOUT:\n{failed['stdout'][-3000:]}\nSTDERR:\n{failed['stderr'][-3000:]}"
    ctx, cites = _consult(state, "debugging playbook: how to find a bug when tests fail, check print verify")
    files = "\n".join(f'<file path="{f}">\n{sb.read_file(run_id, f)[:5000]}</file>' for f in sb.list_files(run_id))
    text, tok = _ask(state, (
        f"Tests are failing (debug loop {loops}).\n{logs}\n\nExpert debugging playbook:\n{ctx}\n\nProject files:\n"
        f"{files[:40000]}\n\nDiagnose in one sentence starting with 'DIAGNOSIS:', then output ONLY the files you change as "
        '<file path="...">...</file> blocks (minimal patch).'))
    patches = extract_files(text)
    diag = re.search(r"DIAGNOSIS:(.*)", text or "")
    if not patches:  # heuristic fixes (offline)
        req = sb.read_file(run_id, "requirements.txt")
        miss = re.search(r"No module named '([\w\.]+)'", logs)
        bad = re.search(r"No matching distribution found for ([\w\-\[\]]+)", logs)
        if miss and miss.group(1).split(".")[0] not in req:
            patches["requirements.txt"] = req.rstrip() + f"\n{miss.group(1).split('.')[0]}\n"
        elif bad:
            patches["requirements.txt"] = "\n".join(l for l in req.splitlines() if not l.startswith(bad.group(1))) + "\n"
    for f, body in patches.items():
        sb.write_file(run_id, f, body)
    detail = (diag.group(1).strip() if diag else "heuristic patch") if patches else "no patch found"
    return {"debug_loops": loops, "citations": cites, "budget_used": _tokens(state, tok),
            "errors": [f"task {state.get('current_task')}: {failed['cmd'][:80]} -> {(failed['stderr'] or failed['stdout'])[-200:]}"],
            "log": [_entry("debug", "ok" if patches else "warn", f"loop {loops}: {detail[:120]} ({', '.join(patches) or '-'})")]}


def give_up_task(state: dict) -> dict:
    """After MAX_DEBUG_LOOPS: record the failure, stub the task and move on."""
    tasks = [dict(t) for t in state["plan"]]
    i = state.get("current_task", 0)
    tasks[i]["status"] = "stubbed"
    return {"plan": tasks, "current_task": i + 1, "debug_loops": 0,
            "limitations": [f"Task '{tasks[i]['title']}' still failed after {state.get('debug_loops')} debug loops; left as-is (stubbed)."],
            "assumptions": [f"Simplified scope: gave up on '{tasks[i]['title']}' to protect the demo."],
            "log": [_entry("debug", "stub", f"task {i} stubbed after max debug loops")]}


def advance(state: dict) -> dict:
    """Mark the current task done and move to the next."""
    tasks = [dict(t) for t in state["plan"]]
    i = state.get("current_task", 0)
    if i < len(tasks):
        tasks[i]["status"] = "done"
    return {"plan": tasks, "current_task": i + 1, "log": [_entry("advance", "ok", f"task {i} done")]}


def review(state: dict) -> dict:
    """Critic: score each spec requirement as met / not met with evidence."""
    run_id = state["run_id"]
    files = {f: sb.read_file(run_id, f) for f in sb.list_files(run_id)}
    blob = "\n".join(f'<file path="{f}">\n{c[:4000]}</file>' for f, c in files.items())
    tr = state.get("test_results", {})
    text, tok = _ask(state, (
        f"Spec: {json.dumps(state['spec'])}\nTests ok: {tr.get('ok')} ({tr.get('passed', 0)} passed)\nFiles:\n{blob[:40000]}\n\n"
        "Act as a strict hackathon judge. Return JSON {\"items\": [{\"requirement\": str, \"met\": bool, \"evidence\": str}], "
        "\"gaps\": [{\"title\": str, \"files\": [path], \"check\": str}]}"))
    r = extract_json(text)
    if not (isinstance(r, dict) and r.get("items")):
        alltext = " ".join(files.values()).lower()
        items = []
        for req in state["spec"]:
            kws = [w for w in re.findall(r"[a-z]{4,}", req.lower()) if w not in {"with", "each", "runs", "that", "from"}]
            hits = [w for w in kws if w in alltext]
            met = bool(tr.get("ok")) and len(hits) >= max(1, len(kws) // 3)
            items.append({"requirement": req, "met": met, "evidence": f"keywords found in code: {hits[:5]}; tests ok={tr.get('ok')}"})
        r = {"items": items, "gaps": []}
    met = sum(1 for x in r["items"] if x.get("met"))
    gaps = [g for g in r.get("gaps", []) if isinstance(g, dict) and g.get("title")]
    update: dict[str, Any] = {"review": r, "review_rounds": state.get("review_rounds", 0) + 1, "budget_used": _tokens(state, tok),
                              "log": [_entry("review", "ok", f"{met}/{len(r['items'])} requirements met, {len(gaps)} gaps")]}
    if gaps:
        tasks = [dict(t) for t in state["plan"]]
        tasks += [{**g, "id": len(tasks) + j, "status": "todo"} for j, g in enumerate(gaps[:2])]
        update["plan"] = tasks
    return update


def package(state: dict) -> dict:
    """Write README, PITCH, ASSUMPTIONS, RUN_REPORT and zip the run."""
    from agent.docs import write_docs

    run_id = state["run_id"]
    tok = write_docs(state)
    rd = sb.run_dir(run_id)
    zip_base = rd.parent / f"{run_id}"
    staging = rd.parent / f".{run_id}_zip"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(rd, staging / run_id, ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "*.zip"))
    archive = shutil.make_archive(str(zip_base), "zip", staging)
    shutil.rmtree(staging)
    shutil.move(archive, rd / f"{run_id}.zip")
    return {"budget_used": _tokens(state, tok),
            "log": [_entry("package", "ok", f"docs written, zip -> runs/{run_id}/{run_id}.zip ({state.get('stop_reason') or 'completed'})")]}
