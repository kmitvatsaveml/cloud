"""HackAgent Copilot: a multimodal, tool-calling agent.

Ask it anything about building your project, with optional screenshots or
diagrams. It decides which tools to call:
  - consult_expert       hybrid-RAG over the expert's course transcripts (with citations)
  - read_course_notes    the per-course Markdown notes
  - web_search           Tavily (when TAVILY_API_KEY is set)
  - run_python           runs a snippet in an isolated scratch run folder with a timeout
  - list_runs / read_run_file   inspect projects the builder agent produced
  - build_project        launches the autonomous builder graph on a problem statement

With Claude: LangGraph ReAct agent (`create_react_agent`) with image input.
Offline: a keyword router calls the relevant tools and returns their output.
"""
from __future__ import annotations

import json
import sys
from typing import Any

from langchain_core.tools import tool

from agent import sandbox as sb
from config import get_settings
from llm import get_chat_model, image_blocks
from memory import consult

_PERSONA = {"slug": None}


@tool
def consult_expert(question: str) -> str:
    """Search the expert's course transcripts for advice on scoping, building or debugging. Returns cited passages."""
    chunks = consult(question, k=5, persona=_PERSONA["slug"])
    return "\n".join(f"- {c['text'][:400]} {c['citation']}" for c in chunks) or "No memory indexed for this persona."


@tool
def read_course_notes(course: str = "") -> str:
    """Read the generated COURSE_NOTES.md for a course slug (default: the active persona)."""
    slug = course or _PERSONA["slug"] or get_settings().default_persona
    p = get_settings().root / "courses" / slug / "COURSE_NOTES.md"
    return p.read_text()[:8000] if p.exists() else f"No notes yet for '{slug}'. Run: ./run.sh notes {slug}"


@tool
def web_search(query: str) -> str:
    """Search the web (Tavily) for libraries, APIs and docs."""
    key = get_settings().tavily_key
    if not key:
        return "web_search unavailable: TAVILY_API_KEY not set."
    from tavily import TavilyClient

    r = TavilyClient(api_key=key).search(query, max_results=4)
    return "\n".join(f"- {x['title']} ({x['url']}): {x['content'][:300]}" for x in r["results"])


@tool
def run_python(code: str) -> str:
    """Run a Python snippet in an isolated scratch folder (runs/_scratch) with a 30 s timeout. Returns stdout/stderr."""
    sb.project_dir("_scratch").mkdir(parents=True, exist_ok=True)
    sb.write_file("_scratch", "snippet.py", code)
    r = sb.run_cmd("_scratch", [sys.executable, "snippet.py"], timeout=30)
    return f"exit={r['code']}\nSTDOUT:\n{r['stdout'][-3000:]}\nSTDERR:\n{r['stderr'][-2000:]}"


@tool
def list_runs() -> str:
    """List projects produced by the builder agent, newest first."""
    d = get_settings().runs_dir
    runs = sorted((p for p in d.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))), reverse=True) if d.exists() else []
    return "\n".join(f"{p.name}: {(p / 'PROBLEM.txt').read_text()[:100] if (p / 'PROBLEM.txt').exists() else ''}" for p in runs[:20]) or "No runs yet."


@tool
def read_run_file(run_id: str, path: str) -> str:
    """Read a file from a run (e.g. 'RUN_REPORT.md' or 'project/app.py')."""
    base = sb.run_dir(run_id).resolve()
    p = (base / path).resolve()
    if base not in p.parents:
        return "path outside run folder"
    return p.read_text(errors="ignore")[:8000] if p.exists() else "not found"


@tool
def build_project(problem: str) -> str:
    """Launch the autonomous builder agent on a hackathon problem statement. Returns the run id and final log."""
    from agent.graph import run_agent

    log, rid = [], ""
    for ev in run_agent(problem, persona=_PERSONA["slug"]):
        rid = ev["run_id"]
        if ev["node"] != "__end__":
            log.append(ev["log"])
    return f"run_id={rid}\n" + "\n".join(log[-15:])


TOOLS = [consult_expert, read_course_notes, web_search, run_python, list_runs, read_run_file, build_project]
SYSTEM = ("You are HackAgent Copilot, a senior engineer who learned from the expert's courses. Use tools: cite the "
          "expert (consult_expert) when giving engineering advice, verify code with run_python, and use build_project "
          "only when asked to build something. If images are attached, read them carefully (screenshots, diagrams, errors).")


def _offline(message: str) -> dict[str, Any]:
    low = message.lower()
    calls = []
    if any(w in low for w in ("build", "make me", "create an app")):
        calls.append(("build_project", {"problem": message}))
    if "note" in low or "course" in low or "summar" in low:
        calls.append(("read_course_notes", {}))
    if "run" in low and ("list" in low or "history" in low):
        calls.append(("list_runs", {}))
    if "```" in message:
        calls.append(("run_python", {"code": message.split("```")[1].removeprefix("python\n")}))
    if not calls or any(w in low for w in ("how", "why", "debug", "should", "approach")):
        calls.append(("consult_expert", {"question": message}))
    by = {t.name: t for t in TOOLS}
    steps = [{"tool": n, "args": a, "output": by[n].invoke(a)} for n, a in calls]
    answer = "\n\n".join(f"**{s['tool']}**\n{s['output']}" for s in steps)
    return {"answer": "(offline mode: tool results without LLM synthesis)\n\n" + answer, "steps": steps}


def chat(message: str, images: list[Any] | None = None, persona: str | None = None,
         history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """One copilot turn. Returns {answer, steps:[{tool,args,output}]}."""
    _PERSONA["slug"] = persona or get_settings().default_persona
    model = get_chat_model()
    if model is None:
        return _offline(message)
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(model, TOOLS)
    msgs: list[Any] = [SystemMessage(content=SYSTEM)]
    for h in history or []:
        msgs.append(HumanMessage(content=h["content"]) if h["role"] == "user" else AIMessage(content=h["content"]))
    content: Any = [*image_blocks(images), {"type": "text", "text": message}] if images else message
    msgs.append(HumanMessage(content=content))
    result = agent.invoke({"messages": msgs}, {"recursion_limit": 30, "run_name": "hackagent-copilot"})
    steps, pending = [], {}
    for m in result["messages"]:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                pending[tc["id"]] = {"tool": tc["name"], "args": tc["args"]}
        elif isinstance(m, ToolMessage) and m.tool_call_id in pending:
            steps.append({**pending[m.tool_call_id], "output": str(m.content)[:3000]})
    final = result["messages"][-1].content
    if isinstance(final, list):
        final = "".join(p.get("text", "") for p in final if isinstance(p, dict))
    return {"answer": final, "steps": steps}


if __name__ == "__main__":
    print(json.dumps(chat(" ".join(sys.argv[1:]) or "How should I debug failing tests?"), indent=2)[:4000])
