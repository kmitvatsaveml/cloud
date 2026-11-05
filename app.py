"""HackAgent UI (Streamlit).

Tabs: Build (autonomous builder with live log), Copilot (multimodal tool agent),
Courses (per-course notes), History (past runs).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from config import get_settings

S = get_settings()
st.set_page_config(page_title="HackAgent", page_icon="🤖", layout="wide")


def personas() -> list[str]:
    d = S.personas_dir
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🤖 HackAgent")
    st.caption("Offline mode (no ANTHROPIC_API_KEY): templates + heuristics" if S.offline else f"LLM: {S.model}")
    plist = personas() or [S.default_persona]
    persona = st.selectbox("Expert persona / course", plist,
                           index=plist.index(S.default_persona) if S.default_persona in plist else 0)
    with st.expander("➕ Add persona / course from playlist"):
        slug = st.text_input("Slug", placeholder="my-expert")
        url = st.text_input("YouTube playlist URL")
        ups = st.file_uploader("…or upload transcripts (.json/.txt/.srt/.vtt)", accept_multiple_files=True)
        if st.button("Ingest + build persona + course notes") and slug:
            from ingest import ingest
            from notes import build_course_notes
            from persona import build_persona

            with st.spinner("Ingesting…"):
                paths = []
                for u in ups or []:
                    p = Path(tempfile.mkdtemp()) / u.name
                    p.write_bytes(u.getvalue())
                    paths.append(str(p))
                try:
                    n = ingest(slug, playlist=url or None, files=paths)
                    build_persona(slug)
                    build_course_notes(slug)
                    st.success(f"Indexed {n} chunks; persona.md and COURSE_NOTES.md written.")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")
    pmd = S.personas_dir / persona / "persona.md"
    with st.expander("📜 persona.md"):
        st.markdown(pmd.read_text() if pmd.exists() else "_Not generated yet: `./run.sh persona`_")

build_tab, copilot_tab, course_tab, hist_tab = st.tabs(["🏗️ Build", "💬 Copilot", "📚 Courses", "🕘 History"])


def show_run(run_id: str) -> None:
    from agent.sandbox import list_files, run_dir

    rd = run_dir(run_id)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Files**")
        st.code("\n".join(list_files(run_id)) or "(none)")
        z = rd / f"{run_id}.zip"
        if z.exists():
            st.download_button("⬇️ Download .zip", z.read_bytes(), file_name=z.name, key=f"dl-{run_id}")
    with c2:
        for doc in ("README.md", "RUN_REPORT.md", "PITCH.md", "ASSUMPTIONS.md"):
            if (rd / doc).exists():
                with st.expander(doc, expanded=doc == "README.md"):
                    st.markdown((rd / doc).read_text())


# ---------------------------------------------------------------- build
with build_tab:
    problem = st.text_area("Hackathon problem statement", height=120,
                           value="Build a web app that detects plant disease from a leaf photo and suggests treatment. 24-hour hackathon.")
    imgs = st.file_uploader("Optional: images (problem screenshots, sketches)", type=["png", "jpg", "jpeg", "webp"],
                            accept_multiple_files=True, key="build_imgs")
    if st.button("🚀 Run", type="primary") and problem.strip():
        from agent.graph import run_agent

        cons = {"images": [{"media_type": i.type, "data": __import__("base64").b64encode(i.getvalue()).decode()} for i in imgs or []]}
        box, lines, rid = st.empty(), [], None
        with st.status("Agent running…", expanded=True) as status:
            for ev in run_agent(problem, persona=persona, constraints=cons):
                rid = ev["run_id"]
                if ev["node"] != "__end__":
                    lines.append(ev["log"])
                    box.code("\n".join(lines[-40:]))
            status.update(label=f"Done: {rid}", state="complete")
        st.session_state["last_run"] = rid
    if st.session_state.get("last_run"):
        show_run(st.session_state["last_run"])

# ---------------------------------------------------------------- copilot
with copilot_tab:
    st.caption("Tool-calling agent: consult_expert · read_course_notes · web_search · run_python · list_runs · read_run_file · build_project")
    hist = st.session_state.setdefault("chat", [])
    for m in hist:
        st.chat_message(m["role"]).markdown(m["content"])
    cimgs = st.file_uploader("Attach images (screenshots, diagrams, error messages)", type=["png", "jpg", "jpeg", "webp"],
                             accept_multiple_files=True, key="chat_imgs")
    if msg := st.chat_input("Ask the copilot…"):
        from agent.toolagent import chat

        st.chat_message("user").markdown(msg)
        images = [{"media_type": i.type, "data": __import__("base64").b64encode(i.getvalue()).decode()} for i in cimgs or []]
        with st.spinner("Thinking + using tools…"):
            res = chat(msg, images=images, persona=persona, history=hist[-8:])
        with st.chat_message("assistant"):
            st.markdown(res["answer"])
            for s in res["steps"]:
                with st.expander(f"🔧 {s['tool']}({json.dumps(s['args'])[:80]})"):
                    st.code(s["output"][:3000])
        hist += [{"role": "user", "content": msg}, {"role": "assistant", "content": res["answer"]}]

# ---------------------------------------------------------------- courses
with course_tab:
    from notes import build_course_notes, list_courses

    courses = list_courses()
    if not courses:
        st.info("No courses ingested yet. Use the sidebar to add one.")
    for c in courses:
        p = S.root / "courses" / c / "COURSE_NOTES.md"
        with st.expander(f"📘 {c}", expanded=c == persona):
            if st.button("Regenerate notes", key=f"regen-{c}"):
                build_course_notes(c)
            if p.exists():
                st.download_button("⬇️ COURSE_NOTES.md", p.read_bytes(), file_name=f"{c}-COURSE_NOTES.md", key=f"n-{c}")
                st.markdown(p.read_text())

# ---------------------------------------------------------------- history
with hist_tab:
    runs = sorted((p for p in S.runs_dir.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))), reverse=True) if S.runs_dir.exists() else []
    if not runs:
        st.info("No runs yet.")
    for r in runs[:30]:
        prob = (r / "PROBLEM.txt").read_text()[:90] if (r / "PROBLEM.txt").exists() else ""
        status = "✅" if (r / "RUN_REPORT.md").exists() and "PASS" in (r / "RUN_REPORT.md").read_text() else "⏳/❌"
        with st.expander(f"{status} {r.name}: {prob}"):
            show_run(r.name)
            if not (r / "RUN_REPORT.md").exists() and st.button("Resume from checkpoint", key=f"res-{r.name}"):
                from agent.graph import resume

                for ev in resume(r.name):
                    st.text(ev["log"])
