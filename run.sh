#!/usr/bin/env bash
# HackAgent task runner: ./run.sh {install|ingest|persona|notes|app|agent|copilot|test}
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
PERSONA="${PERSONA:-karpathy-zero-to-hero}"
PLAYLIST="${PLAYLIST:-https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ}"
case "${1:-help}" in
  install) python3.11 -m venv .venv && $PY -m pip install -U pip && $PY -m pip install -r requirements.txt
           [ -f .env ] || cp .env.example .env ;;
  ingest)  if [ "${2:-}" = "--samples" ]; then $PY ingest.py --samples --persona "$PERSONA"
           else $PY ingest.py --playlist "${2:-$PLAYLIST}" --persona "$PERSONA"; fi ;;
  persona) $PY persona.py --persona "${2:-$PERSONA}" ;;
  notes)   $PY notes.py --course "${2:-$PERSONA}" ;;
  app)     $PY -m streamlit run app.py ;;
  agent)   shift; $PY -m agent.graph "$@" ;;
  copilot) shift; $PY -m agent.toolagent "$@" ;;
  test)    $PY -m pytest -q tests ;;
  *) echo "usage: ./run.sh {install|ingest [playlist|--samples]|persona|notes|app|agent \"problem\"|copilot \"question\"|test}" ;;
esac
