"""Sandboxed execution inside `./runs/{run_id}/`.

- every file write is resolved and must stay under the run's `project/` dir
- every command runs with cwd inside the run dir and a hard timeout
- destructive commands targeting paths outside the run dir are blocked
"""
from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from config import get_settings

DESTRUCTIVE = {"rm", "rmdir", "del", "shred", "mv", "unlink", "dd", "mkfs", "chmod", "chown"}


class SandboxViolation(RuntimeError):
    """Raised when generated code tries to escape the run folder."""


def run_dir(run_id: str) -> Path:
    return get_settings().runs_dir / run_id


def project_dir(run_id: str) -> Path:
    return run_dir(run_id) / "project"


def safe_path(run_id: str, rel: str) -> Path:
    """Resolve `rel` inside project/ or raise SandboxViolation."""
    base = project_dir(run_id).resolve()
    p = (base / rel.lstrip("/")).resolve()
    if base != p and base not in p.parents:
        raise SandboxViolation(f"path escapes project dir: {rel}")
    return p


def write_file(run_id: str, rel: str, content: str) -> Path:
    p = safe_path(run_id, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def read_file(run_id: str, rel: str) -> str:
    p = safe_path(run_id, rel)
    return p.read_text(errors="ignore") if p.exists() else ""


def list_files(run_id: str) -> list[str]:
    base = project_dir(run_id)
    skip = {".venv", "__pycache__", ".pytest_cache", "node_modules"}
    return sorted(
        str(p.relative_to(base)) for p in base.rglob("*")
        if p.is_file() and not (set(p.relative_to(base).parts) & skip)
    )


def check_command(cmd: list[str], rd: Path) -> None:
    """Block destructive commands whose path args resolve outside the run dir."""
    if Path(cmd[0]).name not in DESTRUCTIVE:
        return
    for arg in cmd[1:]:
        if arg.startswith("-"):
            continue
        p = (rd / arg).resolve() if not os.path.isabs(arg) else Path(arg).resolve()
        if p == rd.resolve() or rd.resolve() not in p.parents:
            raise SandboxViolation(f"blocked destructive command outside run dir: {shlex.join(cmd)}")


def run_cmd(run_id: str, cmd: list[str], cwd: str = "project", timeout: int | None = None,
            env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run a command inside the run dir with a timeout. Never raises on failure."""
    rd = run_dir(run_id).resolve()
    wd = (rd / cwd).resolve()
    if wd != rd and rd not in wd.parents:
        raise SandboxViolation(f"cwd escapes run dir: {cwd}")
    check_command(cmd, rd)
    timeout = timeout or get_settings().cmd_timeout
    full_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", **(env or {})}
    full_env.pop("VIRTUAL_ENV", None)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=timeout, env=full_env)
        out = {"ok": r.returncode == 0, "code": r.returncode, "stdout": r.stdout[-6000:], "stderr": r.stderr[-6000:]}
    except subprocess.TimeoutExpired as e:
        out = {"ok": False, "code": -1, "stdout": str(e.stdout or "")[-3000:], "stderr": f"TIMEOUT after {timeout}s"}
    except FileNotFoundError as e:
        out = {"ok": False, "code": -1, "stdout": "", "stderr": str(e)}
    out.update(cmd=shlex.join(cmd), seconds=round(time.time() - t0, 1))
    return out


def venv_python(run_id: str) -> Path:
    return run_dir(run_id) / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_env(run_id: str) -> dict[str, Any]:
    """Create the run venv (once) and install requirements when they change."""
    rd = run_dir(run_id)
    py = venv_python(run_id)
    if not py.exists():
        r = run_cmd(run_id, [sys.executable, "-m", "venv", ".venv"], cwd=".", timeout=180)
        if not r["ok"]:
            return r
    req = project_dir(run_id) / "requirements.txt"
    if not req.exists():
        return {"ok": True, "cmd": "no requirements.txt", "stdout": "", "stderr": "", "code": 0}
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    marker = rd / ".venv" / ".req_hash"
    if marker.exists() and marker.read_text() == digest:
        return {"ok": True, "cmd": "requirements unchanged", "stdout": "", "stderr": "", "code": 0}
    r = run_cmd(run_id, [str(py), "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    if r["ok"]:
        marker.write_text(digest)
    return r
