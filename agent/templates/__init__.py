"""Offline project templates.

Used when no LLM is available, so the whole graph (code -> execute -> debug ->
review -> package) still runs end to end. Each template defines a spec,
an ordered task list, and the files each task writes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent

TEMPLATES: dict[str, dict[str, Any]] = {
    "leaf_doctor": {
        "match": ["plant", "leaf", "crop", "disease"],
        "title": "LeafDoctor",
        "stack": "Python + Flask + Pillow (heuristic color-segmentation classifier)",
        "mvp": "Upload a leaf photo, get a predicted condition with a confidence score and treatment steps.",
        "stretch": ["Swap the heuristic for a PlantVillage-trained MobileNetV2",
                    "Grad-CAM heatmap showing which leaf region drove the prediction",
                    "Multilingual treatment advice", "Offline PWA for farmers"],
        "risks": ["Training a CNN within the time box (mitigated: an interpretable heuristic baseline first)",
                  "Domain shift between lab photos and field photos", "Users may treat the output as a definitive diagnosis"],
        "spec": [
            "Web UI to upload a leaf photo",
            "API endpoint /api/predict returns predicted condition and confidence",
            "Treatment suggestions for each detected condition",
            "Handles invalid or missing uploads gracefully",
            "Automated tests for classifier and API",
            "Runs locally with no paid services",
        ],
        "tasks": [
            {"title": "Treatment knowledge base", "files": ["treatments.py", "requirements.txt"], "check": "import treatments"},
            {"title": "Leaf classifier (color-segmentation baseline) + tests",
             "files": ["model.py", "tests/conftest.py", "tests/test_model.py"], "check": "pytest tests/test_model.py"},
            {"title": "Flask API + upload UI + tests",
             "files": ["app.py", "templates/index.html", "tests/test_app.py"], "check": "pytest tests/test_app.py"},
            {"title": "End-to-end smoke test", "files": ["smoke_test.py"], "check": "python smoke_test.py"},
        ],
    },
    "generic_web": {
        "match": [],
        "title": "HackMVP",
        "stack": "Python + Flask",
        "mvp": "A web app with an input form and an API that processes and stores submissions.",
        "stretch": ["Persistence with SQLite", "LLM-powered analysis", "Auth"],
        "risks": ["Ambiguous problem statement (mitigated: generic input -> analysis -> results loop)"],
        "spec": ["Web UI with input form", "REST API to create and list items",
                 "Input analysis (keywords + summary)", "Automated tests", "Runs locally with no paid services"],
        "tasks": [
            {"title": "Core analysis logic", "files": ["core.py", "requirements.txt"], "check": "import core"},
            {"title": "Flask API + UI + tests", "files": ["app.py", "templates/index.html", "tests/conftest.py",
                                                    "tests/test_app.py"], "check": "pytest"},
            {"title": "End-to-end smoke test", "files": ["smoke_test.py"], "check": "python smoke_test.py"},
        ],
    },
}


def pick_template(problem: str) -> str:
    """Choose the template whose keywords best match the problem."""
    low = problem.lower()
    best, score = "generic_web", 0
    for name, t in TEMPLATES.items():
        s = sum(k in low for k in t["match"])
        if s > score:
            best, score = name, s
    return best


def render_file(template: str, rel: str, problem: str) -> str:
    """Return a template file's content with placeholders filled."""
    text = (HERE / template / rel).read_text()
    title = TEMPLATES[template]["title"]
    return text.replace("__TITLE__", title).replace("__PROBLEM__", re.sub(r"[<>]", "", problem)[:300])
