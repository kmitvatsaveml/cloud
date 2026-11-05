from app import app
from core import analyze


def test_analyze():
    a = analyze("Hackathons are fun. Hackathons are fast.")
    assert a["keywords"][0] == "hackathons" and a["summary"].startswith("Hackathons")


def test_api_roundtrip():
    c = app.test_client()
    assert c.post("/api/items", json={"text": "hello world"}).status_code == 201
    assert len(c.get("/api/items").json) >= 1
    assert c.post("/api/items", json={}).status_code == 400
