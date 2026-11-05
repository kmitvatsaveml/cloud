import sys

from app import app

c = app.test_client()
assert c.get("/health").status_code == 200
assert c.post("/api/items", json={"text": "smoke"}).status_code == 201
print("SMOKE OK")
sys.exit(0)
