"""API tests through Flask's test client."""
import io

from PIL import Image

from app import app


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    buf.seek(0)
    return buf


def test_health():
    assert app.test_client().get("/health").json == {"status": "ok"}


def test_predict_returns_treatment():
    r = app.test_client().post("/api/predict", data={"image": (_png((40, 140, 40)), "leaf.png")},
                               content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.json
    assert body["condition"] and isinstance(body["treatment"], list) and body["treatment"]


def test_predict_rejects_missing_file():
    assert app.test_client().post("/api/predict").status_code == 400


def test_index_page():
    assert b"LeafDoctor" in app.test_client().get("/").data
