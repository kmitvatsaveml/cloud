"""Smoke check: boot the app in-process and run one end-to-end prediction."""
import io
import sys

from PIL import Image, ImageDraw

from app import app


def main() -> int:
    img = Image.new("RGB", (128, 128), (120, 120, 120))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 120, 120), fill=(40, 140, 40))
    d.ellipse((40, 40, 90, 90), fill=(110, 60, 20))
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
    c = app.test_client()
    assert c.get("/health").status_code == 200
    r = c.post("/api/predict", data={"image": (buf, "leaf.png")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.data
    print("SMOKE OK:", r.json["label"], r.json["confidence"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
