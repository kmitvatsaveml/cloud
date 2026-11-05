"""LeafDoctor: upload a leaf photo, get a disease prediction and treatment."""
from __future__ import annotations

import io

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from model import predict
from treatments import DISCLAIMER, get_treatment

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def analyze(file_bytes: bytes) -> dict:
    """Run the classifier on raw image bytes and attach treatment advice."""
    img = Image.open(io.BytesIO(file_bytes))
    p = predict(img)
    t = get_treatment(p.condition)
    return {"condition": p.condition, "label": t["label"], "confidence": p.confidence,
            "severity": t["severity"], "treatment": t["advice"], "features": p.features,
            "disclaimer": DISCLAIMER}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def api_predict():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="upload a leaf image in the 'image' field"), 400
    try:
        return jsonify(analyze(f.read()))
    except UnidentifiedImageError:
        return jsonify(error="not a valid image"), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
