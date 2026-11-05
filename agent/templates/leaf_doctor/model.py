"""Leaf disease classifier.

MVP baseline: an interpretable color-segmentation heuristic. It measures the
share of green, yellow, brown, orange-rust and white-powder pixels inside the
leaf and maps that to a condition. It is fast, needs no GPU or downloads, and
is easy to debug.
Swap-in point: `predict()` keeps the same signature, so a PlantVillage-trained
CNN (for example a MobileNetV2 fine-tune) can replace it later.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass

from PIL import Image

CLASSES = ["healthy", "early_blight", "rust", "powdery_mildew", "chlorosis"]


@dataclass
class Prediction:
    condition: str
    confidence: float
    features: dict[str, float]


def color_features(img: Image.Image, size: int = 128) -> dict[str, float]:
    """Fraction of leaf pixels that fall in each diagnostic color band."""
    img = img.convert("RGB").resize((size, size))
    counts = {"green": 0, "yellow": 0, "brown": 0, "rust": 0, "white": 0}
    leaf = 0
    for r, g, b in img.getdata():
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        hd = h * 360
        if v > 0.85 and s < 0.15:
            counts["white"] += 1; leaf += 1
        elif s < 0.15 or v < 0.08:
            continue  # background (grey / black)
        elif 70 <= hd < 170:
            counts["green"] += 1; leaf += 1
        elif 45 <= hd < 70:
            counts["yellow"] += 1; leaf += 1
        elif 18 <= hd < 45 and v > 0.55 and s > 0.6:
            counts["rust"] += 1; leaf += 1
        elif hd < 45 or hd >= 330:
            counts["brown"] += 1; leaf += 1
    leaf = max(leaf, 1)
    return {k: round(v / leaf, 4) for k, v in counts.items()}


def predict(img: Image.Image) -> Prediction:
    """Classify a leaf image into one of CLASSES."""
    f = color_features(img)
    scores = {
        "powdery_mildew": f["white"] * 2.0,
        "rust": f["rust"] * 2.5,
        "early_blight": f["brown"] * 2.0,
        "chlorosis": f["yellow"] * 1.8,
    }
    worst = max(scores, key=scores.get)
    if scores[worst] < 0.25:
        return Prediction("healthy", round(min(0.99, 0.5 + f["green"] / 2), 2), f)
    return Prediction(worst, round(min(0.99, 0.4 + scores[worst] / 2), 2), f)
