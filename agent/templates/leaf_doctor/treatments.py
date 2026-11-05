"""Treatment knowledge base for each detectable condition."""
from __future__ import annotations

TREATMENTS: dict[str, dict[str, object]] = {
    "healthy": {
        "label": "Healthy leaf",
        "severity": "none",
        "advice": [
            "No disease detected. Keep watering at the base of the plant, not on the leaves.",
            "Keep checking every week for spots or discoloration.",
        ],
    },
    "early_blight": {
        "label": "Early blight / leaf spot (fungal)",
        "severity": "moderate",
        "advice": [
            "Remove and dispose of affected leaves (do not compost).",
            "Apply a copper-based or chlorothalonil fungicide per label instructions.",
            "Mulch the soil and water at the base to stop spores splashing up.",
            "Rotate crops and leave space between plants for airflow.",
        ],
    },
    "rust": {
        "label": "Rust (fungal)",
        "severity": "moderate",
        "advice": [
            "Prune infected leaves and clean tools between cuts.",
            "Use a sulfur or neem-oil spray early in the infection.",
            "Avoid overhead watering and water in the morning.",
        ],
    },
    "powdery_mildew": {
        "label": "Powdery mildew (fungal)",
        "severity": "mild",
        "advice": [
            "Spray a baking-soda solution (1 tbsp per gallon of water plus a drop of soap) or potassium bicarbonate.",
            "Neem oil or sulfur fungicides also work; apply weekly.",
            "Increase sunlight and airflow, and reduce nitrogen fertilizer.",
        ],
    },
    "chlorosis": {
        "label": "Chlorosis / nutrient deficiency (yellowing)",
        "severity": "mild",
        "advice": [
            "Test the soil pH. Iron chlorosis often means the soil is too alkaline.",
            "Apply a balanced fertilizer or chelated iron.",
            "Check for overwatering and poor drainage.",
        ],
    },
}

DISCLAIMER = ("Heuristic screening only, not a diagnosis. For valuable crops, confirm with a local "
              "agricultural extension service.")


def get_treatment(condition: str) -> dict[str, object]:
    """Return treatment info for a condition key (unknown conditions are treated as healthy)."""
    return TREATMENTS.get(condition, TREATMENTS["healthy"])
