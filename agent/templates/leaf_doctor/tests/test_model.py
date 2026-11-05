"""Model tests on synthetic leaves where the right answer is known."""
from PIL import Image, ImageDraw

from model import CLASSES, predict


def leaf(spot_color=None, base=(40, 140, 40), coverage=0.5):
    img = Image.new("RGB", (128, 128), (128, 128, 128))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 120, 120), fill=base)
    if spot_color:
        r = int(56 * coverage ** 0.5)
        d.ellipse((64 - r, 64 - r, 64 + r, 64 + r), fill=spot_color)
    return img


def test_healthy():
    assert predict(leaf()).condition == "healthy"


def test_early_blight_brown_spots():
    assert predict(leaf((110, 60, 20))).condition == "early_blight"


def test_rust_orange_spots():
    assert predict(leaf((230, 120, 20))).condition == "rust"


def test_powdery_mildew_white():
    assert predict(leaf((245, 245, 245))).condition == "powdery_mildew"


def test_chlorosis_yellow():
    assert predict(leaf(base=(220, 210, 40))).condition == "chlorosis"


def test_output_in_classes():
    p = predict(leaf((110, 60, 20), coverage=0.2))
    assert p.condition in CLASSES and 0 <= p.confidence <= 1
