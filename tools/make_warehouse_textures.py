"""Generate the textures for the warehouse props around the inspection line.

    python tools/make_warehouse_textures.py

Writes PNGs to libero/libero/assets/conveyor_line/textures/: plain cardboard, a taped carton,
yellow/black hazard stripes, a wooden pallet board, signs and a shipping label; and the
reject box's cardboard and red REJECT label next to its model (conveyor_line/reject_bin/).
Everything is drawn procedurally (no external images), so the files can be regenerated.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libero", "libero", "assets")
OUT = os.path.join(ASSETS, "conveyor_line", "textures")
REJECT_BIN_DIR = os.path.join(ASSETS, "conveyor_line", "reject_bin")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def kraft(size=512, seed=0, base=(176, 132, 86)):
    """Kraft cardboard: brown base, fibre noise and faint corrugation flutes."""
    rng = np.random.default_rng(seed)
    h = w = size
    img = np.ones((h, w, 3)) * np.array(base, float)
    blotch = np.array(Image.fromarray((rng.random((h // 16, w // 16)) * 255).astype(np.uint8))
                      .resize((w, h), Image.BICUBIC), float) / 255 - 0.5
    img += blotch[..., None] * 18
    fibres = rng.normal(0, 7, (h, w))
    img += fibres[..., None]
    flutes = np.sin(np.arange(w) / w * 2 * np.pi * 60) * 2.5
    img += flutes[None, :, None]
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(0.6))


def carton(size=512):
    """Shipping carton face: kraft, a band of packing tape and 'this side up' arrows."""
    img = kraft(size, seed=1)
    d = ImageDraw.Draw(img, "RGBA")
    t = size // 9
    d.rectangle([0, size // 2 - t // 2, size, size // 2 + t // 2], fill=(205, 170, 115, 170))
    d.line([0, size // 2 - t // 2, size, size // 2 - t // 2], fill=(150, 115, 70, 160), width=2)
    d.line([0, size // 2 + t // 2, size, size // 2 + t // 2], fill=(150, 115, 70, 160), width=2)
    ink = (40, 30, 25, 210)
    for x in (size * 0.14, size * 0.24):
        top = size * 0.12
        d.polygon([(x, top), (x - 18, top + 26), (x + 18, top + 26)], fill=ink)
        d.rectangle([x - 6, top + 26, x + 6, top + 62], fill=ink)
    d.text((size * 0.55, size * 0.70), "FRAGILE", font=_font(size // 14), fill=ink)
    d.rectangle([size * 0.55, size * 0.80, size * 0.88, size * 0.84], fill=ink)
    return img


def hazard(w=512, h=64):
    """Yellow/black diagonal safety stripes."""
    img = Image.new("RGB", (w, h), (225, 175, 20))
    d = ImageDraw.Draw(img)
    step = h
    for x0 in range(-h, w + h, 2 * step):
        d.polygon([(x0, h), (x0 + step, h), (x0 + 2 * step, 0), (x0 + step, 0)], fill=(25, 25, 25))
    return img


def wood(w=512, h=128, seed=2):
    """Rough pallet board."""
    rng = np.random.default_rng(seed)
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    grain = np.sin(y / h * np.pi * 9 + np.sin(x / w * np.pi * 3) * 2 + rng.random() * 6) * 0.5 + 0.5
    base = np.array([168, 128, 84], float)
    img = base * (0.82 + 0.18 * grain[..., None]) + rng.normal(0, 6, (h, w, 1))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))


def sign(text, sub, w=1024, h=256, bg=(20, 60, 140), fg=(255, 255, 255), border=None):
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    if border:
        d.rectangle([6, 6, w - 7, h - 7], outline=border, width=10)
    f, fs = _font(int(h * 0.36)), _font(int(h * 0.17))
    d.text(((w - d.textlength(text, font=f)) / 2, h * 0.14), text, font=f, fill=fg)
    d.text(((w - d.textlength(sub, font=fs)) / 2, h * 0.64), sub, font=fs, fill=fg)
    return img


def shipping_label(w=512, h=320, seed=3):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (240, 238, 230))
    d = ImageDraw.Draw(img)
    ink = (25, 25, 25)
    d.rectangle([4, 4, w - 5, h - 5], outline=ink, width=4)
    d.text((22, 18), "PACKING LINE 2", font=_font(34), fill=ink)
    d.text((22, 62), "CITRUS 250 ml  x 24", font=_font(26), fill=ink)
    d.text((22, 98), "GLASS - HANDLE WITH CARE", font=_font(22), fill=(170, 20, 20))
    x = 22
    while x < w - 30:
        bw = int(rng.integers(2, 7))
        d.rectangle([x, 150, x + bw, 270], fill=ink)
        x += bw + int(rng.integers(2, 6))
    d.text((22, 276), "0 41220 77310 5", font=_font(24), fill=ink)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    files = {
        "cardboard.png": kraft(),
        "carton.png": carton(),
        "hazard.png": hazard(),
        "pallet_wood.png": wood(),
        "sign_inspection.png": sign("VISION INSPECTION", "CAMERA STATION 1  -  AUTOMATIC REJECT"),
        "sign_caution.png": sign("CAUTION", "ROBOT IN OPERATION  -  KEEP OUT", bg=(235, 185, 20),
                                 fg=(20, 20, 20), border=(20, 20, 20)),
        "sign_packing.png": sign("PACKING  ZONE B", "GOOD BOTTLES  ->  CARTON", bg=(25, 110, 60)),
        "shipping_label.png": shipping_label(),
    }
    for name, img in files.items():
        img.save(os.path.join(OUT, name))
        print("wrote", os.path.join(OUT, name))
    # reject box: slightly darker cardboard and a red label (model textures sit next to its XML)
    for name, img in {"cardboard.png": kraft(seed=5, base=(160, 118, 76)),
                      "reject_label.png": sign("REJECT", "DEFECTIVE  -  DO NOT SHIP", bg=(190, 25, 20))}.items():
        img.save(os.path.join(REJECT_BIN_DIR, name))
        print("wrote", os.path.join(REJECT_BIN_DIR, name))


if __name__ == "__main__":
    main()
