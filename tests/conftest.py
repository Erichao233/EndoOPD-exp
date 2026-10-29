"""Synthetic data root that mimics the prepared Kvasir-VQA and HyperKvasir layouts.

The same synthetic frame is written three times (Kvasir-VQA, HyperKvasir
labeled, HyperKvasir segmented) at different sizes, as in the real releases.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Q = {
    "abnormality": "Are there any abnormalities in the image? Check all that are present.",
    "landmark": "Are there any anatomical landmarks in the image? Check all that are present.",
    "instrument": "Are there any instruments in the image? Check all that are present.",
    "text": "Is there text?",
    "box": "Is there a green/black box artefact?",
    "paris": "What type of polyp is present?",
    "procedure": "What type of procedure is the image taken from?",
    "size": "What is the size of the polyp?",
    "polyps": "How many polyps are in the image?",
    "instruments": "How many instrumnets are in the image?",
    "removed": "Have all polyps been removed?",
}


def frame(seed: int, size=(200, 160), polyp=None, green=False, text=False) -> Image.Image:
    """An endoscopy-like frame: pink mucosa in an octagon on black, optional polyp/inset/text."""
    rng = np.random.default_rng(seed)
    w, h = size
    base = np.array([200, 110, 100]) + rng.integers(-40, 40, 3)
    arr = np.clip(base + rng.normal(0, 12, (h, w, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for _ in range(6):  # mucosal folds: make every frame's coarse layout distinct
        x, y, rx, ry = rng.integers(0, w), rng.integers(0, h), rng.integers(10, 50), rng.integers(10, 40)
        shade = tuple(int(v) for v in np.clip(base + int(rng.integers(-70, 50)), 40, 255))
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=shade)
    cut = int(0.12 * min(w, h))
    for corner in ([(0, 0), (cut, 0), (0, cut)], [(w, 0), (w - cut, 0), (w, cut)],
                   [(0, h), (cut, h), (0, h - cut)], [(w, h), (w - cut, h), (w, h - cut)]):
        draw.polygon(corner, fill=(0, 0, 0))
    draw.rectangle([0, 0, w - 1, 15], fill=(0, 0, 0))   # black band for burned-in text
    if polyp is not None:
        cx, cy, r = polyp
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(150, 40, 50))
    if green:
        draw.rectangle([4, h - 45, 50, h - 4], fill=(0, 0, 0))
        draw.line([(8, h - 10), (25, h - 35), (45, h - 20)], fill=(30, 230, 60), width=3)
    if text:
        draw.text((24, 2), "12:00", fill=(255, 255, 255))
    return img


def save(img: Image.Image, path: Path, size=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (img.resize(size, Image.BILINEAR) if size else img).save(path, quality=95)


def make_data_root(root: Path) -> Path:
    rows = []

    def qa(img_id, category, answers):
        for key, answer in answers.items():
            rows.append({"img_id": img_id, "category": category, "question": Q[key], "answer": answer,
                         "image": f"images/{img_id}.jpg"})

    sizes = ["<5mm", "5-10mm", "11-20mm", ">20mm"]
    paris = ["paris is", "paris ip", "paris iia"]
    for i in range(12):  # polyps: also in HyperKvasir labeled and segmented
        polyp = (40 + 10 * (i % 12), 40 + 8 * (i % 9), 12 + 2 * (i % 4))
        img = frame(i, polyp=polyp, green=i % 2 == 0, text=i % 3 == 0)
        img_id = f"clpolyp{i:02d}"
        save(img, root / "kvasir_vqa" / "images" / f"{img_id}.jpg")
        qa(img_id, "Polyps", {"abnormality": "polyp", "landmark": "none", "instrument": "none",
                              "text": "yes" if i % 3 == 0 else "no", "box": "yes" if i % 2 == 0 else "no",
                              "paris": paris[i % 3], "procedure": "colonoscopy", "size": sizes[i % 4],
                              "polyps": "1", "instruments": "0", "removed": "no"})
        save(img, root / "hyperkvasir" / "labeled-images" / "lower-gi-tract" / "pathological-findings"
             / "polyp" / f"hk-polyp-{i:02d}.jpg", size=(180, 144))
        seg = root / "hyperkvasir" / "segmented-images"
        save(img, seg / "images" / f"seg{i:02d}.jpg", size=(220, 176))
        mask = Image.new("L", (200, 160), 0)
        cx, cy, r = polyp
        ImageDraw.Draw(mask).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        save(mask, seg / "masks" / f"seg{i:02d}.jpg", size=(220, 176))
    landmarks = ["cecum", "pylorus", "z-line", "none"]
    for i in range(12):  # normal images with landmarks
        img_id = f"clnormal{i:02d}"
        img = frame(100 + i, green=i % 2 == 1, text=True)
        save(img, root / "kvasir_vqa" / "images" / f"{img_id}.jpg")
        lm = landmarks[i % 4]
        qa(img_id, "Normal", {"abnormality": "none", "landmark": lm, "instrument": "none", "text": "yes",
                              "box": "yes" if i % 2 == 1 else "no", "paris": "none",
                              "procedure": "gastroscopy" if lm in ("pylorus", "z-line") else "colonoscopy",
                              "size": "none", "polyps": "0", "instruments": "0", "removed": "not relevant"})
    for i in range(6):  # an ambiguous answer that must be skipped
        img_id = f"cluc{i:02d}"
        save(frame(200 + i), root / "kvasir_vqa" / "images" / f"{img_id}.jpg")
        qa(img_id, "Ulcerative Colitis", {"abnormality": "ulcerative colitis; polyp" if i == 0 else "ulcerative colitis",
                                          "landmark": "none", "procedure": "colonoscopy", "text": "no", "box": "no"})
    (root / "kvasir_vqa").mkdir(exist_ok=True)
    with open(root / "kvasir_vqa" / "qa.jsonl", "w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)

    labeled = root / "hyperkvasir" / "labeled-images"
    classes = {"bbps-0-1": "lower-gi-tract/quality-of-mucosal-views",
               "bbps-2-3": "lower-gi-tract/quality-of-mucosal-views",
               "ulcerative-colitis-grade-1": "lower-gi-tract/pathological-findings",
               "ulcerative-colitis-grade-3": "lower-gi-tract/pathological-findings",
               "oesophagitis-a": "upper-gi-tract/pathological-findings",
               "oesophagitis-b-d": "upper-gi-tract/pathological-findings",
               "normal-z-line": "upper-gi-tract/anatomical-landmarks"}
    for c, (name, folder) in enumerate(classes.items()):
        for i in range(6):
            save(frame(300 + 10 * c + i), labeled / folder / name / f"{name}-{i}.jpg")
    return root


@pytest.fixture(scope="session")
def data_root(tmp_path_factory) -> Path:
    return make_data_root(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
