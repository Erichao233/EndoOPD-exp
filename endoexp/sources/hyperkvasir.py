"""HyperKvasir (SimulaMet-HOST/HyperKvasir, CC BY 4.0).

Prepared layout: the official zips extracted under ``hyperkvasir/``. Labeled
images live in one folder per class (23 classes, folder name = class); the
segmented set has ``images/`` and ``masks/`` folders with matching file names.
Both are found by searching, so an extra top-level folder from unzipping is fine.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from endoexp.fields import extent_bin, grid_cell
from endoexp.imaging import mask_regions
from endoexp.sources import IMAGE_SUFFIXES, Record

ROOT = "hyperkvasir"

_LOWER = {"procedure": "colonoscopy"}
_UPPER = {"procedure": "gastroscopy"}
_UC = {**_LOWER, "abnormality": "ulcerative colitis"}

CLASS_LABELS: dict[str, dict[str, str]] = {
    "normal-cecum": {**_LOWER, "landmark": "cecum", "abnormality": "none"},
    "ileum": {**_LOWER, "landmark": "ileum"},
    "retroflex-rectum": {**_LOWER, "landmark": "retroflexed rectum"},
    "normal-pylorus": {**_UPPER, "landmark": "pylorus", "abnormality": "none"},
    "normal-z-line": {**_UPPER, "landmark": "z-line", "abnormality": "none"},
    "retroflex-stomach": {**_UPPER, "landmark": "retroflexed stomach"},
    "polyp": {**_LOWER, "abnormality": "polyp"},
    "hemorrhoids": {**_LOWER, "abnormality": "hemorrhoids"},
    "oesophagitis-a": {**_UPPER, "abnormality": "esophagitis", "esophagitis_grade": "grade A"},
    "oesophagitis-b-d": {**_UPPER, "abnormality": "esophagitis", "esophagitis_grade": "grade B to D"},
    "barretts": {**_UPPER, "abnormality": "Barrett's esophagus"},
    "short-segment-barretts": {**_UPPER, "abnormality": "short-segment Barrett's esophagus"},
    "ulcerative-colitis-grade-1": {**_UC, "uc_grade": "grade 1 (mild)"},
    "ulcerative-colitis-grade-2": {**_UC, "uc_grade": "grade 2 (moderate)"},
    "ulcerative-colitis-grade-3": {**_UC, "uc_grade": "grade 3 (severe)"},
    # Intermediate grades are ambiguous, so only the finding is used.
    "ulcerative-colitis-0-1": dict(_UC),
    "ulcerative-colitis-1-2": dict(_UC),
    "ulcerative-colitis-2-3": dict(_UC),
    "bbps-0-1": {**_LOWER, "bbps": "0-1 (inadequate)"},
    "bbps-2-3": {**_LOWER, "bbps": "2-3 (adequate)"},
    "impacted-stool": dict(_LOWER),
    "dyed-lifted-polyps": dict(_LOWER),
    "dyed-resection-margins": dict(_LOWER),
}

_EXCLUDED_PARTS = {"segmented-images", "unlabeled-images", "unlabeled", "videos"}


def _images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_labeled(data_root: Path, limit: int | None = None) -> tuple[list[Record], dict]:
    root = data_root / ROOT
    class_dirs = sorted(
        p for p in root.rglob("*")
        if p.is_dir() and p.name in CLASS_LABELS and not (_EXCLUDED_PARTS & set(p.parts)))
    per_class = Counter()
    records = []
    for folder in class_dirs:
        images = _images(folder)
        if limit is not None:
            images = images[:max(1, limit // len(CLASS_LABELS))]
        for path in images:
            with Image.open(path) as img:
                size = img.size
            records.append(Record(
                source="hyperkvasir", key=f"{folder.name}/{path.stem}",
                image=str(path.relative_to(data_root)), size=size,
                labels=dict(CLASS_LABELS[folder.name]), meta={"class": folder.name}))
            per_class[folder.name] += 1
    return records, {"images": len(records), "per_class": dict(per_class),
                     "class_folders": len(class_dirs)}


def load_segmented(data_root: Path, limit: int | None = None) -> tuple[list[Record], dict]:
    root = data_root / ROOT
    mask_dirs = sorted(p for p in root.rglob("masks") if p.is_dir() and (p.parent / "images").is_dir())
    records, missing = [], 0
    for mask_dir in mask_dirs:
        masks = {p.stem: p for p in _images(mask_dir)}
        images = _images(mask_dir.parent / "images")
        if limit is not None:
            images = images[:limit]
        for path in images:
            mask_path = masks.get(path.stem)
            if mask_path is None:
                missing += 1
                continue
            with Image.open(path) as img:
                size = img.size
            with Image.open(mask_path) as mask:
                mask_size = mask.size
                regions = mask_regions(mask)
            if not regions:
                missing += 1
                continue
            scale_x, scale_y = size[0] / mask_size[0], size[1] / mask_size[1]
            mask_rel = str(mask_path.relative_to(data_root))
            labels = {
                "procedure": "colonoscopy",
                "abnormality": "polyp",
                "polyp_location": grid_cell(*regions[0]["center"]),
                "polyp_extent": extent_bin(sum(r["fraction"] for r in regions)),
            }
            records.append(Record(
                source="hyperkvasir_seg", key=path.stem, image=str(path.relative_to(data_root)),
                size=size, labels=labels,
                regions=[{"label": "polyp", "mask": mask_rel,
                          "box": [r["box"][0] * scale_x, r["box"][1] * scale_y,
                                  r["box"][2] * scale_x, r["box"][3] * scale_y]}
                         for r in regions],
                meta={"regions": len(regions)}))
    return records, {"images": len(records), "missing_or_empty_masks": missing,
                     "mask_folders": len(mask_dirs)}
