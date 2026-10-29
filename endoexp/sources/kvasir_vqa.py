"""Kvasir-VQA (SimulaMet-HOST/Kvasir-VQA, CC BY-NC 4.0).

Prepared layout (see ``tools/prepare_data.py``)::

    kvasir_vqa/qa.jsonl        one row per question: img_id, category, question, answer, image
    kvasir_vqa/images/<img_id>.<ext>

Each image carries about nine template questions. They are converted to
closed-set fields; answers that do not map to exactly one option are skipped
(for example two abnormalities in one image) and counted in the stats.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from endoexp.sources import Record
from endoexp.io import read_jsonl

ROOT = "kvasir_vqa"


def question_key(text: str) -> str:
    text = re.sub(r"[^a-z0-9/ ]", " ", text.lower())
    return " ".join(text.split())


def _single(mapping: dict[str, str]):
    def parse(answer: str) -> str | None:
        values = [v.strip() for v in answer.lower().split(";") if v.strip()]
        if len(values) != 1:
            return None
        return mapping.get(values[0])
    return parse


def _count(cap: int):
    def parse(answer: str) -> str | None:
        answer = answer.strip()
        if not answer.isdigit():
            return None
        n = int(answer)
        if n >= cap:
            return "3 or more" if cap == 3 else "2 or more"
        return str(n)
    return parse


_YES_NO = _single({"yes": "yes", "no": "no"})

# question key -> (field, parser)
QUESTIONS = {
    "what type of procedure is the image taken from": (
        "procedure", _single({"colonoscopy": "colonoscopy", "gastroscopy": "gastroscopy"})),
    "are there any anatomical landmarks in the image check all that are present": (
        "landmark", _single({"none": "none", "cecum": "cecum", "ileum": "ileum",
                             "pylorus": "pylorus", "z-line": "z-line"})),
    "are there any abnormalities in the image check all that are present": (
        "abnormality", _single({"none": "none", "polyp": "polyp", "oesophagitis": "esophagitis",
                                "esophagitis": "esophagitis", "ulcerative colitis": "ulcerative colitis",
                                "barretts": "Barrett's esophagus",
                                "short-segment barretts": "short-segment Barrett's esophagus",
                                "hemorrhoids": "hemorrhoids"})),
    "are there any instruments in the image check all that are present": (
        "instrument", _single({"none": "none", "tube": "tube", "metal clip": "metal clip",
                               "polyp snare": "polyp snare", "injection needle": "injection needle",
                               "biopsy forceps": "biopsy forceps"})),
    "is there text": ("text_overlay", _YES_NO),
    "is there a green/black box artefact": ("box_overlay", _YES_NO),
    "how many polyps are in the image": ("polyp_count", _count(3)),
    "how many instrumnets are in the image": ("instrument_count", _count(2)),  # sic, as released
    "how many instruments are in the image": ("instrument_count", _count(2)),
    "what is the size of the polyp": (
        "polyp_size", _single({"<5mm": "smaller than 5 mm", "5-10mm": "5 to 10 mm",
                               "11-20mm": "11 to 20 mm", ">20mm": "larger than 20 mm"})),
    "what type of polyp is present": (
        "paris_type", _single({"paris ip": "0-Ip (pedunculated)", "paris is": "0-Is (sessile)",
                               "paris iia": "0-IIa (slightly elevated)"})),
}


def load(data_root: Path, limit: int | None = None) -> tuple[list[Record], dict]:
    root = data_root / ROOT
    rows_by_image: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(root / "qa.jsonl"):
        rows_by_image[row["img_id"]].append(row)
    image_ids = sorted(rows_by_image)
    if limit is not None:
        image_ids = image_ids[:limit]

    skipped, unknown_questions, conflicts = Counter(), Counter(), 0
    records = []
    for img_id in image_ids:
        rows = rows_by_image[img_id]
        labels: dict[str, str] = {}
        for row in rows:
            entry = QUESTIONS.get(question_key(row["question"]))
            if entry is None:
                unknown_questions[question_key(row["question"])] += 1
                continue
            field, parse = entry
            value = parse(str(row["answer"]))
            if value is None:
                skipped[field] += 1
                continue
            if labels.get(field, value) != value:
                conflicts += 1
                labels.pop(field)
                continue
            labels[field] = value
        image = f"{ROOT}/{rows[0]['image']}"
        with Image.open(data_root / image) as img:
            size = img.size
        records.append(Record(source="kvasir_vqa", key=img_id, image=image, size=size, labels=labels,
                              meta={"category": rows[0].get("category", "")}))
    stats = {"images": len(records), "skipped_answers": dict(skipped),
             "unmapped_questions": dict(unknown_questions), "conflicts": conflicts}
    return records, stats
