"""Experiment plans: which (sample, field, condition) items each experiment asks about.

Plans depend only on the manifest, the caps and the seed, so every model gets
exactly the same items and comparisons between models are paired.

- ``e1``  who has the eye: each field under the real image, no image, and the
  image of another sample with the same field (``normal``/``blind``/``shuffled``);
- ``e2``  privileged views on images with a marked polyp (box, contour, crop);
- ``e4``  shortcut manipulations: clean overlays, insert a scope inset, write "POLYP";
- ``report``  free generation of the structured findings template;
- ``features``  hidden states under a neutral prompt, for linear probes (E3);
- ``parity``  a small subset of e1 scored by the HF engine to check vLLM.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image

from endoexp.engines import Query
from endoexp.fields import FIELDS, REPORT_FIELDS
from endoexp.imaging import fit, load_rgb
from endoexp.io import unit_hash
from endoexp.prompts import DESCRIBE, mcq, report_template
from endoexp.views import REGION_VIEWS, VIEW_HINTS, VIEWS

SCORED = ("e1", "e2", "e4", "parity")
GENERATED = ("report",)
FEATURES = ("features",)
EXPERIMENTS = SCORED + GENERATED + FEATURES

E2_FIELDS = ("polyp_location", "polyp_extent", "polyp_count", "polyp_size", "paris_type")
E2_VIEWS = ("plain", "box", "contour", "crop")
CROP_EXCLUDED = {"polyp_location", "polyp_extent"}   # a crop re-centres the image

E4_PLAN = {
    "clean": (lambda s: "text_overlay" in s.labels or "box_overlay" in s.labels,
              ("procedure", "text_overlay", "box_overlay", "abnormality", "landmark")),
    "insert_inset": (lambda s: s.labels.get("box_overlay") == "no", ("box_overlay", "procedure")),
    "insert_text_polyp": (lambda s: s.labels.get("abnormality") == "none", ("abnormality",)),
}

DEFAULT_CAPS = {
    "e1_per_source_field": 1000,
    "e2_samples": 400,
    "e4_per_view": 300,
    "report_samples": 300,
    "feature_samples": 3000,
    "parity_items": 64,
}


@dataclass
class Item:
    exp: str
    condition: str
    uid: str
    field: str | None
    label: str | None
    source: str
    group: str
    split: str
    image: str | None                  # image shown, relative to the data root; None = no image
    view: str = "plain"
    boxes: list = field(default_factory=list)
    masks: list = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.exp}|{self.condition}|{self.uid}|{self.field or '-'}"


class Planner:
    def __init__(self, samples, caps: dict | None = None, seed: int = 0, pilot_items: int | None = None):
        self.samples = sorted(samples, key=lambda s: s.uid)
        self.caps = {**DEFAULT_CAPS, **(caps or {})}
        self.seed = seed
        self.pilot_items = pilot_items
        self.test = [s for s in self.samples if s.split == "test"]

    def _rank(self, *parts) -> float:
        return unit_hash(self.seed, *parts)

    def _base(self, exp: str, sample, name: str | None) -> dict:
        return dict(exp=exp, uid=sample.uid, field=name,
                    label=sample.labels.get(name) if name else None,
                    source=sample.source, group=sample.group, split=sample.split)

    def _pilot(self, items: list[Item]) -> list[Item]:
        """In pilot mode keep all conditions of the first few (sample, field) pairs."""
        if self.pilot_items is None:
            return items
        pairs = sorted({(i.uid, i.field or "") for i in items}, key=lambda p: self._rank("pilot", *p))
        keep = set(pairs[:self.pilot_items])
        return [i for i in items if (i.uid, i.field or "") in keep]

    @staticmethod
    def _unique(items: list[Item]) -> list[Item]:
        return list(OrderedDict((i.key, i) for i in items).values())

    def plan(self, exp: str) -> list[Item]:
        if exp not in EXPERIMENTS:
            raise KeyError(f"unknown experiment {exp!r}; known: {EXPERIMENTS}")
        return getattr(self, exp)()

    def e1(self) -> list[Item]:
        populations = defaultdict(list)
        for sample in self.test:
            for name in sample.labels:
                if name in FIELDS:
                    populations[(sample.source, name)].append(sample)
        items = []
        for (_, name), members in sorted(populations.items()):
            members = sorted(members, key=lambda s: self._rank("e1", s.uid, name))
            members = members[:self.caps["e1_per_source_field"]]
            for index, sample in enumerate(members):
                base = self._base("e1", sample, name)
                items.append(Item(condition="normal", image=sample.image, **base))
                items.append(Item(condition="blind", image=None, **base))
                # Another image asked the same question: the next member in the shuffled order.
                for step in range(1, len(members)):
                    partner = members[(index + step) % len(members)]
                    if partner.image != sample.image:
                        items.append(Item(condition="shuffled", image=partner.image, **base))
                        break
        return self._pilot(items)

    def e2(self) -> list[Item]:
        candidates = [s for s in self.test if s.regions and any(f in s.labels for f in E2_FIELDS)]
        candidates = sorted(candidates, key=lambda s: self._rank("e2", s.uid))[:self.caps["e2_samples"]]
        items = []
        for sample in candidates:
            boxes = [r["box"] for r in sample.regions]
            masks = [r["mask"] for r in sample.regions if r.get("mask")]
            for name in E2_FIELDS:
                if name not in sample.labels:
                    continue
                for view in E2_VIEWS:
                    if view == "crop" and name in CROP_EXCLUDED:
                        continue
                    items.append(Item(condition=view, image=sample.image, view=view, boxes=boxes,
                                      masks=masks, **self._base("e2", sample, name)))
        return self._pilot(items)

    def e4(self) -> list[Item]:
        items = []
        for view, (eligible, names) in E4_PLAN.items():
            candidates = sorted((s for s in self.test if eligible(s)),
                                key=lambda s: self._rank("e4", view, s.uid))[:self.caps["e4_per_view"]]
            for sample in candidates:
                for name in names:
                    if name not in sample.labels:
                        continue
                    base = self._base("e4", sample, name)
                    items.append(Item(condition="plain", image=sample.image, **base))
                    items.append(Item(condition=view, image=sample.image, view=view, **base))
        return self._pilot(self._unique(items))

    def report(self) -> list[Item]:
        candidates = [s for s in self.test if sum(f in s.labels for f in REPORT_FIELDS) >= 3]
        candidates = sorted(candidates, key=lambda s: self._rank("report", s.uid))[:self.caps["report_samples"]]
        return self._pilot([Item(condition="normal", image=s.image, **self._base("report", s, None))
                            for s in candidates])

    def features(self) -> list[Item]:
        half = self.caps["feature_samples"] // 2
        e1_uids = {i.uid for i in self.e1() if i.condition == "normal"}
        dev = [s for s in self.samples if s.split == "dev" and s.labels]
        test = [s for s in self.test if s.uid in e1_uids]
        chosen = (sorted(dev, key=lambda s: self._rank("features", s.uid))[:half]
                  + sorted(test, key=lambda s: self._rank("features", s.uid))[:half])
        items = [Item(condition="describe", image=s.image, **self._base("features", s, None)) for s in chosen]
        if self.pilot_items is not None:
            per_split = max(1, self.pilot_items // 2)
            items = ([i for i in items if i.split == "dev"][:per_split]
                     + [i for i in items if i.split == "test"][:per_split])
        return items

    def parity(self) -> list[Item]:
        normal = [i for i in self.e1() if i.condition == "normal"]
        normal = sorted(normal, key=lambda i: self._rank("parity", i.uid, i.field))[:self.caps["parity_items"]]
        return [replace(i, exp="parity") for i in normal]


def shard_items(items: list[Item], index: int, count: int) -> list[Item]:
    """Assign whole samples to shards so all questions about an image stay together."""
    if count <= 1:
        return list(items)
    return [i for i in items if int(unit_hash("shard", i.uid) * count) == index]


class QueryBuilder:
    def __init__(self, data_root: str | Path, max_pixels: int, report_tokens: int = 256, cache_size: int = 64):
        self.data_root = Path(data_root)
        self.max_pixels = max_pixels
        self.report_tokens = report_tokens
        self.cache_size = cache_size
        self._images: OrderedDict[str, Image.Image] = OrderedDict()

    def _load(self, rel: str, mode: str = "RGB") -> Image.Image:
        key = f"{mode}:{rel}"
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        path = self.data_root / rel
        if mode == "RGB":
            img = load_rgb(path)
        else:
            with Image.open(path) as handle:
                img = handle.convert(mode)
        self._images[key] = img
        if len(self._images) > self.cache_size:
            self._images.popitem(last=False)
        return img

    def image(self, item: Item) -> Image.Image | None:
        if item.image is None:
            return None
        img = self._load(item.image)
        if item.view != "plain":
            if item.view in REGION_VIEWS and not item.boxes:
                raise ValueError(f"view {item.view!r} needs regions: {item.key}")
            masks = [self._load(m, "L") for m in item.masks]
            img = VIEWS[item.view](img, item.boxes, masks)
        return fit(img, self.max_pixels, upscale=item.view == "crop")

    def build(self, item: Item) -> Query:
        img = self.image(item)
        images = [img] if img is not None else []
        if item.exp in GENERATED:
            return Query(item.key, report_template(), images, max_tokens=self.report_tokens)
        if item.exp in FEATURES:
            return Query(item.key, DESCRIBE, images)
        spec = FIELDS[item.field]
        text = mcq(spec, hint=VIEW_HINTS.get(item.view), blind=not images)
        return Query(item.key, text, images, n_options=len(spec.options))
