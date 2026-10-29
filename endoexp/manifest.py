"""Build the evaluation manifest: link duplicate images across sources, group, split.

The same frame appears in several Kvasir releases under different names
(Kvasir-VQA, HyperKvasir labeled, HyperKvasir segmented). Records are linked
by an image fingerprint: records of the same image from different sources are
merged into one sample (labels unioned, regions kept), and all linked records
share a group so they never straddle the dev/test split.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from endoexp import sources
from endoexp._commit import code_commit
from endoexp.fields import FIELDS
from endoexp.imaging import fingerprint
from endoexp.io import read_json, read_jsonl, stable_hash, unit_hash, write_json, write_jsonl

# Lower number = preferred primary image when records are merged.
SOURCE_PRIORITY = {"kvasir_vqa": 0, "hyperkvasir": 1, "hyperkvasir_seg": 2}


@dataclass
class Sample:
    uid: str
    source: str
    image: str
    size: list[int]
    group: str
    split: str
    labels: dict[str, str]
    regions: list[dict] = field(default_factory=list)
    members: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _fingerprint(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img.draft("RGB", (256, 256))  # fast JPEG decode at reduced size; no-op for other formats
        return fingerprint(img.convert("RGB"))


def _near_pairs(prints: np.ndarray, aspects: np.ndarray, threshold: float, chunk: int = 1024):
    """Index pairs (i < j) whose fingerprints have similarity >= threshold and equal aspect ratio."""
    pairs = []
    for start in range(0, len(prints), chunk):
        similarity = prints[start:start + chunk] @ prints.T
        same_shape = np.abs(np.log(aspects[start:start + chunk, None] / aspects[None, :])) < 0.05
        for i, j in zip(*np.nonzero((similarity >= threshold) & same_shape)):
            if i + start < j:
                pairs.append((int(i + start), int(j), float(similarity[i, j])))
    return pairs


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        a, b = self.find(i), self.find(j)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _scaled_regions(record, size) -> list[dict]:
    sx, sy = size[0] / record.size[0], size[1] / record.size[1]
    return [{**r, "box": [r["box"][0] * sx, r["box"][1] * sy, r["box"][2] * sx, r["box"][3] * sy]}
            for r in record.regions]


def _merge(members: list[int], records: list, linked) -> list[dict]:
    """Merge one duplicate cluster into samples holding at most one record per source.

    A record joins a sample only if it is directly linked to that sample's
    primary record, so chains of near-duplicates are never merged.
    """
    ordered = sorted(members, key=lambda i: (SOURCE_PRIORITY[records[i].source], i))
    merged: list[dict] = []
    for i in ordered:
        source = records[i].source
        for sample in merged:
            if source not in sample["sources"] and linked(sample["primary"], i):
                sample["members"].append(i)
                sample["sources"].add(source)
                break
        else:
            merged.append({"primary": i, "members": [i], "sources": {source}})
    return merged


def build(data_root: str | Path, source_names, seed: int, threshold: float = 0.9,
          limit_per_source: int | None = None) -> tuple[list[Sample], dict]:
    data_root = Path(data_root)
    records, stats = [], {}
    for name in source_names:
        loaded, info = sources.load(name, data_root, limit=limit_per_source)
        records.extend(loaded)
        stats[name] = info
    if not records:
        raise ValueError(f"no records found under {data_root} for sources {list(source_names)}")

    prints = np.stack([_fingerprint(data_root / r.image) for r in records])
    aspects = np.array([r.size[0] / r.size[1] for r in records], dtype=np.float64)
    pairs = _near_pairs(prints, aspects, threshold)
    dist = {(i, j): d for i, j, d in pairs}

    finder = _UnionFind(len(records))
    for i, j, _ in pairs:
        finder.union(i, j)
    clusters = defaultdict(list)
    for i in range(len(records)):
        clusters[finder.find(i)].append(i)

    def linked(a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in dist

    samples, merges = [], 0
    for members in clusters.values():
        group = "g" + stable_hash(sorted(records[i].uid for i in members), 12)
        split = "dev" if unit_hash(seed, group) < 0.5 else "test"
        for merged in _merge(members, records, linked):
            primary = records[merged["primary"]]
            labels, regions, conflicts = {}, [], []
            for i in sorted(merged["members"], key=lambda k: SOURCE_PRIORITY[records[k].source]):
                record = records[i]
                for name, value in record.labels.items():
                    if name in conflicts:
                        continue
                    if labels.get(name, value) != value:
                        conflicts.append(name)
                        labels.pop(name)
                        continue
                    labels[name] = value
                regions.extend(_scaled_regions(record, primary.size))
            merges += len(merged["members"]) - 1
            samples.append(Sample(
                uid=primary.uid, source=primary.source, image=primary.image, size=list(primary.size),
                group=group, split=split, labels=labels, regions=regions,
                members=sorted(records[i].uid for i in merged["members"]),
                meta={**primary.meta, "conflicts": conflicts} if conflicts else dict(primary.meta)))
    samples.sort(key=lambda s: s.uid)

    field_stats = {}
    for name in FIELDS:
        counts = {"dev": Counter(), "test": Counter()}
        for s in samples:
            if name in s.labels:
                counts[s.split][s.labels[name]] += 1
        field_stats[name] = {split: dict(c) for split, c in counts.items()}
    header = {
        "version": 1, "code_commit": code_commit(), "seed": seed, "threshold": threshold,
        "limit_per_source": limit_per_source, "sources": stats, "records": len(records),
        "samples": len(samples), "merged_records": merges,
        "groups": len(clusters), "near_duplicate_pairs": len(pairs), "fields": field_stats,
        "hash": stable_hash([asdict(s) for s in samples]),
    }
    return samples, header


def save(samples: list[Sample], header: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    write_jsonl(out_dir / "manifest.jsonl.gz", (asdict(s) for s in samples))
    write_json(out_dir / "manifest.json", header)


def load(out_dir: str | Path) -> tuple[list[Sample], dict]:
    out_dir = Path(out_dir)
    samples = [Sample(**row) for row in read_jsonl(out_dir / "manifest.jsonl.gz")]
    return samples, read_json(out_dir / "manifest.json")
