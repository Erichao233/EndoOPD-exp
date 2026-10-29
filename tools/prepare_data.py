"""Download the open datasets and lay them out for the adapters. Run outside the sandbox.

    python tools/prepare_data.py --out <data-root> [--sources kvasir_vqa hyperkvasir]

Layout written under <data-root>:

    kvasir_vqa/qa.jsonl, kvasir_vqa/images/<img_id>.<ext>   (SimulaMet-HOST/Kvasir-VQA, CC BY-NC 4.0)
    hyperkvasir/<labeled and segmented zips, extracted>      (SimulaMet-HOST/HyperKvasir, CC BY 4.0)
    INDEX.json   revisions plus a hash over every file; its sha256 is the dataset revision
                 to quote in the hub manifest.

Downloads are cached in <data-root>/_downloads and can be deleted afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endoexp.io import sha256_file, write_jsonl  # noqa: E402

KVASIR_VQA = "SimulaMet-HOST/Kvasir-VQA"
HYPERKVASIR = "SimulaMet-HOST/HyperKvasir"
HYPERKVASIR_FILES = ("hyper-kvasir-labeled-images.zip", "hyper-kvasir-segmented-images.zip")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _dataset_revision(repo: str, revision: str | None) -> str:
    from huggingface_hub import HfApi
    return revision or HfApi().dataset_info(repo).sha


def prepare_kvasir_vqa(out: Path, revision: str | None) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    revision = _dataset_revision(KVASIR_VQA, revision)
    local = Path(snapshot_download(KVASIR_VQA, repo_type="dataset", revision=revision,
                                   allow_patterns=["data/*.parquet", "README.md"],
                                   local_dir=out / "_downloads" / "kvasir_vqa"))
    root = out / "kvasir_vqa"
    (root / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for parquet in sorted((local / "data").glob("*.parquet")):
        for record in pq.read_table(parquet).to_pylist():
            data = record["image"]["bytes"]
            name = f"{record['img_id']}{'.png' if data[:8] == PNG_MAGIC else '.jpg'}"
            path = root / "images" / name
            if not path.exists():
                path.write_bytes(data)
            rows.append({"img_id": record["img_id"], "category": record["source"],
                         "question": record["question"], "answer": record["answer"], "image": f"images/{name}"})
    write_jsonl(root / "qa.jsonl", rows)
    return {"repo": KVASIR_VQA, "revision": revision, "rows": len(rows)}


def _safe_extract(archive: Path, target: Path) -> int:
    target = target.resolve()
    count = 0
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            destination = (target / member.filename).resolve()
            if target not in destination.parents and destination != target:
                raise ValueError(f"unsafe path in {archive.name}: {member.filename}")
            if member.filename.startswith("__MACOSX/"):
                continue
            handle.extract(member, target)
            count += not member.is_dir()
    return count


def prepare_hyperkvasir(out: Path, revision: str | None) -> dict:
    from huggingface_hub import hf_hub_download

    revision = _dataset_revision(HYPERKVASIR, revision)
    files = {}
    for name in HYPERKVASIR_FILES:
        archive = Path(hf_hub_download(HYPERKVASIR, name, repo_type="dataset", revision=revision,
                                       local_dir=out / "_downloads" / "hyperkvasir"))
        files[name] = _safe_extract(archive, out / "hyperkvasir")
    return {"repo": HYPERKVASIR, "revision": revision, "files": files}


def write_index(out: Path, sources: dict) -> str:
    """Hash every prepared file (names, sizes, contents) into one tree hash."""
    digest = hashlib.sha256()
    count = total = 0
    for folder in ("kvasir_vqa", "hyperkvasir"):
        root = out / folder
        if not root.is_dir():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(out).as_posix()
            size = path.stat().st_size
            digest.update(f"{rel}\t{size}\t{sha256_file(path)}\n".encode())
            count += 1
            total += size
    index = {"sources": sources, "files": count, "bytes": total, "tree_sha256": digest.hexdigest()}
    (out / "INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return sha256_file(out / "INDEX.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="data root")
    parser.add_argument("--sources", nargs="*", default=["kvasir_vqa", "hyperkvasir"])
    parser.add_argument("--kvasir-vqa-revision")
    parser.add_argument("--hyperkvasir-revision")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "INDEX.json"
    sources = json.loads(index_path.read_text())["sources"] if index_path.exists() else {}
    if "kvasir_vqa" in args.sources:
        sources["kvasir_vqa"] = prepare_kvasir_vqa(out, args.kvasir_vqa_revision)
        print("kvasir_vqa:", sources["kvasir_vqa"], flush=True)
    if "hyperkvasir" in args.sources:
        sources["hyperkvasir"] = prepare_hyperkvasir(out, args.hyperkvasir_revision)
        print("hyperkvasir:", sources["hyperkvasir"], flush=True)
    print("dataset revision (sha256 of INDEX.json):", write_index(out, sources))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
