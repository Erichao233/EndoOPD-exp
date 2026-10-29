"""One task in its own process: a model over one shard of its experiments, or a check.

Run as ``python -m endoexp.task --spec <task.json>``. Writes ``done.json`` on
success and ``error.json`` on failure into the task's output folder.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from endoexp import checks, engines, manifest
from endoexp.experiments import FEATURES, GENERATED, SCORED, Planner, QueryBuilder, shard_items
from endoexp.fields import FIELDS
from endoexp.io import read_json, write_json, write_jsonl
from endoexp.prompts import parse_report


def _log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _scored_row(spec: dict, item, result: dict) -> dict:
    options = FIELDS[item.field].options
    probs = result["probs"]
    best = max(range(len(probs)), key=probs.__getitem__)
    return {
        "exp": item.exp, "condition": item.condition, "view": item.view,
        "model": spec["model"], "engine": spec["engine"], "uid": item.uid, "field": item.field,
        "label": item.label, "source": item.source, "group": item.group,
        "probs": [round(float(p), 6) for p in probs], "pred": options[best],
        "coverage": round(float(result.get("coverage", 1.0)), 6), "top_token": result.get("top_token"),
    }


def _report_row(spec: dict, item, text: str) -> dict:
    return {"exp": item.exp, "condition": item.condition, "model": spec["model"], "engine": spec["engine"],
            "uid": item.uid, "source": item.source, "group": item.group, "text": text,
            "parsed": parse_report(text)}


def _save_features(path: Path, model: str, features: dict[str, dict[str, np.ndarray]]) -> None:
    uids = sorted(features)
    names = sorted(set.intersection(*(set(v) for v in features.values())))
    arrays = {f"feat_{name}": np.stack([features[uid][name] for uid in uids]) for name in names}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, uids=np.array(uids), model=np.array(model), **arrays)


def run_model(spec: dict) -> dict:
    samples, header = manifest.load(spec["manifest_dir"])
    planner = Planner(samples, spec.get("caps"), spec["seed"], spec.get("pilot_items"))
    builder = QueryBuilder(spec["data_root"], spec["max_pixels"], spec.get("report_tokens", 256))
    out = Path(spec["out_dir"])
    rows, features, counts = [], {}, {}
    engine = None
    try:
        for exp in spec["experiments"]:
            items = shard_items(planner.plan(exp), spec["shard"], spec["shards"])
            counts[exp] = len(items)
            if not items:
                continue
            if engine is None:
                _log(f"loading {spec['engine']} engine: {spec['model']}")
                engine = engines.create(spec["engine"], spec["model_path"], **spec.get("engine_args", {}))
            # Neighbouring queries share an image, so its prefill is reused from the prefix cache.
            items.sort(key=lambda i: (i.image or "", i.view, i.condition, i.field or ""))
            chunk = spec.get("chunk", 512)
            for start in range(0, len(items), chunk):
                batch = items[start:start + chunk]
                queries = [builder.build(item) for item in batch]
                if exp in SCORED:
                    results = engine.score(queries)
                    rows.extend(_scored_row(spec, item, results[item.key]) for item in batch)
                elif exp in GENERATED:
                    texts = engine.generate(queries)
                    rows.extend(_report_row(spec, item, texts[item.key]) for item in batch)
                elif exp in FEATURES:
                    vectors = engine.features(queries)
                    features.update((item.uid, vectors[item.key]) for item in batch)
                _log(f"{exp}: {min(start + chunk, len(items))}/{len(items)}")
    finally:
        if engine is not None:
            engine.close()
    write_jsonl(out / "rows.jsonl.gz", rows)
    if features:
        _save_features(out / "features.npz", spec["model"], features)
    return {"counts": counts, "rows": len(rows), "features": len(features), "manifest_hash": header["hash"]}


def run(spec: dict) -> int:
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        kind = spec["kind"]
        if kind == "model":
            info = run_model(spec)
        elif kind == "tokenizers":
            info = checks.tokenizers(spec)
        elif kind == "nccl":
            info = checks.nccl(spec)
        else:
            raise ValueError(f"unknown task kind {kind!r}")
        write_json(out / "done.json", {"seconds": round(time.time() - started, 1), **info})
        return 0
    except BaseException:
        error = traceback.format_exc()
        _log(error)
        write_json(out / "error.json", {"seconds": round(time.time() - started, 1), "error": error})
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    return run(read_json(args.spec))


if __name__ == "__main__":
    raise SystemExit(main())
