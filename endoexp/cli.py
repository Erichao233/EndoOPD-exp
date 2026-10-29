"""Command line: ``python -m endoexp {run,build,analyze,hub-manifest}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = "configs/models.toml"
DELIVERABLES = ["results/digest.md", "results/report.md", "results/summary.json", "results/items.jsonl.gz",
                "results/env.json", "logs/suite.log"]
REPOSITORY = "https://github.com/Erichao233/EndoOPD-exp"


def _resolve(path: str) -> Path:
    """Config paths are relative to the working directory, falling back to the repository."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() or candidate.exists() else REPO_ROOT / candidate


def _pairs(values: list[str], parse_values: bool) -> dict:
    """KEY=VALUE options; values are JSON literals when they parse (numbers, booleans)."""
    result = {}
    for item in values or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        if parse_values:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        result[key] = value
    return result


def _run(args) -> int:
    from endoexp import suite
    return suite.run(_resolve(args.suite), _resolve(args.models), args.data_root, args.models_root,
                     args.out, pilot=args.pilot, engine=args.engine, gpus=args.gpus, only=args.only,
                     vllm_overrides=_pairs(args.vllm_arg, True), extra_env=_pairs(args.set_env, False))


def _build(args) -> int:
    from endoexp import manifest
    from endoexp.config import load_suite
    spec = load_suite(_resolve(args.suite))
    samples, header = manifest.build(args.data_root, spec.sources, spec.seed, limit_per_source=args.limit)
    manifest.save(samples, header, args.out)
    print(json.dumps({k: header[k] for k in ("samples", "records", "merged_records", "groups", "hash")}, indent=2))
    return 0


def _analyze(args) -> int:
    from endoexp import analysis
    from endoexp.io import read_json
    out = Path(args.out)
    path = out / "results" / "summary.json"
    summary = read_json(path) if path.is_file() else {"suite": "unknown", "tasks": [], "errors": []}
    analysis.run(out, summary)
    print(out / "results" / "report.md")
    return 0


def _hub_manifest(args) -> int:
    """Print the JSON manifest for a gpu-experiment-hub issue."""
    from endoexp.config import load_models, load_suite
    spec = load_suite(_resolve(args.suite))
    models = load_models(_resolve(args.models))
    revisions = json.loads(Path(args.revisions).read_text()) if args.revisions else {}
    used = sorted({u.model for u in spec.units}, key=lambda k: [u.model for u in spec.units].index(k))
    main = next((k for k in used if models[k].get("role") != "student"), used[0])
    command = ["python", "-m", "endoexp", "run", "--suite", args.suite, "--models", args.models,
               "--data-root", args.data_root, "--models-root", args.models_root, "--out", "/work"]
    manifest = {
        "schema_version": 1,
        "code": {"repository": REPOSITORY, "commit": args.commit},
        "model": {"id": models[main]["repo"], "revision": revisions.get(main, "REPLACE_WITH_REVISION")},
        "dataset": {"id": args.dataset_id, "revision": args.dataset_revision},
        "command": command,
        "parameters": {
            "suite": spec.name, "seed": spec.seed, "sources": spec.sources, "max_pixels": spec.max_pixels,
            "caps": spec.caps, "pilot_command": command + ["--pilot"],
            "models": {k: {"repo": models[k]["repo"], "revision": revisions.get(k, "REPLACE_WITH_REVISION")}
                       for k in used},
        },
        "resources": {"gpu_count": args.gpus, "gpu_uuids": [], "vram_gb_per_gpu": args.vram_gb,
                      "max_runtime_minutes": None, "max_concurrent_jobs": None, "max_disk_gb": args.disk_gb},
        "pilot": {"max_runtime_minutes": args.pilot_minutes, "samples": spec.pilot["items_per_experiment"]},
        "deliverables": DELIVERABLES,
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="endoexp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite (pilot or full)")
    run.add_argument("--suite", required=True)
    run.add_argument("--models", default=DEFAULT_MODELS)
    run.add_argument("--data-root", required=True)
    run.add_argument("--models-root", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--pilot", action="store_true", help="small run of the pilot units only")
    run.add_argument("--engine", choices=["fake"], default=None, help="replace every engine (dry runs)")
    run.add_argument("--gpus", default="auto", help="'auto' or the number of GPUs to use")
    run.add_argument("--only", nargs="*", help="run only these units: MODEL or MODEL:ENGINE")
    run.add_argument("--vllm-arg", action="append", help="override a vLLM engine argument, KEY=VALUE")
    run.add_argument("--set-env", action="append", help="environment variable for every task, KEY=VALUE")
    run.set_defaults(func=_run)

    build = sub.add_parser("build", help="build the evaluation manifest only (CPU)")
    build.add_argument("--suite", required=True)
    build.add_argument("--data-root", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--limit", type=int, default=None, help="records per source (quick checks)")
    build.set_defaults(func=_build)

    analyze = sub.add_parser("analyze", help="recompute the deliverables of a finished run")
    analyze.add_argument("--out", required=True)
    analyze.set_defaults(func=_analyze)

    hub = sub.add_parser("hub-manifest", help="print the JSON manifest for a gpu-experiment-hub issue")
    hub.add_argument("--suite", required=True)
    hub.add_argument("--models", default=DEFAULT_MODELS)
    hub.add_argument("--commit", required=True)
    hub.add_argument("--revisions", help="JSON file {model_key: revision} written by tools/fetch_models.py")
    hub.add_argument("--dataset-id", default="endoopd-exp/kvasir-hyperkvasir")
    hub.add_argument("--dataset-revision", required=True, help="sha256 of the prepared data INDEX.json")
    hub.add_argument("--data-root", default="/inputs/data")
    hub.add_argument("--models-root", default="/inputs/models")
    hub.add_argument("--gpus", type=int, default=1)
    hub.add_argument("--vram-gb", type=int, default=90)
    hub.add_argument("--disk-gb", type=int, default=20)
    hub.add_argument("--pilot-minutes", type=int, default=15)
    hub.set_defaults(func=_hub_manifest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
