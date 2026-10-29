"""Download checkpoints at pinned revisions. Run outside the sandbox (needs network).

    python tools/fetch_models.py --out <models-root> [--only qwen35_2b qwen35_27b]

Each checkpoint goes to <models-root>/<dir> with a REVISION file; the resolved
revisions are also written to <models-root>/revisions.json for the hub manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from endoexp.config import load_models  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=str(Path(__file__).resolve().parents[1] / "configs/models.toml"))
    parser.add_argument("--out", required=True, help="models root")
    parser.add_argument("--only", nargs="*", help="model keys to fetch (default: all)")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi, snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    revisions_path = out / "revisions.json"
    revisions = json.loads(revisions_path.read_text()) if revisions_path.exists() else {}
    for key, entry in load_models(args.models).items():
        if args.only and key not in args.only:
            continue
        revision = entry.get("revision") or api.model_info(entry["repo"]).sha
        target = out / entry["dir"]
        snapshot_download(entry["repo"], revision=revision, local_dir=target)
        (target / "REVISION").write_text(revision + "\n")
        revisions[key] = revision
        print(f"{key}: {entry['repo']}@{revision} -> {target}", flush=True)
    revisions_path.write_text(json.dumps(revisions, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
