"""End-to-end: the whole suite with the fake engine, task subprocesses and two GPU slots."""

import json

from endoexp import suite
from endoexp.cli import DELIVERABLES, main
from endoexp.io import read_json, read_jsonl


def test_full_suite_with_fake_engine(data_root, repo_root, tmp_path):
    out = tmp_path / "run"
    code = suite.run(repo_root / "configs/suites/mainline_v0.toml", repo_root / "configs/models.toml",
                     data_root, tmp_path / "models", out, engine="fake", gpus=2)
    assert code == 0
    for name in DELIVERABLES:
        assert (out / name).is_file(), name
    summary = read_json(out / "results/summary.json")
    assert summary["tasks"] and all(t["status"] == "done" for t in summary["tasks"])
    assert summary["acceptance"]["ok"] and summary["acceptance"]["parity_ok"]
    digest = (out / "results/digest.md").read_text()
    assert "**Acceptance: PASS**" in digest and "## E1 teacher minus student" in digest
    assert len(digest) < 20000
    shards = [t for t in summary["tasks"] if "qwen35_27b-fake" in t["id"]]
    assert len(shards) == 3                        # two vllm shards + one hf shard
    tables = summary["tables"]
    assert {r["exp"] for r in tables["metrics"]} >= {"e1", "e2", "e4", "parity"}
    assert tables["teacher_vs_student"] and tables["priors"] and tables["probes"] and tables["reports"]
    assert tables["parity"][0]["argmax_agree"] == 1.0  # same fake model in both engines
    rows = list(read_jsonl(out / "results/items.jsonl.gz"))
    assert {r["model"] for r in rows} == {"qwen35_2b", "qwen35_27b", "qwen38_27b"}
    report = (out / "results/report.md").read_text()
    assert "## Decision indicators" in report and "## E1 qwen35_2b: accuracy by condition" in report
    assert "## E1 qwen35_27b minus qwen35_2b" in report and "## E2: privileged view minus plain" in report


def test_pilot_runs_only_pilot_units(data_root, repo_root, tmp_path):
    out = tmp_path / "pilot"
    code = suite.run(repo_root / "configs/suites/mainline_v0.toml", repo_root / "configs/models.toml",
                     data_root, tmp_path / "models", out, pilot=True, engine="fake", gpus=1)
    assert code == 0
    summary = read_json(out / "results/summary.json")
    assert {t["id"].split("-")[1] for t in summary["tasks"]} == {"qwen35_2b"}
    rows = list(read_jsonl(out / "results/items.jsonl.gz"))
    assert rows and len({(r["uid"], r.get("field")) for r in rows if r["exp"] == "e1"}) <= 16


def test_failures_are_recorded_and_deliverables_still_written(data_root, repo_root, tmp_path):
    out = tmp_path / "broken"
    code = suite.run(repo_root / "configs/suites/mainline_v0.toml", repo_root / "configs/models.toml",
                     tmp_path / "missing-data", tmp_path / "models", out, engine="fake", gpus=1)
    assert code == 1
    summary = read_json(out / "results/summary.json")
    assert summary["errors"] and not summary["acceptance"]["ok"]
    for name in DELIVERABLES:
        assert (out / name).is_file(), name
    assert "**Acceptance: FAIL**" in (out / "results/digest.md").read_text()


def test_hub_manifest(capsys, repo_root):
    assert main(["hub-manifest", "--suite", "configs/suites/mainline_v0.toml", "--commit", "a" * 40,
                 "--dataset-revision", "b" * 64, "--gpus", "4"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert set(manifest) == {"schema_version", "code", "model", "dataset", "command", "parameters",
                             "resources", "pilot", "deliverables"}
    assert manifest["model"]["id"] == "Qwen/Qwen3.5-27B"
    assert manifest["command"][:4] == ["python", "-m", "endoexp", "run"]
    assert manifest["parameters"]["pilot_command"][-1] == "--pilot"
    assert str(repo_root) not in json.dumps(manifest)


def test_only_and_overrides(data_root, repo_root, tmp_path):
    out = tmp_path / "only"
    code = main(["run", "--suite", str(repo_root / "configs/suites/mainline_v0.toml"),
                 "--data-root", str(data_root), "--models-root", str(tmp_path / "models"), "--out", str(out),
                 "--engine", "fake", "--gpus", "2", "--only", "qwen35_27b:vllm",
                 "--vllm-arg", "gpu_memory_utilization=0.8", "--set-env", "EXTRA_FLAG=1"])
    assert code == 0
    summary = read_json(out / "results/summary.json")
    assert {t["id"].split("-")[1] for t in summary["tasks"]} == {"qwen35_27b"}
    assert summary["options"]["vllm_overrides"] == {"gpu_memory_utilization": 0.8}
    assert summary["options"]["extra_env"] == {"EXTRA_FLAG": "1"}
