"""Run a whole suite in one job.

Stages: environment report -> manifest -> checks -> model tasks packed onto the
job's GPUs -> analysis. Model work is split into one shard per GPU so the job
uses however many GPUs it was given. A failed task never stops the others; the
deliverables are always written, with failures recorded in the summary.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from endoexp import analysis, checks, manifest
from endoexp._commit import code_commit
from endoexp.config import ENGINE_DEFAULTS, load_models, load_suite
from endoexp.io import read_json, write_json

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
log = logging.getLogger("endoexp")


@dataclass
class Task:
    id: str
    spec: dict
    gpus: int
    priority: int
    timeout: float | None = None
    status: str = "pending"
    seconds: float = 0.0
    error: str | None = None
    gpu_ids: list[int] = field(default_factory=list)


def _setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        log.addHandler(handler)


def detect_gpus(timeout: float = 300) -> int:
    """Count CUDA devices in a child process so this process never initializes CUDA."""
    try:
        result = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                                capture_output=True, text=True, timeout=timeout)
        return int(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


def task_environment(gpu_ids: list[int], cache: Path) -> dict:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "VLLM_HOST_IP": "127.0.0.1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(cache / "hf"), "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "TRITON_CACHE_DIR": str(cache / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
    })
    # The sandbox has no /etc/passwd: libraries that ask for the user name must find it here.
    env.setdefault("USER", "endoexp")
    env.setdefault("LOGNAME", env["USER"])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        env.setdefault(name, "8")
    return env


def select_units(units, pilot: bool, only: list[str] | None = None) -> list:
    """Pilot units in pilot mode; ``only`` keeps units named ``model`` or ``model:engine``."""
    chosen = [u for u in units if u.pilot or not pilot]
    if only:
        chosen = [u for u in chosen if u.model in only or f"{u.model}:{u.engine}" in only]
    return chosen


def plan_tasks(suite, models: dict, models_root: Path, data_root: Path, manifest_dir: Path,
               work: Path, gpu_count: int, pilot: bool, engine: str | None,
               only: list[str] | None = None, vllm_overrides: dict | None = None) -> list[Task]:
    tasks = []
    units = select_units(suite.units, pilot, only)
    used = {}
    for unit in units:
        used.setdefault(unit.model, str(models_root / models[unit.model]["dir"]))
    timeout = suite.pilot["task_timeout_minutes"] * 60 if pilot else None
    if engine != "fake":
        if "tokenizers" in suite.checks and used:
            tasks.append(Task("check-tokenizers", {"kind": "tokenizers", "models": used,
                                                   "max_pixels": suite.max_pixels}, 0, -2, timeout))
        if "nccl" in suite.checks and gpu_count >= 2 and not pilot:
            tasks.append(Task("check-nccl", {"kind": "nccl"}, 2, -1, timeout))
    for unit in sorted(units, key=lambda u: u.priority):
        kind = engine or unit.engine
        shards = 1 if pilot else (max(1, gpu_count // unit.gpus) if unit.shards == "auto" else int(unit.shards))
        args = {**ENGINE_DEFAULTS[kind], **unit.engine_args}
        if kind == "vllm":
            args.update(tensor_parallel=unit.gpus, seed=suite.seed, **(vllm_overrides or {}))
        if models[unit.model].get("chat_template"):
            args["chat_template"] = str(models_root / models[unit.model]["dir"] / models[unit.model]["chat_template"])
        for shard in range(shards):
            task_id = f"p{unit.priority}-{unit.model}-{kind}-{shard + 1}of{shards}"
            spec = {
                "kind": "model", "id": task_id, "model": unit.model, "engine": kind,
                "model_path": used[unit.model], "engine_args": args, "experiments": list(unit.experiments),
                "shard": shard, "shards": shards, "manifest_dir": str(manifest_dir),
                "data_root": str(data_root), "out_dir": str(work / "tasks" / task_id),
                "caps": suite.caps, "seed": suite.seed, "max_pixels": suite.max_pixels,
                "pilot_items": suite.pilot["items_per_experiment"] if pilot else None,
                "report_tokens": suite.report_tokens, "chunk": suite.chunk,
            }
            unit_timeout = unit.timeout_minutes * 60 if unit.timeout_minutes else None
            tasks.append(Task(task_id, spec, unit.gpus, unit.priority, timeout or unit_timeout))
    return tasks


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _kill_group(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(3)


def schedule(tasks: list[Task], gpu_count: int, work: Path, logs: Path, poll: float = 2.0,
             extra_env: dict | None = None) -> None:
    """List scheduling in priority order; a task starts when enough GPUs are free."""
    pending = sorted(tasks, key=lambda t: t.priority)
    free = list(range(gpu_count))
    running: dict[str, tuple[Task, subprocess.Popen, float, object]] = {}
    cache = work / "cache"
    while pending or running:
        for task in list(pending):
            if task.gpus > gpu_count:
                task.status, task.error = "failed", f"needs {task.gpus} GPUs, job has {gpu_count}"
                pending.remove(task)
                log.error("%s: %s", task.id, task.error)
                continue
            if task.gpus > len(free):
                continue
            task.gpu_ids, free = free[:task.gpus], free[task.gpus:]
            spec_path = work / "specs" / f"{task.id}.json"
            write_json(spec_path, task.spec)
            log_path = logs / "tasks" / f"{task.id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_path, "w")
            process = subprocess.Popen(
                [sys.executable, "-m", "endoexp.task", "--spec", str(spec_path)],
                cwd=PACKAGE_PARENT, env={**task_environment(task.gpu_ids, cache), **(extra_env or {})},
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            running[task.id] = (task, process, time.time(), handle)
            pending.remove(task)
            task.status = "running"
            log.info("start %s on GPUs %s", task.id, task.gpu_ids)
        for task_id, (task, process, started, handle) in list(running.items()):
            code = process.poll()
            elapsed = time.time() - started
            if code is None and task.timeout and elapsed > task.timeout:
                log.error("%s: timeout after %.0fs", task.id, elapsed)
                _kill_group(process.pid)
                code = process.wait()
                task.error = f"timeout after {elapsed:.0f}s"
            if code is None:
                continue
            _kill_group(process.pid)  # reap any children the task left behind
            handle.close()
            del running[task_id]
            free = sorted(free + task.gpu_ids)
            task.seconds = round(elapsed, 1)
            out = Path(task.spec.get("out_dir", work / "tasks" / task.id))
            if code == 0 and (out / "done.json").is_file():
                task.status = "done"
                log.info("done %s in %.0fs", task.id, elapsed)
            else:
                task.status = "failed"
                error_file = out / "error.json"
                detail = read_json(error_file)["error"][-3000:] if error_file.is_file() else ""
                task.error = task.error or detail or _tail(logs / "tasks" / f"{task.id}.log")
                log.error("failed %s (exit %s)", task.id, code)
        if pending or running:
            time.sleep(poll)


def run(suite_path, models_path, data_root, models_root, out, pilot: bool = False,
        engine: str | None = None, gpus: str | int = "auto", only: list[str] | None = None,
        vllm_overrides: dict | None = None, extra_env: dict | None = None) -> int:
    out, data_root, models_root = Path(out), Path(data_root), Path(models_root)
    logs, results, work = out / "logs", out / "results", out / "work"
    _setup_logging(logs / "suite.log")
    suite = load_suite(suite_path)
    models = load_models(models_path)
    for task_path in (work / "tasks", work / "specs"):
        task_path.mkdir(parents=True, exist_ok=True)
    summary = {"suite": suite.name, "pilot": pilot, "code_commit": code_commit(),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tasks": [], "errors": [],
               "options": {"only": only, "vllm_overrides": vllm_overrides, "extra_env": extra_env}}
    tasks: list[Task] = []
    setup_ok = False
    try:
        log.info("suite %s (pilot=%s) commit %s", suite.name, pilot, summary["code_commit"])
        used = {u.model: str(models_root / models[u.model]["dir"]) for u in select_units(suite.units, pilot, only)}
        summary["roles"] = {key: models[key].get("role", "") for key in used}
        write_json(results / "env.json", {"suite": suite.name, "pilot": pilot,
                                          **checks.environment(used, data_root)})
        manifest_dir = work / "manifest"
        limit = suite.pilot["limit_per_source"] if pilot else None
        samples, header = manifest.build(data_root, suite.sources, suite.seed, limit_per_source=limit)
        manifest.save(samples, header, manifest_dir)
        log.info("manifest: %d samples (%d merged records), hash %s",
                 header["samples"], header["merged_records"], header["hash"])
        if engine == "fake":
            gpu_count = 1 if gpus == "auto" else int(gpus)
        else:
            gpu_count = detect_gpus() if gpus == "auto" else int(gpus)
        summary["gpus"] = gpu_count
        log.info("GPUs available to this job: %d", gpu_count)
        if gpu_count == 0:
            raise RuntimeError("no GPU visible to the job")
        tasks = plan_tasks(suite, models, models_root, data_root, manifest_dir, work, gpu_count, pilot, engine,
                           only=only, vllm_overrides=vllm_overrides)
        setup_ok = True
        schedule(tasks, gpu_count, work, logs, extra_env=extra_env)
    except Exception:
        summary["errors"].append(traceback.format_exc())
        log.error("suite error:\n%s", summary["errors"][-1])
    finally:
        summary["tasks"] = [{"id": t.id, "status": t.status, "seconds": t.seconds, "gpus": t.gpu_ids,
                             "error": t.error} for t in tasks]
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            analysis.run(out, summary)
        except Exception:
            summary["errors"].append(traceback.format_exc())
            log.error("analysis error:\n%s", summary["errors"][-1])
            analysis.write_minimal(out, summary)
    model_tasks = [t for t in tasks if t.spec.get("kind") == "model"]
    if pilot:
        ok = setup_ok and not summary["errors"] and all(t.status == "done" for t in tasks)
    else:
        ok = setup_ok and any(t.status == "done" for t in model_tasks)
    log.info("finished: %s", "ok" if ok else "FAILED")
    return 0 if ok else 1


__all__ = ["run", "plan_tasks", "schedule", "detect_gpus", "task_environment"]
