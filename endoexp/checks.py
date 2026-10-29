"""Runtime checks (E6): environment, tokenizer/template identity, multi-GPU communication."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from endoexp.engines import model_info
from endoexp.io import sha256_file, stable_hash

PACKAGES = ("torch", "transformers", "vllm", "accelerate", "tokenizers", "numpy", "pillow")
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HOME", "USER",
            "OMP_NUM_THREADS", "VLLM_ENABLE_V1_MULTIPROCESSING")


def _read(path: str, limit: int = 4000) -> str | None:
    try:
        return Path(path).read_text(errors="replace")[:limit]
    except OSError:
        return None


def environment(models: dict[str, str], data_root: str | Path) -> dict:
    """Cheap facts about the sandbox; never imports torch or vllm."""
    from importlib import metadata

    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    info = {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "cpu_count": os.cpu_count(), "packages": versions,
        "env": {key: os.environ.get(key) for key in ENV_KEYS},
        "cgroup": {name: _read(f"/sys/fs/cgroup/{name}") for name in
                   ("memory.max", "memory.swap.max", "pids.max", "pids.current")},
        "meminfo": (_read("/proc/meminfo") or "").splitlines()[:3],
        "models": {key: model_info(path) for key, path in models.items()},
    }
    try:
        stat = os.statvfs("/dev/shm")
        info["dev_shm"] = {"free_gb": round(stat.f_bavail * stat.f_frsize / 1e9, 2),
                           "writable": os.access("/dev/shm", os.W_OK)}
    except OSError as error:
        info["dev_shm"] = {"error": str(error)}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            result = subprocess.run(
                [smi, "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
            info["nvidia_smi"] = result.stdout.strip().splitlines() or result.stderr.strip()
        except (OSError, subprocess.SubprocessError) as error:
            info["nvidia_smi"] = str(error)
    index = Path(data_root) / "INDEX.json"
    info["data_index"] = {"path": str(index), "sha256": sha256_file(index)} if index.is_file() else None
    return info


def tokenizers(spec: dict) -> dict:
    """All models must share the student's vocabulary, special tokens and answer letters."""
    from PIL import Image
    from transformers import AutoProcessor

    from endoexp.imaging import fit
    from endoexp.prompts import letter_variants, messages

    image = fit(Image.new("RGB", (720, 576), (120, 60, 40)), spec["max_pixels"])
    reference, report = None, {}
    for key, path in spec["models"].items():
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        tokenizer = processor.tokenizer
        vocab = tokenizer.get_vocab()
        prompt = processor.apply_chat_template(messages("Question?", 1), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        encoded = processor(text=[prompt], images=[image], return_tensors="np")
        image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        entry = {
            "vocab_size": len(vocab), "vocab_hash": stable_hash(sorted(vocab.items())),
            "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
            "letters": letter_variants(tokenizer), "prompt_tail": prompt[-80:],
            "template_hash": stable_hash(processor.chat_template or ""),
            "image_tokens": int((encoded["input_ids"][0] == image_token).sum()),
            "image_size": list(image.size),
        }
        if reference is None:
            reference = entry
        entry["matches_first"] = all(entry[k] == reference[k] for k in
                                     ("vocab_hash", "eos_token_id", "pad_token_id", "letters", "image_tokens"))
        report[key] = entry
    return {"tokenizers": report, "all_match": all(e["matches_first"] for e in report.values())}


def _nccl_worker(rank: int, world: int, port: int, queue) -> None:
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    tensor = torch.ones(1 << 20, device="cuda") * (rank + 1)
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    queue.put((rank, float(tensor[0])))
    dist.destroy_process_group()


def nccl(spec: dict) -> dict:
    import socket

    import torch
    import torch.multiprocessing as mp

    world = torch.cuda.device_count()
    if world < 2:
        return {"skipped": f"{world} GPU visible"}
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_nccl_worker, args=(rank, world, port, queue)) for rank in range(world)]
    for process in processes:
        process.start()
    values = []
    for _ in range(world):
        try:
            values.append(queue.get(timeout=300))
        except Exception:  # queue.Empty: a worker died; exit codes below say which
            break
    for process in processes:
        process.join(timeout=60)
    values.sort()
    expected = world * (world + 1) / 2
    return {"world": world, "values": values, "ok": len(values) == world and all(v == expected for _, v in values),
            "exit_codes": [p.exitcode for p in processes]}
