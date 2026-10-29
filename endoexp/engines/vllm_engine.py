"""vLLM engine: letter distributions and greedy generation for Qwen3.5 checkpoints."""

from __future__ import annotations

import gc
from pathlib import Path

from endoexp.engines import Query
from endoexp.prompts import letter_distribution, letter_variants, messages


def _logprob_dict(step) -> dict[int, float]:
    items = step.items() if hasattr(step, "items") else step
    return {int(token): float(getattr(value, "logprob", value)) for token, value in items}


class VLLMEngine:
    def __init__(self, model_path: str | Path, *, tensor_parallel: int = 1, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.85, max_logprobs: int = 20, seed: int = 0,
                 chat_template: str | None = None, max_images: int = 1, enforce_eager: bool = True,
                 batch_size: int = 256):
        from transformers import AutoProcessor
        from vllm import LLM

        self.model_path = str(model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        if chat_template:
            template = Path(chat_template).read_text(encoding="utf-8")
            self.processor.chat_template = template
            self.processor.tokenizer.chat_template = template
        self.tokenizer = self.processor.tokenizer
        self.variants = letter_variants(self.tokenizer)
        self.max_logprobs = max_logprobs
        self.batch_size = batch_size
        self.llm = LLM(
            model=self.model_path, tokenizer=self.model_path, dtype="bfloat16",
            tensor_parallel_size=tensor_parallel, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, max_logprobs=max_logprobs, seed=seed,
            enforce_eager=enforce_eager, enable_prefix_caching=True,
            limit_mm_per_prompt={"image": max_images})

    def render(self, text: str, n_images: int) -> str:
        return self.processor.apply_chat_template(
            messages(text, n_images), tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _input(self, query: Query) -> dict:
        prompt = self.render(query.text, len(query.images))
        if not query.images:
            return {"prompt": prompt}
        image = query.images[0] if len(query.images) == 1 else list(query.images)
        return {"prompt": prompt, "multi_modal_data": {"image": image}}

    def score(self, queries: list[Query]) -> dict[str, dict]:
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=self.max_logprobs)
        results = {}
        for start in range(0, len(queries), self.batch_size):
            chunk = queries[start:start + self.batch_size]
            outputs = self.llm.generate([self._input(q) for q in chunk], params, use_tqdm=False)
            for query, output in zip(chunk, outputs):
                logprobs = _logprob_dict(output.outputs[0].logprobs[0])
                probs, coverage = letter_distribution(logprobs, self.variants, query.n_options)
                results[query.key] = {"probs": probs, "coverage": coverage,
                                      "top_token": self.tokenizer.decode([max(logprobs, key=logprobs.get)])}
        return results

    def generate(self, queries: list[Query]) -> dict[str, str]:
        from vllm import SamplingParams
        results = {}
        for start in range(0, len(queries), self.batch_size):
            chunk = queries[start:start + self.batch_size]
            params = SamplingParams(max_tokens=max(q.max_tokens for q in chunk), temperature=0.0)
            outputs = self.llm.generate([self._input(q) for q in chunk], params, use_tqdm=False)
            for query, output in zip(chunk, outputs):
                results[query.key] = output.outputs[0].text
        return results

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(getattr(engine, "engine_core", None), "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # best effort: the task process exits right after
                pass
        self.llm = None
        gc.collect()
