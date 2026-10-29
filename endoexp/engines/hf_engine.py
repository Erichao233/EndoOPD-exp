"""Hugging Face engine: exact letter distributions and hidden-state features."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from endoexp.engines import Query
from endoexp.prompts import letter_distribution, letter_variants, messages


class HFEngine:
    def __init__(self, model_path: str | Path, *, device: str = "cuda:0", chat_template: str | None = None):
        import torch
        import transformers
        from transformers import AutoProcessor

        self.torch = torch
        self.device = torch.device(device)
        self.model_path = str(model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        if chat_template:
            template = Path(chat_template).read_text(encoding="utf-8")
            self.processor.chat_template = template
            self.processor.tokenizer.chat_template = template
        self.tokenizer = self.processor.tokenizer
        self.variants = letter_variants(self.tokenizer)
        model_class = (getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
                       or transformers.AutoModelForImageTextToText)
        self.model = model_class.from_pretrained(
            self.model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, device_map={"": str(self.device)}).eval().requires_grad_(False)
        if next(self.model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError("model did not load in bfloat16; check the transformers version")
        image_token = getattr(self.model.config, "image_token_id", None)
        self.image_token_id = image_token if image_token is not None else \
            self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def render(self, text: str, n_images: int) -> str:
        return self.processor.apply_chat_template(
            messages(text, n_images), tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _encode(self, query: Query) -> dict:
        prompt = self.render(query.text, len(query.images))
        inputs = self.processor(text=[prompt], images=list(query.images) or None, return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    def score(self, queries: list[Query]) -> dict[str, dict]:
        torch = self.torch
        ids = sorted({i for variant in self.variants for i in variant})
        results = {}
        with torch.inference_mode():
            for query in queries:
                logits = self.model(**self._encode(query), use_cache=False, logits_to_keep=1).logits[0, -1]
                logp = torch.log_softmax(logits.float(), dim=-1)
                logprobs = {i: float(logp[i]) for i in ids}
                probs, coverage = letter_distribution(logprobs, self.variants, query.n_options)
                results[query.key] = {"probs": probs, "coverage": coverage,
                                      "top_token": self.tokenizer.decode([int(logp.argmax())])}
        return results

    def features(self, queries: list[Query]) -> dict[str, dict[str, np.ndarray]]:
        """Mean over image-token positions and the last position, at three depths."""
        torch = self.torch
        results = {}
        with torch.inference_mode():
            for query in queries:
                inputs = self._encode(query)
                hidden = self.model(**inputs, use_cache=False, output_hidden_states=True).hidden_states
                depth = len(hidden) - 1
                positions = inputs["input_ids"][0] == self.image_token_id
                vectors = {}
                for layer in sorted({0, depth // 2, depth}):
                    states = hidden[layer][0]
                    if bool(positions.any()):
                        vectors[f"L{layer}_image"] = states[positions].float().mean(0).cpu().numpy()
                    vectors[f"L{layer}_last"] = states[-1].float().cpu().numpy()
                results[query.key] = {name: v.astype(np.float16) for name, v in vectors.items()}
        return results

    def generate(self, queries: list[Query]) -> dict[str, str]:
        results = {}
        with self.torch.inference_mode():
            for query in queries:
                inputs = self._encode(query)
                output = self.model.generate(**inputs, max_new_tokens=query.max_tokens, do_sample=False)
                results[query.key] = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                                            skip_special_tokens=True)
        return results

    def close(self) -> None:
        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
