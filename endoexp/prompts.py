"""Prompt texts and parsing. Engines wrap these texts in each model's chat template."""

from __future__ import annotations

import math
import re

from endoexp.fields import FIELDS, LETTERS, REPORT_FIELDS, Field

BLIND_NOTE = "No image is available for this question. Give your best guess."
DESCRIBE = "Describe this endoscopy image."


def mcq(field: Field, hint: str | None = None, blind: bool = False) -> str:
    lines = []
    if blind:
        lines.append(BLIND_NOTE)
    if hint:
        lines.append(hint)
    lines.append(field.question)
    lines.append("Options:")
    lines.extend(f"{letter}. {option}" for letter, option in zip(LETTERS, field.options))
    lines.append("Answer with the letter of the correct option only.")
    return "\n".join(lines)


def messages(text: str, n_images: int) -> list[dict]:
    content = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def report_template(field_names=REPORT_FIELDS) -> str:
    lines = ["Describe this endoscopy image by filling in the template below.",
             "For every line, write exactly one of the allowed values shown in brackets.",
             ""]
    for name in field_names:
        field = FIELDS[name]
        lines.append(f"{field.label}: [{' | '.join(field.options)}]")
    return "\n".join(lines)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9%<>.\-' ]", " ", text.lower()).split())


def parse_report(text: str, field_names=REPORT_FIELDS) -> dict[str, str | None]:
    """Map each template line back to an option; unparseable lines give None."""
    found: dict[str, str | None] = {name: None for name in field_names}
    by_label = {_norm(FIELDS[name].label): name for name in field_names}
    for raw in text.splitlines():
        if ":" not in raw:
            continue
        label, value = raw.split(":", 1)
        name = by_label.get(_norm(label.strip(" -*#")))
        if name is None or found[name] is not None:
            continue
        value = _norm(value.strip().strip("[]"))
        options = {_norm(option): option for option in FIELDS[name].options}
        if value in options:
            found[name] = options[value]
            continue
        prefixed = [o for key, o in options.items() if value and (key.startswith(value) or value.startswith(key))]
        if len(prefixed) == 1:
            found[name] = prefixed[0]
    return found


def letter_variants(tokenizer, n: int = len(LETTERS)) -> list[list[int]]:
    """Single-token ids that spell each answer letter (with and without a leading space)."""
    variants = []
    for letter in LETTERS[:n]:
        ids = []
        for text in (letter, " " + letter):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(encoded[0])
        if not ids:
            raise ValueError(f"letter {letter!r} is not a single token for this tokenizer")
        variants.append(ids)
    return variants


def letter_distribution(logprobs: dict[int, float], variants: list[list[int]],
                        n_options: int) -> tuple[list[float], float]:
    """Normalized distribution over option letters from next-token log-probabilities.

    ``logprobs`` may be a top-k subset. A letter missing from it gets half the
    smallest returned probability (an upper bound on what it can have).
    Returns the distribution and the raw probability mass the letters had.
    """
    floor = 0.5 * math.exp(min(logprobs.values())) if logprobs else 1.0
    raw = []
    for ids in variants[:n_options]:
        mass = sum(math.exp(logprobs[i]) for i in ids if i in logprobs)
        raw.append(mass)
    coverage = sum(raw)
    filled = [mass if mass > 0 else floor for mass in raw]
    total = sum(filled)
    return [mass / total for mass in filled], coverage
