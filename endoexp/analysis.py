"""Turn task outputs into the deliverables: summary.json, report.md, items.jsonl.gz.

Every comparison is paired on the same (sample, field) and its interval
resamples whole image groups.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from endoexp import manifest
from endoexp.fields import FIELDS, KINDS
from endoexp.io import read_jsonl, write_json, write_jsonl
from endoexp.metrics import clustered_bootstrap, fmt, summarize
from endoexp.probe import ridge_probe

# Option whose share should move under a manipulation if the model reads the artefact.
SHIFT_TARGETS = {
    ("insert_text_polyp", "abnormality"): "polyp",
    ("insert_inset", "box_overlay"): "yes",
    ("insert_inset", "procedure"): "colonoscopy",
    ("clean", "text_overlay"): "yes",
    ("clean", "box_overlay"): "yes",
}
MIN_PROBE_TRAIN, MIN_PROBE_TEST = 20, 10


def load_rows(work: Path) -> list[dict]:
    rows = []
    for path in sorted((work / "tasks").glob("*/rows.jsonl.gz")):
        rows.extend(read_jsonl(path))
    return rows


def _field_order(name: str) -> tuple:
    return (KINDS.index(FIELDS[name].kind), list(FIELDS).index(name))


def condition_metrics(scored: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in scored:
        groups[(row["exp"], row["model"], row["condition"], row["field"])].append(row)
    table = []
    for (exp, model, condition, name), rows in sorted(groups.items()):
        spec = FIELDS[name]
        labels = [spec.index(r["label"]) for r in rows]
        probs = [r["probs"] for r in rows]
        table.append({"exp": exp, "model": model, "condition": condition, "field": name, "kind": spec.kind,
                      **summarize(labels, probs, spec.ordinal)})
    return table


def _grouped(scored: list[dict]) -> dict:
    """(exp, model, condition, field) -> {uid: row}."""
    grouped = defaultdict(dict)
    for row in scored:
        grouped[(row["exp"], row["model"], row["condition"], row["field"])][row["uid"]] = row
    return grouped


def _correct(row: dict) -> float:
    return float(row["pred"] == row["label"])


def _p_true(row: dict) -> float:
    return row["probs"][FIELDS[row["field"]].index(row["label"])]


def _pairs(grouped: dict, exp: str, model: str, name: str, a: str, b: str):
    first, second = grouped.get((exp, model, a, name), {}), grouped.get((exp, model, b, name), {})
    return [(first[uid], second[uid]) for uid in sorted(first) if uid in second]


def contrasts(scored: list[dict], seed: int = 0) -> list[dict]:
    """Condition a minus condition b for the same model, field and samples."""
    grouped = _grouped(scored)
    combos = sorted({(r["exp"], r["model"], r["field"], r["condition"]) for r in scored})
    table = []
    for exp, model, name, condition in combos:
        if exp == "e1" and condition in ("blind", "shuffled"):
            a, b = "normal", condition
        elif exp in ("e2", "e4") and condition != "plain":
            a, b = condition, "plain"
        else:
            continue
        pairs = _pairs(grouped, exp, model, name, a, b)
        if not pairs:
            continue
        groups = [p[0]["group"] for p in pairs]
        entry = {
            "exp": exp, "model": model, "field": name, "a": a, "b": b,
            "acc_diff": clustered_bootstrap([_correct(x) - _correct(y) for x, y in pairs], groups, seed=seed),
            "p_true_diff": clustered_bootstrap([_p_true(x) - _p_true(y) for x, y in pairs], groups, seed=seed),
            "flip_rate": float(np.mean([x["pred"] != y["pred"] for x, y in pairs])),
        }
        target = SHIFT_TARGETS.get((a, name))
        if target is not None:
            entry["shift_target"] = target
            entry["shift"] = clustered_bootstrap(
                [float(x["pred"] == target) - float(y["pred"] == target) for x, y in pairs], groups, seed=seed)
        table.append(entry)
    return table


def teacher_vs_student(scored: list[dict], roles: dict[str, str], seed: int = 0) -> list[dict]:
    """E1: teacher minus student accuracy, and the same for the image-specific gain (normal - shuffled)."""
    students = [m for m, r in roles.items() if r == "student"]
    teachers = [m for m, r in roles.items() if r != "student"]
    grouped = _grouped(scored)
    present = {r["model"] for r in scored}
    table = []
    for student in (s for s in students if s in present):
        for teacher in (t for t in teachers if t in present):
            for name in sorted({r["field"] for r in scored if r["exp"] == "e1"}, key=_field_order):
                diffs, gains, groups = [], [], []
                s_normal = grouped.get(("e1", student, "normal", name), {})
                t_normal = grouped.get(("e1", teacher, "normal", name), {})
                s_shuffled = grouped.get(("e1", student, "shuffled", name), {})
                t_shuffled = grouped.get(("e1", teacher, "shuffled", name), {})
                for uid in sorted(s_normal):
                    s_row, t_row = s_normal[uid], t_normal.get(uid)
                    s_shuf, t_shuf = s_shuffled.get(uid), t_shuffled.get(uid)
                    if t_row is None:
                        continue
                    diffs.append(_correct(t_row) - _correct(s_row))
                    groups.append(s_row["group"])
                    if s_shuf is not None and t_shuf is not None:
                        gains.append(((_correct(t_row) - _correct(t_shuf))
                                      - (_correct(s_row) - _correct(s_shuf)), s_row["group"]))
                if not diffs:
                    continue
                table.append({
                    "field": name, "kind": FIELDS[name].kind, "student": student, "teacher": teacher,
                    "acc_diff": clustered_bootstrap(diffs, groups, seed=seed),
                    "image_gain_diff": clustered_bootstrap([g for g, _ in gains], [k for _, k in gains], seed=seed),
                })
    return table


def prior_baselines(samples, scored: list[dict]) -> list[dict]:
    """Accuracy of answering each field with its most frequent dev label within the source."""
    counts = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        if sample.split == "dev":
            for name, value in sample.labels.items():
                counts[(sample.source, name)][value] += 1
    prior = {key: max(values, key=values.get) for key, values in counts.items()}
    items = {}
    for row in scored:
        if row["exp"] == "e1" and row["condition"] == "normal":
            items[(row["field"], row["uid"])] = row
    per_field = defaultdict(list)
    for (name, _), row in items.items():
        guess = prior.get((row["source"], name))
        if guess is not None:
            per_field[name].append(float(guess == row["label"]))
    return [{"field": name, "n": len(values), "acc": float(np.mean(values))}
            for name, values in sorted(per_field.items(), key=lambda kv: _field_order(kv[0]))]


def probes(work: Path, samples, scored: list[dict]) -> list[dict]:
    by_model = defaultdict(list)
    for path in sorted((work / "tasks").glob("*/features.npz")):
        data = np.load(path, allow_pickle=False)
        by_model[str(data["model"])].append({k: data[k] for k in data.files})
    by_uid = {s.uid: s for s in samples}
    answers = {(r["model"], r["field"], r["uid"]): _correct(r) for r in scored
               if r["exp"] == "e1" and r["condition"] == "normal"}
    table = []
    for model, parts in sorted(by_model.items()):
        uids = np.concatenate([p["uids"] for p in parts])
        names = sorted(set.intersection(*(set(k for k in p if k.startswith("feat_")) for p in parts)))
        for feature in names:
            matrix = np.concatenate([p[feature] for p in parts]).astype(np.float32)
            for name in sorted(FIELDS, key=_field_order):
                spec = FIELDS[name]
                rows = [(i, by_uid[u]) for i, u in enumerate(uids) if u in by_uid and name in by_uid[u].labels]
                train = [(i, s) for i, s in rows if s.split == "dev"]
                test = [(i, s) for i, s in rows if s.split == "test"]
                if len(train) < MIN_PROBE_TRAIN or len(test) < MIN_PROBE_TEST:
                    continue
                y_train = np.array([spec.index(s.labels[name]) for _, s in train])
                if len(set(y_train)) < 2:
                    continue
                y_test = np.array([spec.index(s.labels[name]) for _, s in test])
                result = ridge_probe(matrix[[i for i, _ in train]], y_train,
                                     matrix[[i for i, _ in test]], y_test, len(spec.options))
                pred = result.pop("pred")
                answered = [(k, answers[(model, name, s.uid)]) for k, (_, s) in enumerate(test)
                            if (model, name, s.uid) in answers]
                entry = {"model": model, "feature": feature[len("feat_"):], "field": name,
                         "kind": spec.kind, **result}
                if answered:
                    keep = [k for k, _ in answered]
                    entry["paired_n"] = len(keep)
                    entry["probe_acc_paired"] = float((pred[keep] == y_test[keep]).mean())
                    entry["answer_acc_paired"] = float(np.mean([a for _, a in answered]))
                table.append(entry)
    return table


def report_metrics(reports: list[dict], samples) -> list[dict]:
    by_uid = {s.uid: s for s in samples}
    stats = defaultdict(lambda: {"parsed": 0, "labeled": 0, "correct": 0, "n": 0})
    for row in reports:
        sample = by_uid.get(row["uid"])
        for name, value in row["parsed"].items():
            entry = stats[(row["model"], name)]
            entry["n"] += 1
            entry["parsed"] += value is not None
            if sample is not None and name in sample.labels:
                entry["labeled"] += 1
                entry["correct"] += value == sample.labels[name]
    return [{"model": model, "field": name, "n": s["n"], "parse_rate": s["parsed"] / s["n"],
             "labeled": s["labeled"], "acc": s["correct"] / s["labeled"] if s["labeled"] else None}
            for (model, name), s in sorted(stats.items(), key=lambda kv: (kv[0][0], _field_order(kv[0][1])))]


def parity(scored: list[dict]) -> list[dict]:
    vllm = {(r["model"], r["field"], r["uid"]): r for r in scored if r["exp"] == "e1" and r["condition"] == "normal"}
    groups = defaultdict(list)
    for row in scored:
        if row["exp"] == "parity":
            other = vllm.get((row["model"], row["field"], row["uid"]))
            if other is not None:
                tv = 0.5 * sum(abs(a - b) for a, b in zip(row["probs"], other["probs"]))
                groups[row["model"]].append((tv, row["pred"] == other["pred"]))
    return [{"model": model, "n": len(v), "mean_tv": float(np.mean([t for t, _ in v])),
             "max_tv": float(np.max([t for t, _ in v])), "argmax_agree": float(np.mean([a for _, a in v]))}
            for model, v in sorted(groups.items())]


def coverage(scored: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in scored:
        groups[(row["model"], row["exp"], row["condition"])].append(row["coverage"])
    return [{"model": m, "exp": e, "condition": c, "n": len(v), "mean": float(np.mean(v)),
             "p05": float(np.percentile(v, 5))} for (m, e, c), v in sorted(groups.items())]


def _ci(entry: dict | None) -> str:
    if not entry or entry.get("mean") is None:
        return "–"
    return f"{fmt(entry['mean'])} [{fmt(entry['lo'])}, {fmt(entry['hi'])}]"


def _above_zero(entry: dict | None) -> bool:
    return bool(entry) and entry.get("lo") is not None and entry["lo"] > 0


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(row) + " |" for row in rows])


def _best_probes(table: list[dict]) -> dict:
    best = {}
    for row in table:
        key = (row["model"], row["field"])
        if key not in best or row["acc"] > best[key]["acc"]:
            best[key] = row
    return best


def indicators(tables: dict) -> list[str]:
    """Plain-language list of the differences whose 95% interval excludes zero."""
    lines = []
    by_teacher = defaultdict(lambda: {"acc": [], "gain": []})
    for row in tables.get("teacher_vs_student", []):
        if row["kind"] in ("descriptor", "impression"):
            if _above_zero(row["acc_diff"]):
                by_teacher[row["teacher"]]["acc"].append(row["field"])
            if _above_zero(row["image_gain_diff"]):
                by_teacher[row["teacher"]]["gain"].append(row["field"])
    for row in tables.get("teacher_vs_student", [])[:1]:
        lines.append(f"- Student: {row['student']}.")
    for teacher in sorted({r["teacher"] for r in tables.get("teacher_vs_student", [])}):
        found = by_teacher[teacher]
        lines.append(f"- {teacher} beats the student on descriptor/impression fields: "
                     f"{', '.join(found['acc']) or 'none'}; with a larger image-specific gain: "
                     f"{', '.join(found['gain']) or 'none'}.")
    helped = defaultdict(list)
    shortcuts = defaultdict(list)
    for row in tables.get("contrasts", []):
        if row["exp"] == "e2" and _above_zero(row["acc_diff"]):
            helped[row["model"]].append(f"{row['field']}/{row['a']}")
        if row["exp"] == "e4" and _above_zero(row.get("shift")):
            shortcuts[row["model"]].append(f"{row['field']}/{row['a']}")
    for model in sorted({r["model"] for r in tables.get("contrasts", []) if r["exp"] == "e2"}):
        lines.append(f"- E2 views that help {model}: {', '.join(helped[model]) or 'none'}.")
    for model in sorted({r["model"] for r in tables.get("contrasts", []) if r["exp"] == "e4"}):
        lines.append(f"- E4 artefacts that move {model}'s answers: {', '.join(shortcuts[model]) or 'none'}.")
    gaps = defaultdict(list)
    for (model, name), row in _best_probes(tables.get("probes", [])).items():
        informative = row["acc"] - row["majority_acc"] >= 0.05
        if informative and row.get("answer_acc_paired") is not None \
                and row["probe_acc_paired"] - row["answer_acc_paired"] >= 0.10:
            gaps[model].append(name)
    for model in sorted({r["model"] for r in tables.get("probes", [])}):
        lines.append(f"- E3 fields readable from {model}'s frozen features but missed by its answers "
                     f"(probe ≥ majority + 5 and ≥ answer + 10 points): {', '.join(gaps[model]) or 'none'}.")
    low = [r for r in tables.get("coverage", []) if r["mean"] < 0.8]
    if low:
        lines.append("- Warning: little probability on option letters for "
                     + ", ".join(f"{r['model']}/{r['exp']}/{r['condition']} ({fmt(r['mean'])}%)" for r in low) + ".")
    for row in tables.get("parity", []):
        if row["mean_tv"] > 0.05 or row["argmax_agree"] < 0.95:
            lines.append(f"- Warning: vLLM and HF disagree for {row['model']} "
                         f"(mean TV {row['mean_tv']:.3f}, argmax agreement {fmt(row['argmax_agree'])}%).")
    return lines


def render(summary: dict) -> str:
    t = summary.get("tables", {})
    lines = [f"# {summary.get('suite')}{' (pilot)' if summary.get('pilot') else ''}", "",
             f"Commit `{summary.get('code_commit')}`; GPUs: {summary.get('gpus')}; "
             f"{summary.get('started')} → {summary.get('finished')}. Numbers are percentages; "
             "intervals are 95% and resample whole image groups.", ""]
    found = indicators(t)
    if found:
        lines += ["## Decision indicators", ""] + found
    for error in summary.get("errors", []):
        lines += ["", "```", error.strip()[-2000:], "```"]

    metrics = t.get("metrics", [])
    models = sorted({r["model"] for r in metrics})
    e1 = {(r["model"], r["condition"], r["field"]): r for r in metrics if r["exp"] == "e1"}
    prior = {r["field"]: r for r in t.get("priors", [])}
    for model in models:
        names = sorted({k[2] for k in e1 if k[0] == model}, key=_field_order)
        if not names:
            continue
        rows = []
        for name in names:
            normal = e1.get((model, "normal", name), {})
            rows.append([name, FIELDS[name].kind, str(normal.get("n", 0)), fmt(prior.get(name, {}).get("acc")),
                         fmt(normal.get("acc")), fmt(e1.get((model, "blind", name), {}).get("acc")),
                         fmt(e1.get((model, "shuffled", name), {}).get("acc")), fmt(normal.get("auc")),
                         fmt(normal.get("p_true"))])
        lines += ["", f"## E1 {model}: accuracy by condition", ""]
        lines += _table(["field", "kind", "n", "dev prior", "normal", "blind", "shuffled", "AUC", "p(true)"], rows)
    by_teacher = defaultdict(list)
    for row in t.get("teacher_vs_student", []):
        by_teacher[(row["teacher"], row["student"])].append(row)
    for (teacher, student), table in sorted(by_teacher.items()):
        lines += ["", f"## E1 {teacher} minus {student}", ""]
        lines += _table(["field", "kind", "accuracy", "image-specific gain (normal - shuffled)"],
                        [[r["field"], r["kind"], _ci(r["acc_diff"]), _ci(r["image_gain_diff"])] for r in table])
    for exp, title in (("e2", "E2: privileged view minus plain"), ("e4", "E4: manipulation minus plain")):
        table = [r for r in t.get("contrasts", []) if r["exp"] == exp]
        if not table:
            continue
        exp_models = sorted({r["model"] for r in table})
        cells = defaultdict(dict)
        for r in table:
            text = _ci(r["acc_diff"])
            if "shift" in r:
                text += f"; {r['shift_target']} {_ci(r['shift'])}"
            cells[(r["a"], r["field"])][r["model"]] = text
        rows = [[view, name] + [cells[(view, name)].get(m, "–") for m in exp_models]
                for view, name in sorted(cells, key=lambda k: (k[0], _field_order(k[1])))]
        lines += ["", f"## {title} (accuracy; target-option shift)", ""]
        lines += _table(["view", "field"] + exp_models, rows)
    best = _best_probes(t.get("probes", []))
    if best:
        rows = [[model, name, r["feature"], fmt(r["acc"]), fmt(r["majority_acc"]), fmt(r.get("probe_acc_paired")),
                 fmt(r.get("answer_acc_paired")), str(r["n_test"])]
                for (model, name), r in sorted(best.items(), key=lambda kv: (kv[0][0], _field_order(kv[0][1])))]
        lines += ["", "## E3: linear probe on frozen features vs the model's own answer (best feature per field)", ""]
        lines += _table(["model", "field", "feature", "probe", "majority", "probe (paired)", "answer (paired)",
                         "n test"], rows)
    reports = t.get("reports", [])
    if reports:
        report_models = sorted({r["model"] for r in reports})
        cells = defaultdict(dict)
        for r in reports:
            cells[r["field"]][r["model"]] = f"{fmt(r['parse_rate'])} / {fmt(r['acc'])}"
        lines += ["", "## Structured findings, free generation (parse rate / accuracy)", ""]
        lines += _table(["field"] + report_models,
                        [[name] + [cells[name].get(m, "–") for m in report_models]
                         for name in sorted(cells, key=_field_order)])
    lines += ["", "## Tasks", ""]
    lines += _table(["task", "status", "seconds", "GPUs", "error"],
                    [[task["id"], task["status"], str(task["seconds"]), str(task["gpus"]),
                      ((task.get("error") or "").strip().splitlines() or [""])[-1][:120]]
                     for task in summary.get("tasks", [])])
    m = summary.get("manifest") or {}
    if m:
        lines += ["", "## Manifest", "", f"{m.get('samples')} samples from {m.get('records')} records "
                  f"({m.get('merged_records')} merged duplicates), hash `{m.get('hash')}`.", ""]
        lines += _table(["field", "dev", "test"], [[name, str(sum(c["dev"].values())), str(sum(c["test"].values()))]
                                                   for name, c in m.get("fields", {}).items()])
    if t.get("parity"):
        lines += ["", "## vLLM vs HF parity", ""]
        lines += _table(["model", "n", "mean TV", "max TV", "argmax agree"],
                        [[r["model"], str(r["n"]), f"{r['mean_tv']:.4f}", f"{r['max_tv']:.4f}",
                          fmt(r["argmax_agree"])] for r in t["parity"]])
    if t.get("coverage"):
        lines += ["", "## Probability mass on option letters", ""]
        lines += _table(["model", "exp", "condition", "mean", "5th percentile"],
                        [[r["model"], r["exp"], r["condition"], fmt(r["mean"]), fmt(r["p05"])]
                         for r in t["coverage"]])
    return "\n".join(lines) + "\n"


def write_minimal(out: Path, summary: dict) -> None:
    results = Path(out) / "results"
    write_json(results / "summary.json", summary)
    if not (results / "items.jsonl.gz").exists():
        write_jsonl(results / "items.jsonl.gz", [])
    (results / "report.md").write_text(render(summary), encoding="utf-8")
    if not (results / "env.json").exists():
        write_json(results / "env.json", {"error": "environment report was not written"})


def run(out: str | Path, summary: dict) -> None:
    out = Path(out)
    work, results = out / "work", out / "results"
    samples, header = ([], {})
    if (work / "manifest" / "manifest.json").is_file():
        samples, header = manifest.load(work / "manifest")
        summary["manifest"] = {k: header.get(k) for k in
                               ("samples", "records", "merged_records", "groups", "hash", "fields", "sources")}
    rows = load_rows(work)
    scored = [r for r in rows if "probs" in r]
    reports = [r for r in rows if r.get("exp") == "report"]
    summary["tables"] = {
        "metrics": condition_metrics(scored),
        "contrasts": contrasts(scored),
        "teacher_vs_student": teacher_vs_student(scored, summary.get("roles", {})),
        "priors": prior_baselines(samples, scored),
        "probes": probes(work, samples, scored),
        "reports": report_metrics(reports, samples),
        "parity": parity(scored),
        "coverage": coverage(scored),
    }
    write_jsonl(results / "items.jsonl.gz", rows)
    write_minimal(out, summary)
