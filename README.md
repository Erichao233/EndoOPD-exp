# EndoOPD-exp

Experiment code for studying on-policy distillation (OPD) of vision-language
models in gastrointestinal endoscopy. The question behind it: can OPD transfer
*seeing* (finding and describing the lesion in the image) to a small student,
rather than answering habits (label priors, formatting, burned-in text)?

This repository holds code only. No data, images, labels, checkpoints or
results are stored here.

## What the first suite measures

`configs/suites/mainline_v0.toml` runs zero-training measurements that decide
which teacher a distillation study should use. The student is Qwen3.5-2B; the
candidate teachers are Qwen3.5-27B and Qwen3.8-27B (same architecture and
vocabulary). Every model answers the same closed-set report fields (MST-style:
finding → attribute → value), read out as a distribution over the options.

| id | question | how |
|---|---|---|
| E1 | Does a teacher see endoscopy better than the student? | Each field under the real image, no image (`blind`) and another image with the same field (`shuffled`); a dev-split label prior as baseline; teacher minus student on accuracy and on the image-specific gain |
| E2 | Does a privileged view help? | Polyp images with expert masks, shown plain, with a box, with a contour, or as a zoomed crop |
| E3 | Is the student missing the visual information, or not using it? | Linear probes on frozen hidden states vs the model's own answers on the same images |
| E4 | Do burned-in artefacts drive the answers? | Remove overlays, insert a scope-position inset, or write "POLYP" on normal images |
| E6 | Is the runtime trustworthy? | Tokenizer/template/image-token identity across models, vLLM vs HF letter distributions, multi-GPU NCCL, environment report |
| report | Can the models write the structured findings? | Free generation of the report template; parse rate and per-field accuracy |

All comparisons are paired on the same (image, field); intervals resample whole
image groups (duplicates of a frame across releases share a group).

## Layout

```
endoexp/            package
  fields.py         closed-set report fields
  sources/          dataset adapters (Kvasir-VQA, HyperKvasir labeled and segmented)
  manifest.py       link duplicate frames across releases, group, dev/test split
  views.py          privileged views and artefact manipulations
  experiments.py    which (image, field, condition) items each experiment asks
  engines/          vllm (scoring, generation), hf (exact scoring, features), fake (tests)
  task.py           one model x one shard in its own process
  suite.py          environment report, manifest, checks, GPU packing, analysis
  analysis.py       metrics, paired contrasts, probes, report
configs/            models and suites
tools/              run outside the sandbox: fetch checkpoints, prepare data
tests/              CPU tests on synthetic data
```

## Inputs (prepared outside the sandbox)

```
python tools/fetch_models.py --out <models-root> --only qwen35_2b qwen35_27b qwen38_27b
python tools/prepare_data.py --out <data-root>
```

`fetch_models.py` pins each checkpoint to a Hugging Face revision (a `REVISION`
file per model and `revisions.json`). `prepare_data.py` writes
`kvasir_vqa/` and `hyperkvasir/` and an `INDEX.json` whose sha256 identifies the
data version. Data sources and licenses:

- Kvasir-VQA, `SimulaMet-HOST/Kvasir-VQA`, CC BY-NC 4.0
- HyperKvasir labeled and segmented images, `SimulaMet-HOST/HyperKvasir`, CC BY 4.0

Use the data under their own terms; this repository does not redistribute them.

## Running

```
# CPU dry run with a deterministic fake engine
python -m endoexp run --suite configs/suites/mainline_v0.toml \
    --data-root <data-root> --models-root <models-root> --out <out> --engine fake

# real run: a short pilot, then the full suite
python -m endoexp run --suite configs/suites/mainline_v0.toml \
    --data-root <data-root> --models-root <models-root> --out <out> --pilot
python -m endoexp run --suite configs/suites/mainline_v0.toml \
    --data-root <data-root> --models-root <models-root> --out <out>
```

The suite uses every GPU visible to the job: each vLLM unit is split into one
shard per GPU and shards run in priority order. Runs are offline (no network,
no W&B). A failed task is recorded and the others continue.

Deliverables (relative to `--out`): `results/summary.json`, `results/report.md`,
`results/items.jsonl.gz`, `results/env.json`, `logs/suite.log`.

The pilot exits non-zero if any task fails; the full run exits non-zero only if
setup failed or no model task finished.

## gpu-experiment-hub

The job runs as `python -m endoexp run ... --out /work` with the repository
as the working directory. The environment, checkpoints and prepared data must
be mounted read-only into the sandbox; pass their paths with `--data-root` and
`--models-root`. To draft the issue manifest:

```
python -m endoexp hub-manifest --suite configs/suites/mainline_v0.toml \
    --commit <40-char sha> --revisions <models-root>/revisions.json \
    --dataset-revision <sha256 of INDEX.json> --gpus <n>
```

`git archive` writes the commit into `endoexp/_commit.py`, so results record the
exact code version.

## Tests

```
python -m pytest
```

The tests need only numpy, Pillow and pytest; they build a synthetic data root
and run the whole suite with the fake engine.

## License

Apache License 2.0 (see `LICENSE`).
