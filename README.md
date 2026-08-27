# CogNVS Reproduction

A reproduction and evaluation harness for **CogNVS** (Consistent Novel-View
Video Synthesis), built around the authors' upstream codebase
([`Kaihua-Chen/cog-nvs`](https://github.com/Kaihua-Chen/cog-nvs),
commit `9d7d00cb378b07b4969472c8970d0f1c7aa8adeb`). This repo does not
reimplement CogNVS itself — it wraps the upstream model with experiment
orchestration, metric computation, and analysis/visualization tooling so
that reproductions are configurable, repeatable, and testable without a GPU
in the loop for anything except the actual diffusion inference.

## What this repo actually does vs. what it doesn't

This is a hackathon/coursework-scale reproduction, run on a **single
A4000 (16GB)**. That constraint shaped every scoping decision below, and
we've tried to be explicit about it rather than quietly narrowing scope:

| | Status |
|---|---|
| **EXP01 — zero-shot novel-view angle sweep** | Implemented, this is the deliverable |
| **EXP02 — fine-tuning-steps ablation** (FT000/FT050/FT100/FT200 vs. paired GT) | Not implemented — needs ≥5 GPUs for test-time fine-tuning, out of scope for this environment |
| **Top-K probabilistic evaluation** (paper's Fig. 8, right panel) | Evaluation/plotting code exists (`evaluate_topk.py`, `plot_topk.py`), but there is no experiment config or multi-seed generation orchestration wired up yet — see [Next: Top-K experiment](#next-top-k-experiment) |

We'd rather document a gap than paper over it with a metric that doesn't
mean what it looks like it means.

## EXP01: Zero-shot novel-view angle sweep

The experiment we actually run end-to-end. CogNVS generates novel-view
video at four target angles, with **no test-time fine-tuning**
(`fine_tuning_steps: 0` in every config — this is zero-shot by definition)
and **no paired ground truth** (in-the-wild sequences; the alternate camera
angle was never physically captured, matching the qualitative-only setup
of the paper's own Fig. 10).

| Run tag | Angle | Sequences | GT available |
|---|---:|---|---|
| ANGLE015 | 15° | davis_bear, sora_balloon | No |
| ANGLE030 | 30° | davis_bear, sora_balloon | No |
| ANGLE060 | 60° | davis_bear, sora_balloon | No |
| ANGLE090 | 90° | davis_bear, sora_balloon | No |

Because there's no paired GT at the novel angle, PSNR/SSIM/LPIPS are `null`
for every EXP01 row by design — evaluation instead relies on FID/KID
against the source frames as a reference distribution, plus qualitative
side-by-side comparison. See `docs/analysis/evaluation_protocol.md`
("No-GT mode") for the full reasoning.

**Pipeline:**

```
configs/exp01_*.yaml
      │
      ▼
src.experiments.runner (ExperimentRunner)
      │  validates config, creates run directory, calls into inference
      ▼
src.inference.run_cognvs  ──►  upstream cognvs-codebase (data_gen.py, demo.py)
      │  output.mp4 written, frames extracted automatically via video_utils.py
      ▼
src.evaluation.evaluate --no_gt --reference_dir <source_frames>
      │  per-run metrics.json {psnr:null, ssim:null, lpips:null, fid, kid}
      ▼
src.evaluation.aggregator + add_angle.py  ──►  results/final_metrics.csv
      │
      ├──► src.visualization.plots --x_col angle   (metric-vs-angle trend plots)
      └──► scripts/analyze_angle_baseline_exp01.py (FID/KID relative to the
                                                      smallest angle actually run —
                                                      there's no zero-angle
                                                      baseline for this experiment)
```

**Run it:**

```bash
# Dry run — verifies config validation and output layout, no GPU needed
python -m src.experiments.runner --config configs/exp01_davis_bear_angle030.yaml --dry_run

# Real run (needs the upstream cognvs-codebase + checkpoints, see docs/setup.md)
python -m src.experiments.runner --config configs/exp01_davis_bear_angle030.yaml

# Evaluate
python -m src.evaluation.evaluate --gen_dir results/.../frames --no_gt \
    --reference_dir <source_frames> --out results/metrics/ANGLE030.json --tag ANGLE030

# Aggregate + tag + plot
python -m src.evaluation.aggregator ...
python -m src.evaluation.add_angle ...
python -m src.visualization.plots --csv results/final_metrics.csv --out_dir results/plots --x_col angle
```

All `src/` scripts are invoked as modules (`python -m src.<package>.<module>`)
from the repo root — this is the one invocation style guaranteed to resolve
imports correctly (see the fix history below for why this matters in
practice, not just in theory).

## EXP02: Fine-tuning-steps ablation (not implemented)

An earlier draft of our planning docs called this "EXP01" — a
fine-tuning-steps ablation (FT000/FT050/FT100/FT200) evaluated against
paired ground truth on a synthetic dataset (Kubric-4D / ParallelDomain-4D
style). That's a real, valid experiment from the paper, but it requires
**test-time fine-tuning on ≥5 GPUs** per the upstream README, which we
don't have. To avoid two experiments sharing an ambiguous ID once we
realized the angle sweep was the one we could actually deliver, we renamed
the fine-tuning ablation to **EXP02** and marked it explicitly out of
scope, rather than deleting it and losing the design work.

The tag format, config field (`fine_tuning_steps`), and the FT000-baseline
comparison logic in `scripts/analyze_baseline_exp01.py` were all built for
EXP02 ahead of time and are ready to be pointed at real fine-tuning runs if
GPU budget allows later — they currently refuse to run against EXP01
angle-sweep data (and vice versa) rather than silently producing a
mislabeled result. See `docs/p2_analysis_workflow.md` and `docs/team.md`
for the full history of the rename.

### Files that reference EXP02 as the fine-tuning experiment

| File | What it says about EXP02 |
|---|---|
| `docs/team.md` | Ownership doc: lists EXP02 as "not implemented, future work" alongside EXP01 |
| `docs/p2_analysis_workflow.md` | Full section "`EXP02 (fine-tuning-steps ablation) — not yet implemented`": explains the rename from the original ambiguous "EXP01", the FT000/FT050/FT100/FT200 tag table, and the GPU blocker |
| `scripts/analyze_baseline_exp01.py` | Refuses to run against angle-sweep CSVs with a message naming EXP02 as the experiment it *is* built for |
| `scripts/analyze_angle_baseline_exp01.py` | The inverse guard: refuses to run against fine-tuning-steps CSVs, naming EXP02 as the format it's rejecting |
| `issue6-fix.patch` | The patch that added the above cross-guards and the EXP02 section to `p2_analysis_workflow.md` |

Related but *not* EXP02-specific (they support the `fine_tuning_steps`
config field / FT### tag format in general, which EXP02 will use once
implemented, but don't name EXP02 directly):
`src/evaluation/add_steps.py`, `src/evaluation/aggregator.py`,
`src/visualization/plots.py`, `src/experiments/metadata.py`,
`scripts/validate_experiment.py`, `docs/analysis/evaluation_protocol.md`,
`docs/setup.md` (pre-finetuned checkpoint download instructions), and the
config/test files that carry a `fine_tuning_steps: 0` field.

## Next: Top-K experiment

The paper's Fig. 8 (right panel) reports a **Best-of-K** result: generate K
samples of the same condition with different seeds, pool their frames, and
compute FID/KID against the reference distribution as K grows (1, 2, 4, 8).
This is a genuinely different axis from both EXP01 (angle) and EXP02
(fine-tuning steps) — it's about sampling diversity, not the generation
condition itself — and it's the direction we're planning to pursue next
since it doesn't require the ≥5-GPU fine-tuning setup EXP02 needs.

**What's already in place:**
- `src/evaluation/evaluate_topk.py` — pools frames from `sample_00 ...
  sample_0(K-1)/generated/` folders and computes FID/KID per K value. No
  paired GT, same reasoning as EXP01's no-GT mode.
- `src/visualization/plot_topk.py` — plotting counterpart.
- `tests/test_evaluate_topk.py` — unit coverage for the pooling/K-selection
  logic.
- `ExperimentRunner`/`run_cognvs.py` already accept a `seed` config field,
  explicitly documented as "required for Best-of-K runs to actually
  produce K different outputs" (see `src/experiments/runner.py`).

**What's still missing before this is a runnable experiment, not just an
evaluator:**
- No experiment config (`configs/topk_*.yaml`) or run-ID/tag convention for
  top-K sweeps.
- No orchestration to launch K seeded runs of the same sequence/angle and
  land them in the `sample_00/generated, sample_01/generated, ...` layout
  `evaluate_topk.py` expects — right now that folder structure would have
  to be assembled by hand from K separate `ExperimentRunner` runs.
- No baseline-relative analysis script analogous to
  `analyze_angle_baseline_exp01.py` (e.g., FID/KID at K vs. K=1).
- No documentation section in `docs/p2_analysis_workflow.md` or
  `docs/team.md` for this experiment yet (both currently only describe
  EXP01 and EXP02).

Practically: add a `configs/topk_*.yaml` schema (sequence, angle, K, list
or range of seeds), a thin orchestration script that loops
`ExperimentRunner` over seeds and renames each run's `frames/` into
`sample_XX/generated/`, then wire the result into `evaluate_topk.py` and
`plot_topk.py`, which are otherwise ready to consume it.

## Repository layout

```
configs/                   EXP01 YAML configs + camera trajectory files
src/
  inference/                run_cognvs.py — drives the upstream cognvs-codebase
  experiments/              ExperimentRunner, output layout, metadata persistence
  evaluation/                evaluate.py, metrics.py, fid_kid.py, aggregator.py,
                              add_angle.py, add_steps.py, video_utils.py,
                              evaluate_topk.py
  visualization/             plots.py, comparison.py, plot_topk.py
scripts/                    check_environment.py, validate_experiment.py,
                             analyze_baseline_exp01.py (EXP02),
                             analyze_angle_baseline_exp01.py (EXP01)
docs/                       setup.md, team.md, p2_analysis_workflow.md,
                             analysis/metrics_guide.md, analysis/evaluation_protocol.md
tests/                      71 tests across 12 files (pytest, no GPU required)
issue*-fix.patch            Applied fix history — see below
```

## Team ownership

Split three ways to keep GPU-bound work isolated from GPU-free tooling:

- **P1 — CogNVS core + GPU**: `src/inference/`, GPU execution, upstream
  integration, runtime profiling.
- **P2 — Experiment engineering**: `src/experiments/`, `configs/`,
  experiment validation and orchestration.
- **P3 — Evaluation + analysis**: `src/evaluation/`, `src/visualization/`,
  metrics, plots.

Full detail in `docs/team.md`.

## Environment setup

Three separate environments, split so that config validation and the test
suite don't require pulling in GPU/ML dependencies:

| Environment | Covers | Requirements file | GPU needed? |
|---|---|---|---|
| Upstream CogNVS | Actual inference (`data_gen.py`, `demo.py`) | `cognvs_requirements.txt` (in the sibling `cognvs-codebase` folder) | Yes |
| Orchestration tooling | `src/experiments/`, validators, test suite | `requirements.txt` | No |
| Evaluation & visualization | `src/evaluation/`, `src/visualization/` | `requirements-eval.txt` | No (CPU works, slower) |

Full clone/checkpoint instructions, including the pinned upstream commit
and the pre-fine-tuned checkpoints we substitute for actual test-time
fine-tuning, are in `docs/setup.md`. Verify a working environment with:

```bash
python scripts/check_environment.py
```

## Metrics

PSNR, SSIM, LPIPS (paired, pixel/perceptual fidelity — `null` wherever no
GT exists), FID and KID (distributional, used for all no-GT comparisons),
and masked variants (mPSNR/mSSIM/mLPIPS, isolating visible-region fidelity
from hallucinated-region quality, per the paper's Table 8). Full
explanations and when to trust which metric: `docs/analysis/metrics_guide.md`.

## Testing

```bash
pip install -r requirements.txt
pytest tests/
```

71 tests across 12 files, all GPU-free — they exercise config validation,
output-layout logic, tag-format guards (EXP01 vs. EXP02 CSV rejection),
frame extraction, seed wiring, and the top-K pooling logic, without ever
touching the upstream codebase or a checkpoint.

## Fix history

This reproduction went through several rounds of debugging real
integration problems rather than being written once and left alone. The
`issue*-fix.patch` files are kept in the repo root as an honest record of
what broke and why, rather than being squashed away:

| Patch | What it fixed |
|---|---|
| `issue3-fix.patch` | Documented that frame extraction from `output.mp4` now happens automatically (`video_utils.extract_frames`), removing a manual step `evaluate.py` used to require |
| `issue4-fix.patch` | Standardized the `python -m src.<package>.<module>` invocation convention after `evaluate.py`'s imports broke when run the same way as every other script |
| `issue5-fix.patch` | Documented the three-environment split (upstream / orchestration / evaluation) so contributors stop installing the whole ML stack just to run `pytest` |
| `issue6-fix.patch` | Added cross-guards so `analyze_baseline_exp01.py` (EXP02) and `analyze_angle_baseline_exp01.py` (EXP01) each refuse the other experiment's CSV format instead of producing a silently mislabeled result |
| `issue7-fix.patch` | Added integration tests covering config validation *and* `ExperimentRunner.prepare()` together — previously tested in isolation, so a config that passed validation but broke the output layout would have gone unnoticed |
| `issue8-fix.patch` | Added `angle_deg` to required-field and value validation in `scripts/validate_experiment.py` (previously only `fine_tuning_steps` was value-checked) |
| `issue9-fix.patch` | Clarified `comparison.py`'s no-GT usage (`--no_gt --reference_dir`) in the evaluation protocol doc, since the paired mode's `--gt_dir` requirement isn't satisfiable for in-the-wild sequences |
| `issue10-fix.patch` | Wired `seed` through `ExperimentRunner`/`run_cognvs.py` and documented it as required for Best-of-K runs to actually diverge — the groundwork the Top-K experiment above depends on |

## Attribution

CogNVS itself is the work of the original authors
([code](https://github.com/Kaihua-Chen/cog-nvs)). This repository only adds
reproduction scaffolding — experiment orchestration, evaluation, and
analysis — around their model and checkpoints. The upstream codebase
carries no license (all rights reserved) and is not redistributed here; it
must be cloned separately per `docs/setup.md`.
