# P2 Experiment Analysis Workflow

## Purpose

P2 owns experiment configuration, validation, metadata persistence, run
organization, and baseline-relative experiment analysis.

P3 owns metric computation, raw metric aggregation, and visualization.

## Invocation convention

Every script under `src/` is run as a module, from the repo root:

```
python -m src.<package>.<module> [args]
```

e.g. `python -m src.experiments.runner --config <path>`,
`python -m src.evaluation.evaluate --gen_dir <dir> --no_gt --out <path> --tag <tag>`,
`python -m src.visualization.plots --csv <path> --out_dir <dir> --x_col angle`.

This is the one invocation style that's guaranteed to resolve every script's
imports correctly (`evaluate.py`'s `metrics`/`fid_kid` imports previously only
worked if run from inside `src/evaluation/`, which broke as soon as it was
invoked the same way as every other script — see the fix history). Top-level
utility scripts in `scripts/` (not part of the `src` package) are still run
directly, e.g. `python scripts/check_environment.py`.

---

## EXP01 Conditions

EXP01 is the **zero-shot novel-view angle sweep** — the experiment this repo
actually implements and runs end-to-end (`src/inference/run_cognvs.py`,
wired through `ExperimentRunner` in `src/experiments/runner.py`). It runs
CogNVS with no test-time fine-tuning (`fine_tuning_steps: 0` in every EXP01
config) across four target angles, on in-the-wild sequences with no paired
ground truth at the novel angle:

| Run tag | Angle | Sequence examples | GT available |
|---|---:|---|---|
| ANGLE015 | 15° | davis_bear, sora_balloon | No |
| ANGLE030 | 30° | davis_bear, sora_balloon | No |
| ANGLE060 | 60° | davis_bear, sora_balloon | No |
| ANGLE090 | 90° | davis_bear, sora_balloon | No |

Because there is no GT, EXP01 evaluation is FID/KID-vs-reference plus
qualitative comparison — see `docs/analysis/evaluation_protocol.md`'s
"No-GT mode" section. The angle-sweep analysis path is:

```
aggregator.py + add_angle.py -> final_metrics.csv (has an 'angle' column)
    ├── plots.py --x_col angle          -> metric-vs-angle trend plots
    └── scripts/analyze_angle_baseline_exp01.py
                                         -> FID/KID relative to the
                                            smallest angle actually run
                                            (no FT000-equivalent baseline
                                            exists for this experiment)
```

Use `add_angle.py`, not `add_steps.py`, to tag EXP01 metrics — both now
reject being pointed at the other experiment's tag format instead of
silently mislabeling data (e.g. `add_steps.py` used to read `ANGLE030` as
`steps=30` via a generic trailing-digit fallback; that fallback is gone).
`analyze_baseline_exp01.py` similarly refuses angle-sweep CSVs (missing
`steps`/`FT000`) with a message pointing at
`scripts/analyze_angle_baseline_exp01.py` instead of failing on a
confusing missing-column error. PSNR/SSIM/LPIPS are never compared for
EXP01 (they're `null` in every row, no GT to pair against) — only
FID/KID.

### EXP02 (fine-tuning-steps ablation) — not yet implemented

An earlier draft of this doc described a different "EXP01": a
fine-tuning-steps ablation (FT000/FT050/FT100/FT200 against paired ground
truth on a synthetic dataset). That experiment is real future work but is
**not what's built or run in this repo** — the assignment scope is
zero-shot inference plus one experiment, which is the angle sweep above.
To avoid two experiments sharing an ambiguous ID, that ablation has been
renamed **EXP02** and is out of scope for now (it also needs test-time
fine-tuning, which requires ≥5 GPUs per `docs/setup.md` — unavailable in
this environment). If EXP02 is implemented later, it keeps the FT### tag
format and `analyze_baseline_exp01.py`'s FT000-baseline logic, which was
built for it and refuses to run against EXP01 angle-sweep data (see
`docs/analysis/evaluation_protocol.md`).

| Run | Fine-tuning steps | Role |
|---|---:|---|
| FT000 | 0 | Baseline |
| FT050 | 50 | Fine-tuned condition |
| FT100 | 100 | Fine-tuned condition |
| FT200 | 200 | Fine-tuned condition |

---

## Pipeline Ownership

```text
P1
│
├── Model execution
└── Generated frames / ground-truth paths

P3
│
├── evaluate.py
├── metrics.json
├── aggregator.py
├── final_metrics.csv
├── plots.py
└── comparison.py

P2
│
├── Experiment configuration
├── Validation
├── Metadata persistence
├── Run organization
└── Baseline-relative analysis
        │
        └── relative_to_FT000.csv