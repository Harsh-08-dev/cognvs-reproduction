# P2 Experiment Analysis Workflow

## Purpose

P2 owns experiment configuration, validation, metadata persistence, run
organization, and baseline-relative experiment analysis.

P3 owns metric computation, raw metric aggregation, and visualization.

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
"No-GT mode" section, and use `add_angle.py` (not `add_steps.py`) /
`analyze_baseline_exp01.py`'s angle-aware counterpart for aggregation.

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
built for it and should not be pointed at EXP01 angle-sweep data (see
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