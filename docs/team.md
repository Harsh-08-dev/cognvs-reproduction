# Team Ownership

## Roles

- P1 — CogNVS Core + GPU
- P2 — Experiment Engineering
- P3 — Evaluation + Analysis

## P1 owns
- src/inference/
- GPU execution
- CogNVS integration
- runtime profiling

## P2 owns
- src/experiments/
- configs/
- scripts/run_experiment.py
- experiment validation

## P3 owns
- src/evaluation/
- src/visualization/
- plots
- metrics
- visualization

## Current experiments
- **EXP01** (implemented, this is what we run for the deliverable): zero-shot
  novel-view angle sweep, no fine-tuning, no paired GT. Driven by
  `src/inference/run_cognvs.py`, orchestrated via `ExperimentRunner`
  (`src/experiments/runner.py`). Details: `docs/p2_analysis_workflow.md`.
- **EXP02** (not implemented, future work): fine-tuning-steps ablation
  (FT000/FT050/FT100/FT200) against paired ground truth. Needs test-time
  fine-tuning (≥5 GPUs), out of scope for this environment.

## Rules
- Don't modify another person's owned folder without discussion.
- Update documentation when changing experiment behavior.