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

## Rules
- Don't modify another person's owned folder without discussion.
- Update documentation when changing experiment behavior.