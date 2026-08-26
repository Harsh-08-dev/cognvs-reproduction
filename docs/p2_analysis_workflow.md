# P2 Experiment Analysis Workflow

## Purpose

P2 owns experiment configuration, validation, metadata persistence, run
organization, and baseline-relative experiment analysis.

P3 owns metric computation, raw metric aggregation, and visualization.

---

## EXP01 Conditions

EXP01 compares four fine-tuning conditions:

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