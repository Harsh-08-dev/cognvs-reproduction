"""
src/evaluation/aggregator.py

Combines multiple metrics.json files (e.g. one per finetuning-step checkpoint)
into a single results/final_metrics.csv

Usage (run from repo root, like every other script in this repo):
    python -m src.evaluation.aggregator --metrics_dir results/metrics --out results/final_metrics.csv
"""
import argparse
import json
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for fname in sorted(os.listdir(args.metrics_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(args.metrics_dir, fname)) as f:
            data = json.load(f)
        rows.append(data)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df)


if __name__ == "__main__":
    main()