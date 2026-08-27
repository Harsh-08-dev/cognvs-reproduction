"""
src/visualization/plots.py

Generates metric-vs-independent-variable plots.
Works for both finetuning-step sweeps AND angle sweeps — just point --x_col
at whichever column your CSV uses ("steps" or "angle").

Usage (run from repo root, like every other script in this repo):
    python -m src.visualization.plots --csv results/final_metrics.csv --out_dir results/plots --x_col angle --x_label "Camera deviation (degrees)"
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df, metric, x_col, x_label, out_dir, higher_better):
    df_clean = df.dropna(subset=[metric])
    if df_clean.empty:
        print(f"[INFO] Skipping {metric} — all values are null (likely no-GT mode).")
        return

    df_sorted = df_clean.sort_values(x_col)
    plt.figure(figsize=(6, 4))
    plt.plot(df_sorted[x_col], df_sorted[metric], marker="o")
    plt.xlabel(x_label)
    plt.ylabel(metric.upper())
    arrow = "↑ higher better" if higher_better else "↓ lower better"
    plt.title(f"{metric.upper()} vs {x_label} ({arrow})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{metric}_vs_{x_col}.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--x_col", default="steps", help="Column name to use as x-axis, e.g. 'steps' or 'angle'")
    parser.add_argument("--x_label", default="Fine-tuning steps", help="Human-readable x-axis label")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    if args.x_col not in df.columns:
        raise ValueError(f"Column '{args.x_col}' not found in CSV. Available columns: {list(df.columns)}")

    metric_directions = {
        "psnr": True, "ssim": True,
        "lpips": False, "fid": False, "kid_mean": False,
    }

    for metric, higher_better in metric_directions.items():
        if metric in df.columns:
            plot_metric(df, metric, args.x_col, args.x_label, args.out_dir, higher_better)

    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()