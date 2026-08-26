"""
src/visualization/plots.py

Generates metric-vs-finetuning-steps plots, matching the paper's Fig. 8 style.

Usage:
    python src/visualization/plots.py --csv results/final_metrics.csv --out_dir results/plots
Expects final_metrics.csv to have a 'steps' column (int) and metric columns.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(df, metric, out_dir, higher_better):
    df_sorted = df.sort_values("steps")
    plt.figure(figsize=(6, 4))
    plt.plot(df_sorted["steps"], df_sorted[metric], marker="o")
    plt.xlabel("Fine-tuning steps")
    plt.ylabel(metric.upper())
    arrow = "↑ higher better" if higher_better else "↓ lower better"
    plt.title(f"{metric.upper()} vs Fine-tuning Steps ({arrow})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{metric}_vs_steps.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    metric_directions = {
        "psnr": True, "ssim": True,
        "lpips": False, "fid": False, "kid_mean": False,
    }

    for metric, higher_better in metric_directions.items():
        if metric in df.columns:
            plot_metric(df, metric, args.out_dir, higher_better)

    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()