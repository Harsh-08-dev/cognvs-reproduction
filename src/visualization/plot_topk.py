"""
src/visualization/plot_topk.py

Plots FID/KID vs. k (number of pooled samples), matching the paper's
Fig. 8 (right panel) style.

Usage:
    python plot_topk.py --json ../evaluation/results/metrics/TOPK_ANGLE030.json --out_dir results/plots
"""
import argparse
import json
import os
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to evaluate_topk.py output JSON")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    k_results = data["k_results"]
    ks = sorted(int(k) for k in k_results.keys())

    fid_vals = [k_results[str(k)]["fid"] for k in ks]
    kid_vals = [k_results[str(k)]["kid_mean"] for k in ks]

    # FID plot
    plt.figure(figsize=(6, 4))
    plt.plot(ks, fid_vals, marker="o")
    plt.xlabel("k (number of pooled samples)")
    plt.ylabel("FID")
    plt.title(f"FID vs k — {data['tag']} (↓ lower better)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{data['tag']}_fid_vs_k.png"), dpi=150)
    plt.close()

    # KID plot
    plt.figure(figsize=(6, 4))
    plt.plot(ks, kid_vals, marker="o", color="darkorange")
    plt.xlabel("k (number of pooled samples)")
    plt.ylabel("KID (mean)")
    plt.title(f"KID vs k — {data['tag']} (↓ lower better)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{data['tag']}_kid_vs_k.png"), dpi=150)
    plt.close()

    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()