"""
src/evaluation/evaluate_topk.py

Top-K probabilistic evaluation (paper's Fig. 8, right panel).

Given K generated samples (same condition, different seeds), pools frames
from the first k samples together and computes FID/KID against a reference
distribution, for k = 1, 2, 4, 8 (or whichever subset is available).

No paired ground truth is used here either (same reasoning as the angle-sweep
--no_gt mode) — only FID/KID against the original source frames.

Expected folder structure:
    <samples_dir>/
        sample_00/generated/*.png
        sample_01/generated/*.png
        ...
        sample_07/generated/*.png

Usage:
    python evaluate_topk.py --samples_dir results/EXP01/TOPK_ANGLE030_K8 --reference_dir <source_frames> --out results/metrics/TOPK_ANGLE030.json --tag TOPK_ANGLE030
"""
import argparse
import json
import os
import cv2

from fid_kid import compute_fid, compute_kid


def load_frames(frame_dir):
    files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    frames = []
    for f in files:
        img_bgr = cv2.imread(os.path.join(frame_dir, f))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)
    return frames


def find_sample_dirs(samples_dir):
    """Returns sorted list of sample_XX directories inside samples_dir."""
    entries = sorted([
        d for d in os.listdir(samples_dir)
        if os.path.isdir(os.path.join(samples_dir, d)) and d.startswith("sample_")
    ])
    return [os.path.join(samples_dir, d, "generated") for d in entries]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", required=True, help="Folder containing sample_00, sample_01, ... subfolders")
    parser.add_argument("--reference_dir", required=True, help="Folder of original source frames (real reference distribution)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--k_values", default="1,2,4,8", help="Comma-separated k values to evaluate, e.g. 1,2,4,8")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sample_dirs = find_sample_dirs(args.samples_dir)
    if not sample_dirs:
        raise ValueError(f"No sample_XX/generated folders found in {args.samples_dir}")

    print(f"Found {len(sample_dirs)} samples: {sample_dirs}")

    reference_frames = load_frames(args.reference_dir)

    k_values = [int(k) for k in args.k_values.split(",")]

    results = {"tag": args.tag, "num_samples_available": len(sample_dirs), "k_results": {}}

    for k in k_values:
        if k > len(sample_dirs):
            print(f"[WARN] Skipping k={k} — only {len(sample_dirs)} samples available.")
            continue

        pooled_frames = []
        for sd in sample_dirs[:k]:
            pooled_frames.extend(load_frames(sd))

        fid_val = compute_fid(pooled_frames, reference_frames, device=args.device)
        try:
            kid_mean, kid_std = compute_kid(pooled_frames, reference_frames, device=args.device)
        except ValueError as e:
            print(f"[WARN] Skipping KID for k={k}: {e}")
            kid_mean, kid_std = None, None

        results["k_results"][str(k)] = {
            "fid": fid_val,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
            "num_pooled_frames": len(pooled_frames),
        }
        print(f"k={k}: FID={fid_val:.4f}, KID_mean={kid_mean}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()