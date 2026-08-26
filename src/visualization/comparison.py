"""
src/visualization/comparison.py

Creates GT | Generated | Diff comparison images, matching the paper's Fig. 4/11/12 style.

Usage (run from repo root, like every other script in this repo):
    python -m src.visualization.comparison \
        --gen_dir results/raw_outputs/EXP01_100/generated \
        --gt_dir results/raw_outputs/EXP01_100/gt \
        --out_dir results/comparisons/EXP01_100
"""
import argparse
import os
import cv2
import numpy as np


def make_comparison_row(gen_img, gt_img):
    if gen_img.shape != gt_img.shape:
        gen_img = cv2.resize(gen_img, (gt_img.shape[1], gt_img.shape[0]))
    diff = cv2.absdiff(gen_img, gt_img)
    diff_heat = cv2.applyColorMap(diff.mean(axis=2).astype(np.uint8), cv2.COLORMAP_JET)
    row = np.concatenate([gt_img, gen_img, diff_heat], axis=1)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gen_files = sorted(os.listdir(args.gen_dir))
    gt_files = sorted(os.listdir(args.gt_dir))

    for gf, tf in zip(gen_files, gt_files):
        gen_img = cv2.imread(os.path.join(args.gen_dir, gf))
        gt_img = cv2.imread(os.path.join(args.gt_dir, tf))
        row = make_comparison_row(gen_img, gt_img)
        cv2.imwrite(os.path.join(args.out_dir, f"compare_{gf}"), row)

    print(f"Saved {len(gen_files)} comparison images to {args.out_dir}")


if __name__ == "__main__":
    main()