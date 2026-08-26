"""
src/evaluation/evaluate.py

Usage:
    python src/evaluation/evaluate.py \
        --gen_dir results/raw_outputs/EXP01_100/generated \
        --gt_dir results/raw_outputs/EXP01_100/gt \
        --out results/metrics/EXP01_100.json \
        --tag EXP01_100
"""
import argparse
import json
import os
import cv2
import numpy as np

from metrics import compute_psnr, compute_ssim, compute_lpips
from fid_kid import compute_fid, compute_kid


def load_frames(frame_dir):
    """Loads all images in a directory, sorted by filename, as RGB uint8 arrays."""
    files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    frames = []
    for f in files:
        img_bgr = cv2.imread(os.path.join(frame_dir, f))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)
    return frames, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True, help="Folder of generated frames")
    parser.add_argument("--gt_dir", required=True, help="Folder of ground-truth frames")
    parser.add_argument("--out", required=True, help="Output metrics.json path")
    parser.add_argument("--tag", required=True, help="Name of this experiment/sequence")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    gen_frames, gen_files = load_frames(args.gen_dir)
    gt_frames, gt_files = load_frames(args.gt_dir)

    assert len(gen_frames) == len(gt_frames), \
        f"Frame count mismatch: {len(gen_frames)} generated vs {len(gt_frames)} GT. Check with P1."

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for gen, gt in zip(gen_frames, gt_frames):
        if gen.shape != gt.shape:
            gen = cv2.resize(gen, (gt.shape[1], gt.shape[0]))
        psnr_vals.append(compute_psnr(gen, gt))
        ssim_vals.append(compute_ssim(gen, gt))
        lpips_vals.append(compute_lpips(gen, gt, device=args.device))

    fid_val = compute_fid(gen_frames, gt_frames, device=args.device)
    try:
        kid_mean, kid_std = compute_kid(gen_frames, gt_frames, device=args.device)
    except ValueError as e:
        print(f"[WARN] Skipping KID: {e}")
        kid_mean, kid_std = None, None

    results = {
        "tag": args.tag,
        "num_frames": len(gen_frames),
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)),
        "fid": fid_val,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()