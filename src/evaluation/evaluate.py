"""
src/evaluation/evaluate.py

Usage (normal, paired ground truth exists):
    python evaluate.py --gen_dir <gen_frames> --gt_dir <gt_frames> --out <out.json> --tag <tag>

Usage (no ground truth exists, e.g. in-the-wild angle sweep beyond captured views):
    python evaluate.py --gen_dir <gen_frames> --no_gt --reference_dir <source_frames> --out <out.json> --tag <tag>
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
    parser.add_argument("--gt_dir", required=False, default=None, help="Folder of ground-truth frames (omit if --no_gt)")
    parser.add_argument("--no_gt", action="store_true", help="Skip PSNR/SSIM/LPIPS — use when no paired GT exists (e.g. in-the-wild angle sweep)")
    parser.add_argument("--reference_dir", required=False, default=None, help="Folder of real reference frames for FID/KID when no paired GT exists (e.g. original source frames)")
    parser.add_argument("--out", required=True, help="Output metrics.json path")
    parser.add_argument("--tag", required=True, help="Name of this experiment/sequence")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    gen_frames, gen_files = load_frames(args.gen_dir)

    results = {"tag": args.tag, "num_frames": len(gen_frames)}

    if args.no_gt:
        # No paired ground truth exists (in-the-wild sequences at novel angles).
        # FID/KID need a reference distribution — use reference_dir if given,
        # otherwise skip FID/KID too and only rely on qualitative comparison.
        results["psnr"] = None
        results["ssim"] = None
        results["lpips"] = None

        if args.reference_dir:
            ref_frames, _ = load_frames(args.reference_dir)
            results["fid"] = compute_fid(gen_frames, ref_frames, device=args.device)
            try:
                kid_mean, kid_std = compute_kid(gen_frames, ref_frames, device=args.device)
            except ValueError as e:
                print(f"[WARN] Skipping KID: {e}")
                kid_mean, kid_std = None, None
            results["kid_mean"] = kid_mean
            results["kid_std"] = kid_std
        else:
            print("[INFO] No --reference_dir given, skipping FID/KID. Only qualitative comparison available.")
            results["fid"] = None
            results["kid_mean"] = None
            results["kid_std"] = None

    else:
        if not args.gt_dir:
            raise ValueError("--gt_dir is required unless --no_gt is set.")

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

        results["psnr"] = float(np.mean(psnr_vals))
        results["ssim"] = float(np.mean(ssim_vals))
        results["lpips"] = float(np.mean(lpips_vals))
        results["fid"] = compute_fid(gen_frames, gt_frames, device=args.device)
        try:
            kid_mean, kid_std = compute_kid(gen_frames, gt_frames, device=args.device)
        except ValueError as e:
            print(f"[WARN] Skipping KID: {e}")
            kid_mean, kid_std = None, None
        results["kid_mean"] = kid_mean
        results["kid_std"] = kid_std

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()