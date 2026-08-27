"""
src/visualization/comparison.py

Creates side-by-side comparison images, matching the paper's Fig. 4/11/12 style.

Two modes:

Paired ground truth exists (e.g. synthetic datasets with a captured GT
novel view):
    python -m src.visualization.comparison \
        --gen_dir results/raw_outputs/EXP01_100/generated \
        --gt_dir results/raw_outputs/EXP01_100/gt \
        --out_dir results/comparisons/EXP01_100
Produces GT | Generated | Diff rows.

No ground truth exists (in-the-wild angle sweep beyond captured views,
e.g. EXP01's davis_bear/sora_balloon runs — see
docs/analysis/evaluation_protocol.md's "No-GT mode" section, which calls
this the primary evidence for that experiment):
    python -m src.visualization.comparison \
        --gen_dir results/EXP01/davis_bear/angle_030/frames \
        --no_gt --reference_dir demo_data/davis_bear/frames \
        --out_dir results/comparisons/EXP01/davis_bear/angle_030
Produces Source | Generated rows (no diff — reference is a different
camera angle, not a pixel-aligned pair, so a diff heatmap would be
meaningless here).
"""
import argparse
import os
import cv2
import numpy as np


def list_images(dir_path):
    return sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def make_paired_comparison_row(gen_img, gt_img):
    """GT | Generated | Diff. Requires a pixel-aligned ground-truth frame."""
    if gen_img.shape != gt_img.shape:
        gen_img = cv2.resize(gen_img, (gt_img.shape[1], gt_img.shape[0]))
    diff = cv2.absdiff(gen_img, gt_img)
    diff_heat = cv2.applyColorMap(diff.mean(axis=2).astype(np.uint8), cv2.COLORMAP_JET)
    return np.concatenate([gt_img, gen_img, diff_heat], axis=1)


def make_reference_comparison_row(gen_img, ref_img):
    """Source | Generated. No diff: the reference is a different camera
    angle, not a pixel-aligned pair, so absdiff would just show that the
    whole frame changed rather than anything meaningful about quality."""
    if gen_img.shape != ref_img.shape:
        gen_img = cv2.resize(gen_img, (ref_img.shape[1], ref_img.shape[0]))
    return np.concatenate([ref_img, gen_img], axis=1)


def run_paired(gen_dir, gt_dir, out_dir):
    gen_files = list_images(gen_dir)
    gt_files = list_images(gt_dir)

    if len(gen_files) != len(gt_files):
        raise ValueError(
            f"Frame count mismatch: {len(gen_files)} generated vs "
            f"{len(gt_files)} GT frames in {gen_dir!r} / {gt_dir!r}. "
            f"Paired comparison requires exactly one GT frame per "
            f"generated frame — check with P1 that frame indices line up."
        )

    os.makedirs(out_dir, exist_ok=True)
    for gf, tf in zip(gen_files, gt_files):
        gen_img = cv2.imread(os.path.join(gen_dir, gf))
        gt_img = cv2.imread(os.path.join(gt_dir, tf))
        row = make_paired_comparison_row(gen_img, gt_img)
        cv2.imwrite(os.path.join(out_dir, f"compare_{gf}"), row)

    return len(gen_files)


def run_no_gt(gen_dir, reference_dir, out_dir):
    gen_files = list_images(gen_dir)
    ref_files = list_images(reference_dir)

    if not ref_files:
        raise ValueError(f"No reference frames found in {reference_dir!r}.")

    # No frame-to-frame pairing exists between a novel-view sweep and the
    # single-camera source video (different angles, not the same moment
    # in time necessarily) — reuse one representative reference frame
    # (the first one) against every generated frame instead of zipping,
    # which would silently pair unrelated frames if counts differ.
    ref_img = cv2.imread(os.path.join(reference_dir, ref_files[0]))

    os.makedirs(out_dir, exist_ok=True)
    for gf in gen_files:
        gen_img = cv2.imread(os.path.join(gen_dir, gf))
        row = make_reference_comparison_row(gen_img, ref_img)
        cv2.imwrite(os.path.join(out_dir, f"compare_{gf}"), row)

    return len(gen_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", required=True, help="Folder of generated frames")
    parser.add_argument("--gt_dir", required=False, default=None,
                         help="Folder of paired ground-truth frames (omit if --no_gt)")
    parser.add_argument("--no_gt", action="store_true",
                         help="Use when no paired GT exists (e.g. in-the-wild angle sweep)")
    parser.add_argument("--reference_dir", required=False, default=None,
                         help="Folder of source/reference frames for --no_gt mode "
                              "(e.g. the original source video frames)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.no_gt:
        if not args.reference_dir:
            raise ValueError("--reference_dir is required when --no_gt is set.")
        count = run_no_gt(args.gen_dir, args.reference_dir, args.out_dir)
    else:
        if not args.gt_dir:
            raise ValueError("--gt_dir is required unless --no_gt is set.")
        count = run_paired(args.gen_dir, args.gt_dir, args.out_dir)

    print(f"Saved {count} comparison images to {args.out_dir}")


if __name__ == "__main__":
    main()
