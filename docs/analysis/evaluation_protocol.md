# Evaluation Protocol

## Inputs required from P1
- Generated novel-view video (or frame sequence) — `output.mp4` or `frames/*.png`.
  As of the inference/evaluation integration fix, `frames/*.png` is produced
  automatically: both `run_cognvs.py`'s `collect_outputs()` and
  `ExperimentRunner.execute()` call `src/evaluation/video_utils.py`'s
  `extract_frames()` on the generated `output.mp4` right after it's written,
  so `evaluate.py --gen_dir <run_dir>/frames` works with no manual
  intermediate step. `video_utils.py` is still runnable standalone for ad-hoc
  video → frames conversion (e.g. re-extracting an existing `output.mp4`).
- Ground-truth novel-view video/frames for the SAME target camera path (if using
  a synthetic dataset like Kubric-4D/ParallelDomain-4D this exists by construction;
  if using a real captured video with no GT novel view, we CANNOT compute
  PSNR/SSIM/LPIPS — only FID/KID against a reference distribution, or qualitative
  comparison only)
- `runtime.json` — how long finetuning + inference took (for Table 3-style runtime
  comparison)
- `config.yaml` — steps used, learning rate, which video/dataset

## Output
- Per-sequence `metrics.json`: {psnr, ssim, lpips, fid, kid}
- `results/final_metrics.csv` — aggregated across sequences/steps
- Comparison images (GT | Generated | Diff)
- Plots: metric vs finetuning steps (matches paper's Fig. 8 style ablation)

## Frame alignment requirement
PSNR/SSIM/LPIPS are PAIRED metrics — generated frame N must correspond exactly
to GT frame N (same camera pose, same timestep). Confirm with P1 that frame
indices line up 1:1 before computing anything.

## No-GT mode (in-the-wild sequences, novel angles beyond captured views)
For sequences like davis_bear/sora_balloon at 15°/30°/60°/90°, no ground truth
exists at the novel angle — these are single-camera in-the-wild videos, and the
alternate angle was never physically captured. This matches the paper's own
Fig. 10, which is qualitative-only for the same reason.

For these runs:
- `psnr`, `ssim`, `lpips` will be `null` in the output JSON (no paired GT to compare).
- `fid`/`kid_mean`/`kid_std` are computed against `--reference_dir` (the original
  source video frames) as the "real" reference distribution.
- Visual/qualitative comparison via `comparison.py` becomes the primary evidence
  for this experiment, alongside FID/KID trends across angles.
- Baseline-relative comparison uses `scripts/analyze_angle_baseline_exp01.py`,
  which compares each angle's FID/KID to the smallest angle actually run
  (there's no zero-angle/FT000-equivalent condition to use instead). Don't
  point `scripts/analyze_baseline_exp01.py` (the FT000-baseline script) at
  this data — it's for the FT###-based fine-tuning-steps ablation and will
  refuse to run against angle-sweep CSVs.