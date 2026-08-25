# Evaluation Protocol

## Inputs required from P1
- Generated novel-view video (or frame sequence) — `output.mp4` or `frames/*.png`
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