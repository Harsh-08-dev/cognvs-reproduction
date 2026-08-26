# Metrics Guide (CogNVS reproduction)

## PSNR (Peak Signal-to-Noise Ratio)
- Pixel-level reconstruction fidelity.
- Higher = Better
- Sensitive to pixel misalignment; penalizes blur less than you'd think, penalizes
  small pixel shifts a lot.

## SSIM (Structural Similarity Index)
- Measures structural/luminance/contrast similarity in local windows.
- Higher = Better
- More perceptually aligned than raw PSNR but still a "shallow" metric.

## LPIPS (Learned Perceptual Image Patch Similarity)
- Distance between deep features of a pretrained network (AlexNet/VGG backbone).
- Lower = Better
- Correlates much better with human perception than PSNR/SSIM.

## FID (Fréchet Inception Distance)
- Distance between the *distribution* of generated frames and real frames in
  Inception-v3 feature space. Not a per-frame paired metric — needs a batch of
  generated images and a batch of real images.
- Lower = Better

## KID (Kernel Inception Distance)
- Like FID but uses a polynomial kernel MMD instead of Gaussian assumption —
  more reliable with small sample sizes (which we will have, since we only
  have a handful of test videos).
- Lower = Better

## Masked variants (mPSNR / mSSIM / mLPIPS)
- Paper also reports these (Table 8, Appendix B) — computed ONLY on pixels that
  are co-visible / valid (i.e. not counting hallucinated/inpainted regions).
- Use these if we want to isolate "did the visible-region reconstruction stay
  faithful" from "did the hallucinated region look good."