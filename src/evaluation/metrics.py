"""
src/evaluation/metrics.py
Core metric functions used by evaluate.py
"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import lpips

# Load LPIPS network once (module-level, so it's not reloaded every call)
_lpips_net = None

def _get_lpips_net(device="cpu"):
    global _lpips_net
    if _lpips_net is None:
        _lpips_net = lpips.LPIPS(net='alex').to(device)
    return _lpips_net


def compute_psnr(gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
    """
    gen_frame, gt_frame: HxWx3 uint8 arrays, same shape, range [0,255]
    """
    return float(sk_psnr(gt_frame, gen_frame, data_range=255))


def compute_ssim(gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
    """
    HxWx3 uint8 arrays. channel_axis=2 for color images (skimage >= 0.19).
    """
    return float(sk_ssim(gt_frame, gen_frame, channel_axis=2, data_range=255))


def compute_lpips(gen_frame: np.ndarray, gt_frame: np.ndarray, device="cpu") -> float:
    """
    HxWx3 uint8 arrays -> converts to normalized tensors in [-1, 1] as LPIPS expects.
    """
    net = _get_lpips_net(device)

    def to_tensor(img):
        t = torch.from_numpy(img).float() / 127.5 - 1.0   # [-1, 1]
        t = t.permute(2, 0, 1).unsqueeze(0)                # 1x3xHxW
        return t.to(device)

    with torch.no_grad():
        d = net(to_tensor(gen_frame), to_tensor(gt_frame))
    return float(d.item())


def compute_masked_metrics(gen_frame: np.ndarray, gt_frame: np.ndarray, mask: np.ndarray, device="cpu"):
    """
    mask: HxW boolean or {0,1} array, True/1 = valid/co-visible pixel to score.
    Returns dict with mPSNR, mSSIM computed only on masked region.
    (mLPIPS is trickier since LPIPS is a deep-feature metric over the whole image;
    common approximation: zero out non-masked regions in both images before feeding
    to LPIPS. This is what most NVS papers do in practice.)
    """
    mask3 = np.stack([mask, mask, mask], axis=-1)
    gen_masked = (gen_frame * mask3).astype(np.uint8)
    gt_masked = (gt_frame * mask3).astype(np.uint8)

    return {
        "mpsnr": compute_psnr(gen_masked, gt_masked),
        "mssim": compute_ssim(gen_masked, gt_masked),
        "mlpips": compute_lpips(gen_masked, gt_masked, device=device),
    }