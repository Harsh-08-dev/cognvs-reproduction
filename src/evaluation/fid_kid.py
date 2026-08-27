"""
src/evaluation/fid_kid.py
FID and KID computed over a batch of generated vs real frames (distribution-level,
NOT paired per-frame like PSNR/SSIM/LPIPS).
"""
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


def _load_batch_as_uint8_tensor(frame_list):
    """
    frame_list: list of HxWx3 uint8 numpy arrays, all same size.
    Returns: Nx3xHxW uint8 tensor (torchmetrics FID/KID expect uint8, range [0,255]).
    """
    tensors = []
    for f in frame_list:
        t = torch.from_numpy(f).permute(2, 0, 1)  # 3xHxW
        tensors.append(t)
    return torch.stack(tensors)  # Nx3xHxW


def compute_fid(gen_frames, gt_frames, device="cpu"):
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.update(_load_batch_as_uint8_tensor(gt_frames).to(device), real=True)
    fid.update(_load_batch_as_uint8_tensor(gen_frames).to(device), real=False)
    return float(fid.compute().item())


def compute_kid(gen_frames, gt_frames, device="cpu", subset_size=None):
    n = min(len(gen_frames), len(gt_frames))
    # KID requires subset_size <= number of samples; default to n if small
    subset_size = subset_size or min(50, n)
    if subset_size < 2:
        raise ValueError("Need at least 2 frames per set to compute KID.")
    kid = KernelInceptionDistance(feature=2048, subset_size=subset_size, normalize=False).to(device)
    kid.update(_load_batch_as_uint8_tensor(gt_frames).to(device), real=True)
    kid.update(_load_batch_as_uint8_tensor(gen_frames).to(device), real=False)
    mean, std = kid.compute()
    return float(mean.item()), float(std.item())