"""
Covers issue #7 (part 3): tests/test_evaluate_imports.py (issue #4) only
exercises evaluate.py's --no_gt / no --reference_dir path, to prove the
import fix works. It never exercises the paired ground-truth branch, the
--no_gt + --reference_dir branch (which DOES compute FID/KID), the
frame-count-mismatch guard, or the "--gt_dir required unless --no_gt"
guard. This file covers those, using the same sys.modules stub for
torch/lpips/skimage/torchmetrics as test_evaluate_imports.py, so it runs
without requirements-eval.txt installed — but then replaces the imported
compute_* functions with small deterministic fakes, so what's under test
is evaluate.py's own branching/aggregation logic, not the real metric
math (that's covered separately by test_eval_dependencies.py once the
real eval stack is installed).
"""
import json
import sys
import types
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest


def _stub_heavy_deps(monkeypatch):
    """Same stand-ins as tests/test_evaluate_imports.py, so metrics.py/
    fid_kid.py import cleanly without torch/lpips/skimage/torchmetrics
    actually installed."""
    torch_mod = MagicMock()
    torch_mod.no_grad.return_value.__enter__ = MagicMock()
    torch_mod.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "lpips", MagicMock())

    skimage_mod = types.ModuleType("skimage")
    skimage_metrics_mod = MagicMock()
    monkeypatch.setitem(sys.modules, "skimage", skimage_mod)
    monkeypatch.setitem(sys.modules, "skimage.metrics", skimage_metrics_mod)

    torchmetrics_mod = types.ModuleType("torchmetrics")
    torchmetrics_image_mod = types.ModuleType("torchmetrics.image")
    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics_mod)
    monkeypatch.setitem(sys.modules, "torchmetrics.image", torchmetrics_image_mod)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.fid", MagicMock())
    monkeypatch.setitem(sys.modules, "torchmetrics.image.kid", MagicMock())


def _fresh_import_evaluate(monkeypatch):
    _stub_heavy_deps(monkeypatch)
    for name in ("src.evaluation.metrics", "src.evaluation.fid_kid", "src.evaluation.evaluate"):
        sys.modules.pop(name, None)
    import importlib
    return importlib.import_module("src.evaluation.evaluate")


def _write_frames(dir_path, num_frames, fill=0):
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        cv2.imwrite(str(dir_path / f"{i:04d}.png"), np.full((8, 8, 3), fill, dtype=np.uint8))


@pytest.fixture
def evaluate_module(monkeypatch):
    module = _fresh_import_evaluate(monkeypatch)

    # Deterministic fakes standing in for the real metric math, so these
    # tests check evaluate.py's control flow, not skimage/torchmetrics.
    monkeypatch.setattr(module, "compute_psnr", lambda gen, gt: 30.0)
    monkeypatch.setattr(module, "compute_ssim", lambda gen, gt: 0.9)
    monkeypatch.setattr(module, "compute_lpips", lambda gen, gt, device="cpu": 0.1)
    monkeypatch.setattr(module, "compute_fid", lambda gen, ref, device="cpu": 12.5)
    monkeypatch.setattr(module, "compute_kid", lambda gen, ref, device="cpu": (0.01, 0.002))

    return module


# -- paired ground-truth branch ---------------------------------------------

def test_paired_gt_branch_computes_all_metrics(tmp_path, monkeypatch, evaluate_module):
    gen_dir = tmp_path / "gen"
    gt_dir = tmp_path / "gt"
    _write_frames(gen_dir, 3)
    _write_frames(gt_dir, 3)
    out_path = tmp_path / "metrics.json"

    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--gen_dir", str(gen_dir),
        "--gt_dir", str(gt_dir),
        "--out", str(out_path),
        "--tag", "FT100",
    ])

    evaluate_module.main()

    results = json.loads(out_path.read_text())
    assert results["tag"] == "FT100"
    assert results["num_frames"] == 3
    assert results["psnr"] == pytest.approx(30.0)
    assert results["ssim"] == pytest.approx(0.9)
    assert results["lpips"] == pytest.approx(0.1)
    assert results["fid"] == pytest.approx(12.5)
    assert results["kid_mean"] == pytest.approx(0.01)
    assert results["kid_std"] == pytest.approx(0.002)


def test_paired_gt_branch_rejects_frame_count_mismatch(tmp_path, monkeypatch, evaluate_module):
    gen_dir = tmp_path / "gen"
    gt_dir = tmp_path / "gt"
    _write_frames(gen_dir, 3)
    _write_frames(gt_dir, 2)  # deliberately mismatched
    out_path = tmp_path / "metrics.json"

    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--gen_dir", str(gen_dir),
        "--gt_dir", str(gt_dir),
        "--out", str(out_path),
        "--tag", "FT100",
    ])

    with pytest.raises(AssertionError, match="Frame count mismatch"):
        evaluate_module.main()

    assert not out_path.exists()


def test_paired_gt_branch_handles_kid_skip_for_too_few_frames(tmp_path, monkeypatch, evaluate_module):
    """compute_kid raising ValueError (e.g. <2 frames) must be caught and
    recorded as a null pair, not crash the whole run."""
    def raising_kid(gen, ref, device="cpu"):
        raise ValueError("Need at least 2 frames per set to compute KID.")

    monkeypatch.setattr(evaluate_module, "compute_kid", raising_kid)

    gen_dir = tmp_path / "gen"
    gt_dir = tmp_path / "gt"
    _write_frames(gen_dir, 1)
    _write_frames(gt_dir, 1)
    out_path = tmp_path / "metrics.json"

    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--gen_dir", str(gen_dir),
        "--gt_dir", str(gt_dir),
        "--out", str(out_path),
        "--tag", "FT000",
    ])

    evaluate_module.main()

    results = json.loads(out_path.read_text())
    assert results["kid_mean"] is None
    assert results["kid_std"] is None
    assert results["fid"] == pytest.approx(12.5)  # FID still computed


def test_missing_gt_dir_without_no_gt_raises(tmp_path, monkeypatch, evaluate_module):
    gen_dir = tmp_path / "gen"
    _write_frames(gen_dir, 3)
    out_path = tmp_path / "metrics.json"

    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--gen_dir", str(gen_dir),
        "--out", str(out_path),
        "--tag", "FT100",
        # neither --gt_dir nor --no_gt given
    ])

    with pytest.raises(ValueError, match="--gt_dir is required unless --no_gt is set"):
        evaluate_module.main()


# -- --no_gt branch WITH --reference_dir (FID/KID computed, unlike the
# --no_gt-without-reference_dir case test_evaluate_imports.py covers) ------

def test_no_gt_branch_with_reference_dir_computes_fid_kid_but_not_paired_metrics(
    tmp_path, monkeypatch, evaluate_module
):
    gen_dir = tmp_path / "gen"
    reference_dir = tmp_path / "reference"
    _write_frames(gen_dir, 4)
    _write_frames(reference_dir, 6)  # doesn't need to match gen count — unpaired
    out_path = tmp_path / "metrics.json"

    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--gen_dir", str(gen_dir),
        "--no_gt",
        "--reference_dir", str(reference_dir),
        "--out", str(out_path),
        "--tag", "ANGLE060",
    ])

    evaluate_module.main()

    results = json.loads(out_path.read_text())
    assert results["tag"] == "ANGLE060"
    assert results["num_frames"] == 4
    # no paired ground truth -> these stay null
    assert results["psnr"] is None
    assert results["ssim"] is None
    assert results["lpips"] is None
    # but FID/KID ARE computed against the reference distribution
    assert results["fid"] == pytest.approx(12.5)
    assert results["kid_mean"] == pytest.approx(0.01)
    assert results["kid_std"] == pytest.approx(0.002)


def test_no_gt_branch_without_reference_dir_skips_fid_kid_too(
    tmp_path, monkeypatch, evaluate_module
):
    """Sanity check alongside test_evaluate_imports.py's equivalent test:
    with neither GT nor a reference distribution, every metric is null and
    only num_frames/tag are meaningful."""
    gen_dir = tmp_path / "gen"
    _write_frames(gen_dir, 3)
    out_path = tmp_path / "metrics.json"

    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--gen_dir", str(gen_dir),
        "--no_gt",
        "--out", str(out_path),
        "--tag", "ANGLE090",
    ])

    evaluate_module.main()

    results = json.loads(out_path.read_text())
    for key in ("psnr", "ssim", "lpips", "fid", "kid_mean", "kid_std"):
        assert results[key] is None
