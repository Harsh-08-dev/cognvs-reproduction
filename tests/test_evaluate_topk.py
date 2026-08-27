

"""
Tests for evaluate_topk.py and plot_topk.py — the top-K / Best-of-K sampling
experiment (paper's Fig. 8, right panel; docs/experiment_01.md's bonus
ablation). These two files shipped with zero test coverage, unlike every
other script in src/evaluation and src/visualization — which is how
evaluate_topk.py's bare `from fid_kid import ...` (broken under this repo's
standard `python -m src.<package>.<module>` invocation, same class of bug as
evaluate.py's own fix — see test_evaluate_imports.py) went undetected: nothing
ever actually imported or ran the module.

Mirrors test_evaluate_imports.py's dependency-stubbing approach so these
tests don't require torch/torchmetrics to actually be installed.
"""
import importlib
import json
import sys
import types
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest


def _stub_heavy_deps(monkeypatch):
    """Same lightweight stand-ins as test_evaluate_imports.py, so importing
    src.evaluation.fid_kid (evaluate_topk.py's dependency) doesn't require
    torch/torchmetrics to actually be installed."""
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


def _fresh_import(module_name):
    """(Re-)import a module, discarding any previously cached (possibly
    broken) entry in sys.modules."""
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _fresh_evaluate_topk(monkeypatch):
    _stub_heavy_deps(monkeypatch)
    for name in ("src.evaluation.fid_kid", "src.evaluation.evaluate_topk"):
        sys.modules.pop(name, None)
    return _fresh_import("src.evaluation.evaluate_topk")


def make_sample_frames(base_dir, num_samples, num_frames=3, size=8):
    """Creates sample_00/generated, sample_01/generated, ... under base_dir,
    each with num_frames tiny real PNGs."""
    for k in range(num_samples):
        gen_dir = base_dir / f"sample_{k:02d}" / "generated"
        gen_dir.mkdir(parents=True)
        for i in range(num_frames):
            cv2.imwrite(str(gen_dir / f"{i:04d}.png"), np.zeros((size, size, 3), dtype=np.uint8))


# --- import / interface ---

def test_evaluate_topk_imports_fid_kid_as_package_relative(monkeypatch):
    """The actual bug: evaluate_topk.py had `from fid_kid import ...`, which
    only resolves when run from inside src/evaluation/, breaking under this
    repo's standard `python -m src.<package>.<module>` invocation."""
    evaluate_topk = _fresh_evaluate_topk(monkeypatch)

    assert hasattr(evaluate_topk, "main")
    assert hasattr(evaluate_topk, "find_sample_dirs")
    # confirm it actually pulled in the real functions, not stubs of its own
    assert evaluate_topk.compute_fid is not None
    assert evaluate_topk.compute_kid is not None


# --- find_sample_dirs ---

def test_find_sample_dirs_returns_sorted_generated_paths(tmp_path, monkeypatch):
    evaluate_topk = _fresh_evaluate_topk(monkeypatch)

    make_sample_frames(tmp_path, num_samples=4)

    dirs = evaluate_topk.find_sample_dirs(str(tmp_path))

    assert len(dirs) == 4
    assert all(d.endswith("generated") for d in dirs)
    assert dirs == sorted(dirs)


def test_find_sample_dirs_returns_empty_list_when_no_samples(tmp_path, monkeypatch):
    evaluate_topk = _fresh_evaluate_topk(monkeypatch)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert evaluate_topk.find_sample_dirs(str(empty_dir)) == []


# --- main() end-to-end ---

def test_evaluate_topk_cli_runs_end_to_end(tmp_path, monkeypatch):
    """Exercises the real main() entry point: finds sample_XX dirs, pools
    frames per k, writes k_results to the output JSON. compute_fid/compute_kid
    are monkeypatched to fixed values — this checks the plumbing (pooling,
    k-selection, JSON structure), not metric correctness, which is
    metrics.py's/fid_kid.py's own job."""
    evaluate_topk = _fresh_evaluate_topk(monkeypatch)

    make_sample_frames(tmp_path, num_samples=4)
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    cv2.imwrite(str(ref_dir / "0000.png"), np.zeros((8, 8, 3), dtype=np.uint8))

    monkeypatch.setattr(evaluate_topk, "compute_fid", lambda gen, ref, device="cpu": 12.5)
    monkeypatch.setattr(evaluate_topk, "compute_kid", lambda gen, ref, device="cpu": (0.01, 0.001))

    out_path = tmp_path / "topk_metrics.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_topk.py",
        "--samples_dir", str(tmp_path),
        "--reference_dir", str(ref_dir),
        "--out", str(out_path),
        "--tag", "TOPK_ANGLE030",
        "--k_values", "1,2,4",
    ])

    evaluate_topk.main()

    results = json.loads(out_path.read_text())
    assert results["tag"] == "TOPK_ANGLE030"
    assert results["num_samples_available"] == 4
    assert set(results["k_results"].keys()) == {"1", "2", "4"}
    assert results["k_results"]["1"]["fid"] == 12.5
    assert results["k_results"]["2"]["num_pooled_frames"] == 6   # 2 samples * 3 frames
    assert results["k_results"]["4"]["num_pooled_frames"] == 12  # 4 samples * 3 frames


def test_evaluate_topk_skips_k_larger_than_available_samples(tmp_path, monkeypatch):
    """k=8 requested but only 2 samples exist -> should be skipped with a
    warning, not crash."""
    evaluate_topk = _fresh_evaluate_topk(monkeypatch)

    make_sample_frames(tmp_path, num_samples=2)
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    cv2.imwrite(str(ref_dir / "0000.png"), np.zeros((8, 8, 3), dtype=np.uint8))

    monkeypatch.setattr(evaluate_topk, "compute_fid", lambda gen, ref, device="cpu": 1.0)
    monkeypatch.setattr(evaluate_topk, "compute_kid", lambda gen, ref, device="cpu": (0.0, 0.0))

    out_path = tmp_path / "topk_metrics.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_topk.py",
        "--samples_dir", str(tmp_path),
        "--reference_dir", str(ref_dir),
        "--out", str(out_path),
        "--tag", "TOPK_TEST",
        "--k_values", "1,2,8",
    ])

    evaluate_topk.main()

    results = json.loads(out_path.read_text())
    assert set(results["k_results"].keys()) == {"1", "2"}


def test_main_raises_when_no_samples_found(tmp_path, monkeypatch):
    evaluate_topk = _fresh_evaluate_topk(monkeypatch)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()

    monkeypatch.setattr(sys, "argv", [
        "evaluate_topk.py",
        "--samples_dir", str(empty_dir),
        "--reference_dir", str(ref_dir),
        "--out", str(tmp_path / "out.json"),
        "--tag", "TOPK_EMPTY",
    ])

    with pytest.raises(ValueError):
        evaluate_topk.main()


# --- plot_topk.py ---

def test_plot_topk_produces_fid_and_kid_png_files(tmp_path, monkeypatch):
    """plot_topk.py only needs matplotlib/json/os — no heavy ML deps — so
    no stubbing needed here, just a real (fast) matplotlib render."""
    plot_topk = _fresh_import("src.visualization.plot_topk")

    data = {
        "tag": "TOPK_TEST",
        "k_results": {
            "1": {"fid": 30.0, "kid_mean": 0.02},
            "2": {"fid": 25.0, "kid_mean": 0.01},
            "4": {"fid": 20.0, "kid_mean": 0.005},
        },
    }
    json_path = tmp_path / "topk_metrics.json"
    json_path.write_text(json.dumps(data))

    out_dir = tmp_path / "plots"

    monkeypatch.setattr(sys, "argv", [
        "plot_topk.py",
        "--json", str(json_path),
        "--out_dir", str(out_dir),
    ])

    plot_topk.main()

    assert (out_dir / "TOPK_TEST_fid_vs_k.png").exists()
    assert (out_dir / "TOPK_TEST_kid_vs_k.png").exists()