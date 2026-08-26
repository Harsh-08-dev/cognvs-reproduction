"""
Covers issue #4: evaluate.py did `from metrics import ...` / `from fid_kid
import ...`, which only resolves when run from inside src/evaluation/ (e.g.
`python evaluate.py ...`). Every other script in the repo is documented and
invoked repo-root-relative (`python -m src.<package>.<module> ...`), so this
broke the standard invocation. Fixed by switching to absolute,
package-relative imports (`from src.evaluation.metrics import ...`).

metrics.py / fid_kid.py pull in torch/lpips/skimage/torchmetrics, which
aren't declared in requirements.txt yet (that's issue #5, not this one). We
stub those heavy modules in sys.modules so these tests verify the import
*path* is correct, independent of whether those ML libs happen to be
installed on a given machine.
"""
import importlib
import json
import sys
import types
from unittest.mock import MagicMock

import cv2
import numpy as np


def _stub_heavy_deps(monkeypatch):
    """Lightweight stand-ins so metrics.py/fid_kid.py import cleanly without
    torch/lpips/skimage/torchmetrics actually installed."""
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


def test_evaluate_imports_metrics_and_fid_kid_as_package_relative(monkeypatch):
    """The actual bug: bare `from metrics import ...` / `from fid_kid import
    ...` don't resolve unless cwd/sys.path is src/evaluation/. This must
    import cleanly the same way every other module in the repo does — via
    its package path, with the repo root on sys.path."""
    _stub_heavy_deps(monkeypatch)
    for name in ("src.evaluation.metrics", "src.evaluation.fid_kid", "src.evaluation.evaluate"):
        sys.modules.pop(name, None)

    evaluate = _fresh_import("src.evaluation.evaluate")

    assert hasattr(evaluate, "main")
    assert hasattr(evaluate, "load_frames")
    # confirm it actually pulled in the real functions, not stubs of its own
    assert evaluate.compute_psnr is not None
    assert evaluate.compute_fid is not None


def test_evaluate_cli_runs_end_to_end_no_gt(tmp_path, monkeypatch):
    """Exercises the fixed import + argparse + load_frames + json-write path
    via the real main() entry point, in --no_gt / no --reference_dir mode
    (which never calls into metrics.py's/fid_kid.py's compute_* functions,
    so this only needs cv2/numpy — already required by evaluate.py itself)."""
    _stub_heavy_deps(monkeypatch)
    for name in ("src.evaluation.metrics", "src.evaluation.fid_kid", "src.evaluation.evaluate"):
        sys.modules.pop(name, None)
    evaluate = _fresh_import("src.evaluation.evaluate")

    gen_dir = tmp_path / "gen_frames"
    gen_dir.mkdir()
    for i in range(3):
        cv2.imwrite(str(gen_dir / f"{i:04d}.png"), np.zeros((8, 8, 3), dtype=np.uint8))

    out_path = tmp_path / "metrics.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py",
        "--gen_dir", str(gen_dir),
        "--no_gt",
        "--out", str(out_path),
        "--tag", "ANGLE030",
    ])

    evaluate.main()

    results = json.loads(out_path.read_text())
    assert results["tag"] == "ANGLE030"
    assert results["num_frames"] == 3
    assert results["psnr"] is None
    assert results["fid"] is None
