"""
Covers issue #5: requirements.txt only declared P2's orchestration deps
(pyyaml, pytest) and said so in its own header, while src/evaluation/'s and
src/visualization/'s modules need opencv-python, numpy, torch, lpips,
scikit-image, torchmetrics, pandas, and matplotlib — none of which were
declared anywhere in this repo.

This test only runs meaningfully in an environment where
requirements-eval.txt has actually been installed (`pip install -r
requirements-eval.txt`); skip cleanly otherwise rather than failing the
whole suite for people who only set up the lightweight orchestration env.
"""
import importlib

import pytest

EVAL_STACK_MODULES = [
    "cv2",
    "numpy",
    "torch",
    "lpips",
    "skimage.metrics",
    "torchmetrics.image.fid",
    "torchmetrics.image.kid",
    "pandas",
    "matplotlib.pyplot",
]


def _eval_stack_installed():
    for mod in EVAL_STACK_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            return False
    return True


@pytest.mark.skipif(
    not _eval_stack_installed(),
    reason="requirements-eval.txt not installed in this environment",
)
def test_evaluation_and_visualization_modules_import_cleanly():
    """Every module that src/evaluation/ and src/visualization/ actually
    ship must import without error once requirements-eval.txt is installed
    — no reliance on being run from inside src/evaluation/ (see issue #4),
    and no dependency missing from requirements-eval.txt."""
    for module_name in (
        "src.evaluation.metrics",
        "src.evaluation.fid_kid",
        "src.evaluation.evaluate",
        "src.evaluation.aggregator",
        "src.evaluation.add_angle",
        "src.evaluation.add_steps",
        "src.evaluation.video_utils",
        "src.visualization.plots",
        "src.visualization.comparison",
    ):
        importlib.import_module(module_name)
