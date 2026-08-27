"""
Covers issue #8 (part 2): comparison.py only supported a paired-GT mode
(--gt_dir required), but docs/analysis/evaluation_protocol.md calls it the
"primary evidence" for EXP01 — the zero-shot angle sweep, which by
definition has no ground truth. Without a --no_gt mode, the tool
documented as EXP01's primary evidence could not actually be run on
EXP01 data at all.

Skips cleanly if requirements-eval.txt (cv2, numpy) isn't installed, same
pattern as tests/test_eval_dependencies.py.
"""
import importlib

import pytest

EVAL_STACK_MODULES = ["cv2", "numpy"]


def _eval_stack_installed():
    for mod in EVAL_STACK_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            return False
    return True


pytestmark = pytest.mark.skipif(
    not _eval_stack_installed(),
    reason="requirements-eval.txt not installed in this environment",
)


def make_frames(dir_path, num_frames, size=(8, 8), fill_start=0):
    import cv2
    import numpy as np

    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        img = np.full((size[1], size[0], 3), fill_start + i * 10, dtype=np.uint8)
        cv2.imwrite(str(dir_path / f"{i:04d}.png"), img)


def test_no_gt_mode_produces_one_output_per_generated_frame(tmp_path):
    from src.visualization.comparison import run_no_gt

    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"
    make_frames(gen_dir, num_frames=4)
    make_frames(ref_dir, num_frames=1, fill_start=200)

    count = run_no_gt(str(gen_dir), str(ref_dir), str(out_dir))

    assert count == 4
    assert len(list(out_dir.glob("compare_*.png"))) == 4


def test_no_gt_mode_raises_on_empty_reference_dir(tmp_path):
    from src.visualization.comparison import run_no_gt

    gen_dir = tmp_path / "gen"
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"
    make_frames(gen_dir, num_frames=2)
    ref_dir.mkdir(parents=True)  # empty, no reference frames

    with pytest.raises(ValueError, match="No reference frames"):
        run_no_gt(str(gen_dir), str(ref_dir), str(out_dir))


def test_paired_mode_still_works_with_matching_gt(tmp_path):
    from src.visualization.comparison import run_paired

    gen_dir = tmp_path / "gen"
    gt_dir = tmp_path / "gt"
    out_dir = tmp_path / "out"
    make_frames(gen_dir, num_frames=3)
    make_frames(gt_dir, num_frames=3, fill_start=100)

    count = run_paired(str(gen_dir), str(gt_dir), str(out_dir))

    assert count == 3
    assert len(list(out_dir.glob("compare_*.png"))) == 3


def test_paired_mode_raises_clearly_on_frame_count_mismatch(tmp_path):
    """Previously this silently zipped mismatched frame lists together,
    pairing generated frame N with the wrong GT frame N instead of
    failing — see evaluate.py's equivalent explicit assert for gt_dir."""
    from src.visualization.comparison import run_paired

    gen_dir = tmp_path / "gen"
    gt_dir = tmp_path / "gt"
    out_dir = tmp_path / "out"
    make_frames(gen_dir, num_frames=5)
    make_frames(gt_dir, num_frames=3, fill_start=100)

    with pytest.raises(ValueError, match="Frame count mismatch"):
        run_paired(str(gen_dir), str(gt_dir), str(out_dir))


def test_cli_requires_reference_dir_when_no_gt_set(tmp_path, monkeypatch, capsys):
    import sys
    from src.visualization import comparison

    gen_dir = tmp_path / "gen"
    make_frames(gen_dir, num_frames=1)

    argv = [
        "comparison.py",
        "--gen_dir", str(gen_dir),
        "--no_gt",
        "--out_dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="--reference_dir is required"):
        comparison.main()


def test_cli_requires_gt_dir_when_no_gt_not_set(tmp_path, monkeypatch):
    import sys
    from src.visualization import comparison

    gen_dir = tmp_path / "gen"
    make_frames(gen_dir, num_frames=1)

    argv = [
        "comparison.py",
        "--gen_dir", str(gen_dir),
        "--out_dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match="--gt_dir is required"):
        comparison.main()
