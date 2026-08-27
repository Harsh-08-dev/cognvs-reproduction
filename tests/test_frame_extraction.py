"""
Covers issue #3: the inference stage must automatically produce the
frames/ folder that evaluate.py --gen_dir expects, without a manual
video_utils.py step in between.

No GPU/codebase needed: we synthesize a tiny real .mp4 with cv2 so
extract_frames() has real bytes to decode, and for the ExperimentRunner
path we monkeypatch run_inference() so we don't shell out to data_gen.py
/ demo.py (which don't exist outside the real cognvs-codebase).
"""
import json

import cv2
import numpy as np
import pytest

from src.inference.run_cognvs import collect_outputs
from src.experiments.runner import ExperimentRunner
import src.experiments.runner as runner_module


def make_fake_video(path, num_frames=5, size=(32, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size
    )
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_collect_outputs_extracts_frames_from_video(tmp_path):
    """run_cognvs.py's own CLI path (collect_outputs) must leave a
    populated frames/ dir next to output.mp4, not just the video."""
    codebase_path = tmp_path / "cognvs-codebase"
    repo_root = tmp_path / "cognvs-reproduction"
    src_output = codebase_path / "demo_data" / "davis_bear" / "outputs" / "eval_render1_out.mp4"
    make_fake_video(src_output, num_frames=5)

    dest_dir = collect_outputs(
        codebase_path, repo_root, sequence="davis_bear", angle=30,
        elapsed_seconds=1.23, dry_run=False,
    )

    frames = sorted((dest_dir / "frames").glob("*.png"))
    assert len(frames) == 5
    assert (dest_dir / "output.mp4").exists()


def test_collect_outputs_dry_run_does_not_require_real_video(tmp_path):
    """Dry run must not touch cv2/decode a real file — just report intent
    and still leave the expected directory shape behind."""
    codebase_path = tmp_path / "cognvs-codebase"
    repo_root = tmp_path / "cognvs-reproduction"

    dest_dir = collect_outputs(
        codebase_path, repo_root, sequence="davis_bear", angle=30,
        elapsed_seconds=0.0, dry_run=True,
    )

    assert (dest_dir / "frames").exists()
    assert not (dest_dir / "output.mp4").exists()


def make_config(tmp_path, codebase, dry_run):
    return {
        "experiment_id": "EXP01",
        "run_id": "EXP01_davis_bear_angle030",
        "input_sequence": "davis_bear",
        "angle_deg": 30,
        "checkpoint": "cognvs_ckpt_inpaint",
        "fine_tuning_steps": 0,
        "resolution": "480p",
        "seed": 0,
        "output_dir": str(tmp_path / "out"),
        "codebase_path": str(codebase),
        "dry_run": dry_run,
    }


def test_experiment_runner_execute_populates_frames_dir(tmp_path, monkeypatch):
    """ExperimentRunner.execute() must extract self.video_path into
    self.frames_dir once the real (non-dry-run) video is copied in."""
    codebase = tmp_path / "cognvs-codebase"
    (codebase / "trajs").mkdir(parents=True)

    fake_video = tmp_path / "fake_render.mp4"
    make_fake_video(fake_video, num_frames=4)

    captured = {}

    def fake_run_inference(codebase_path, repo_root, sequence, angle, dry_run, seed=None):
        captured["seed"] = seed
        return 2.5, fake_video

    monkeypatch.setattr(runner_module, "run_inference", fake_run_inference)

    config = make_config(tmp_path, codebase, dry_run=False)
    config["seed"] = 42
    runner = ExperimentRunner(config)
    runner.run()

    assert captured["seed"] == 42

    frames = sorted(runner.frames_dir.glob("*.png"))
    assert len(frames) == 4
    assert runner.video_path.exists()

    metadata = json.loads((runner.output_dir / "metadata.json").read_text())
    assert metadata["status"] == "complete"


def test_experiment_runner_execute_dry_run_leaves_frames_dir_empty(tmp_path, monkeypatch):
    codebase = tmp_path / "cognvs-codebase"
    (codebase / "trajs").mkdir(parents=True)

    config = make_config(tmp_path, codebase, dry_run=True)
    runner = ExperimentRunner(config)
    runner.run()

    assert runner.frames_dir.exists()
    assert list(runner.frames_dir.iterdir()) == []
