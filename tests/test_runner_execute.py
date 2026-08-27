"""
Covers issue #2: ExperimentRunner.execute() must actually call P1's
CogNVS inference logic (src/inference/run_cognvs.py) and populate the
output layout P3's evaluate.py/aggregator.py expect. Runs entirely with
--dry_run, no GPU and no real cognvs-codebase checkout required.
"""

import json

import pytest

from src.experiments.runner import ExperimentRunner


def make_fake_codebase(tmp_path):
    codebase = tmp_path / "cognvs-codebase"
    (codebase / "trajs").mkdir(parents=True)
    (codebase / "trajs" / "placeholder.txt").write_text("orig")
    return codebase


def make_config(tmp_path, codebase):
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
        "dry_run": True,
    }


def test_execute_populates_expected_output_layout(tmp_path):
    codebase = make_fake_codebase(tmp_path)
    config = make_config(tmp_path, codebase)

    runner = ExperimentRunner(config)
    runner.run()

    produced = {p.name for p in runner.output_dir.iterdir()}
    assert {"frames", "config.yaml", "metadata.json", "runtime.json"} <= produced


def test_execute_writes_real_runtime_not_placeholder(tmp_path):
    codebase = make_fake_codebase(tmp_path)
    config = make_config(tmp_path, codebase)

    runner = ExperimentRunner(config)
    runner.run()

    runtime = json.loads(runner.runtime_path.read_text())
    assert runtime["status"] == "complete"
    assert runtime["runtime_seconds"] is not None


def test_execute_marks_metadata_complete(tmp_path):
    codebase = make_fake_codebase(tmp_path)
    config = make_config(tmp_path, codebase)

    runner = ExperimentRunner(config)
    runner.run()

    metadata = json.loads((runner.output_dir / "metadata.json").read_text())
    assert metadata["status"] == "complete"


def test_execute_restores_trajs_dir_even_though_it_ran(tmp_path):
    """The trajectory swap/restore in run_cognvs.py must still leave the
    upstream codebase's trajs/ folder intact after ExperimentRunner runs it
    end-to-end, not just when run_cognvs.py's own CLI is used directly."""
    codebase = make_fake_codebase(tmp_path)
    config = make_config(tmp_path, codebase)

    runner = ExperimentRunner(config)
    runner.run()

    assert (codebase / "trajs" / "placeholder.txt").exists()
    assert not (codebase / "trajs_backup_tmp").exists()


def test_execute_missing_codebase_path_raises(tmp_path):
    missing_codebase = tmp_path / "does-not-exist"
    config = make_config(tmp_path, missing_codebase)

    runner = ExperimentRunner(config)
    runner.prepare()

    with pytest.raises(FileNotFoundError):
        runner.execute()
