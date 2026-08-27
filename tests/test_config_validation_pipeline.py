"""
Covers issue #7 (part 2): config validation (scripts/validate_experiment.py,
P2's tooling) and ExperimentRunner.prepare()'s output layout were only ever
exercised separately. Nothing tested that a config which *passes*
validate_required_fields()/validate_values() actually produces the output
layout prepare() promises, or that prepare() surfaces the same missing/bad
fields validate_experiment.py is meant to catch before a real (GPU) run is
attempted.

No GPU/codebase needed — prepare() never touches cognvs-codebase, only the
output directory.
"""
import json

import pytest
import yaml

from scripts.validate_experiment import validate_required_fields, validate_values
from src.experiments.runner import ExperimentRunner, load_config


def make_valid_config(tmp_path):
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
    }


# -- a config that passes P2's validator actually works with prepare() -----

def test_validated_config_produces_expected_prepare_layout(tmp_path):
    config = make_valid_config(tmp_path)

    # must pass P2's own pre-flight checks first
    validate_required_fields(config)
    validate_values(config)

    runner = ExperimentRunner(config)
    runner.prepare()

    produced = {p.name for p in runner.output_dir.iterdir()}
    assert {"frames", "config.yaml", "metadata.json", "runtime.json"} <= produced
    assert runner.frames_dir == runner.output_dir / "frames"
    assert runner.frames_dir.is_dir()
    assert list(runner.frames_dir.iterdir()) == []  # empty until execute()


def test_prepare_writes_config_copy_matching_input_config(tmp_path):
    config = make_valid_config(tmp_path)
    validate_required_fields(config)
    validate_values(config)

    runner = ExperimentRunner(config)
    runner.prepare()

    written = yaml.safe_load((runner.output_dir / "config.yaml").read_text())
    assert written == config


def test_prepare_writes_metadata_with_prepared_status(tmp_path):
    config = make_valid_config(tmp_path)
    validate_required_fields(config)
    validate_values(config)

    runner = ExperimentRunner(config)
    runner.prepare()

    metadata = json.loads((runner.output_dir / "metadata.json").read_text())
    assert metadata["status"] == "prepared"
    assert metadata["run_id"] == config["run_id"]
    assert metadata["input_sequence"] == config["input_sequence"]
    assert metadata["angle_deg"] == config["angle_deg"]


def test_prepare_writes_not_run_runtime_placeholder(tmp_path):
    config = make_valid_config(tmp_path)
    validate_required_fields(config)
    validate_values(config)

    runner = ExperimentRunner(config)
    runner.prepare()

    runtime = json.loads(runner.runtime_path.read_text())
    assert runtime["status"] == "not_run"
    assert runtime["runtime_seconds"] is None


# -- configs P2's validator rejects should never reach prepare() -----------

@pytest.mark.parametrize("missing_field", [
    "experiment_id", "run_id", "input_sequence", "angle_deg", "checkpoint",
    "fine_tuning_steps", "resolution", "seed", "output_dir",
])
def test_validate_required_fields_catches_each_missing_field_before_prepare(
    tmp_path, missing_field
):
    config = make_valid_config(tmp_path)
    del config[missing_field]

    with pytest.raises(ValueError, match="Missing required fields"):
        validate_required_fields(config)


def test_validate_values_rejects_negative_fine_tuning_steps_before_prepare(tmp_path):
    config = make_valid_config(tmp_path)
    config["fine_tuning_steps"] = -5

    validate_required_fields(config)  # all fields present, so this passes
    with pytest.raises(ValueError, match="fine_tuning_steps cannot be negative"):
        validate_values(config)


def test_validate_values_rejects_non_integer_seed_before_prepare(tmp_path):
    config = make_valid_config(tmp_path)
    config["seed"] = "zero"

    validate_required_fields(config)
    with pytest.raises(ValueError, match="seed must be an integer"):
        validate_values(config)


def test_validate_values_rejects_non_numeric_angle_deg_before_prepare(tmp_path):
    config = make_valid_config(tmp_path)
    config["angle_deg"] = "thirty"

    validate_required_fields(config)
    with pytest.raises(ValueError, match="angle_deg must be a number"):
        validate_values(config)


def test_validate_values_rejects_zero_or_negative_angle_deg_before_prepare(tmp_path):
    config = make_valid_config(tmp_path)
    config["angle_deg"] = 0

    validate_required_fields(config)
    with pytest.raises(ValueError, match="angle_deg must be positive"):
        validate_values(config)


def test_prepare_itself_raises_keyerror_on_a_config_the_validator_would_reject(tmp_path):
    """If validate_experiment.py is ever skipped, prepare() (via
    metadata.create_metadata) still fails loudly on a config missing a
    required field, rather than silently writing incomplete metadata."""
    config = make_valid_config(tmp_path)
    del config["checkpoint"]  # validate_required_fields would have caught this

    runner = ExperimentRunner(config)
    with pytest.raises(KeyError):
        runner.prepare()


# -- the on-disk example config (configs/exp01_davis_bear_angle030.yaml)
# is itself valid and produces the expected layout end-to-end --------------

def test_shipped_exp01_config_is_valid_and_prepares_successfully(tmp_path):
    config = load_config("configs/exp01_davis_bear_angle030.yaml")
    validate_required_fields(config)
    validate_values(config)

    # redirect output_dir into tmp_path so this test doesn't write into
    # the real results/ folder
    config = dict(config)
    config["output_dir"] = str(tmp_path / "out")

    runner = ExperimentRunner(config)
    runner.prepare()

    produced = {p.name for p in runner.output_dir.iterdir()}
    assert {"frames", "config.yaml", "metadata.json", "runtime.json"} <= produced
