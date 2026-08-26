from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

from src.experiments.output_manager import get_config_path, get_runtime_path


def create_metadata(config):
    return {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "input_sequence": config["input_sequence"],
        "checkpoint": config["checkpoint"],
        "fine_tuning_steps": config["fine_tuning_steps"],
        "resolution": config["resolution"],
        "seed": config["seed"],
        "output_dir": config["output_dir"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_metadata(metadata, output_dir):
    output_path = Path(output_dir) / "metadata.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return output_path

def update_metadata_status(metadata, output_dir, status):
    metadata["status"] = status
    save_metadata(metadata, output_dir)


def save_config_copy(config, output_dir):
    """
    Writes a copy of the resolved run config as config.yaml in the run's
    output directory. P3's evaluation protocol lists config.yaml as a
    required input alongside the generated frames/video and runtime.json,
    so this needs to live next to them, not just inside metadata.json.
    """
    config_path = get_config_path(output_dir)

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return config_path


def init_runtime_placeholder(output_dir):
    """
    Creates a runtime.json placeholder with the fields P1 is expected to
    fill in after running CogNVS (Week-1 task: "Record runtime and GPU
    memory usage"). Keeping the schema fixed here means P1 only has to
    populate values, not decide on field names P3's tooling will expect.
    """
    runtime_path = get_runtime_path(output_dir)

    placeholder = {
        "status": "not_run",
        "runtime_seconds": None,
        "peak_gpu_memory_mb": None,
        "gpu_name": None,
        "notes": None,
    }

    with runtime_path.open("w", encoding="utf-8") as file:
        json.dump(placeholder, file, indent=2)

    return runtime_path