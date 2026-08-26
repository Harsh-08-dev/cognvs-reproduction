from datetime import datetime, timezone
import json
from pathlib import Path


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