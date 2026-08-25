from datetime import datetime, timezone


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