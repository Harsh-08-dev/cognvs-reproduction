import argparse
from pathlib import Path
import yaml


REQUIRED_FIELDS = [
    "experiment_id",
    "run_id",
    "input_sequence",
    "checkpoint",
    "fine_tuning_steps",
    "resolution",
    "seed",
    "output_dir",
]


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML dictionary.")

    return config


def validate_required_fields(config):
    missing = [
        field for field in REQUIRED_FIELDS
        if field not in config
    ]

    if missing:
        raise ValueError(
            f"Missing required fields: {', '.join(missing)}"
        )


def validate_values(config):
    if not isinstance(config["fine_tuning_steps"], int):
        raise ValueError("fine_tuning_steps must be an integer.")

    if config["fine_tuning_steps"] < 0:
        raise ValueError("fine_tuning_steps cannot be negative.")

    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer.")

    if not config["experiment_id"]:
        raise ValueError("experiment_id cannot be empty.")

    if not config["run_id"]:
        raise ValueError("run_id cannot be empty.")

    if not config["output_dir"]:
        raise ValueError("output_dir cannot be empty.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate a CogNVS experiment configuration."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML configuration."
    )

    args = parser.parse_args()

    config = load_config(args.config)

    validate_required_fields(config)
    validate_values(config)

    print("Configuration loaded successfully.")
    print(f"Experiment: {config['experiment_id']}")
    print(f"Run: {config['run_id']}")
    print(f"Fine-tuning steps: {config['fine_tuning_steps']}")
    print(f"Output directory: {config['output_dir']}")


if __name__ == "__main__":
    main()