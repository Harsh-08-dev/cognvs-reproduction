import argparse
from pathlib import Path

import yaml

from src.experiments.output_manager import create_output_directory
from src.experiments.metadata import create_metadata, save_metadata


class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.output_dir = None
        self.metadata = None

    def prepare(self):
        self.output_dir = create_output_directory(
        self.config["output_dir"]
        )

        self.metadata = create_metadata(self.config)

        self.metadata_path = save_metadata(
        self.metadata,
        self.output_dir
        )
        self.metadata["status"] = "prepared"
        save_metadata(self.metadata, self.output_dir)

    def execute(self):
        """
        Placeholder for actual CogNVS execution.

        This will be connected to P1's implementation
        once the exact inference interface is confirmed.
        """
        raise NotImplementedError(
            "CogNVS execution interface has not been connected yet."
        )

    def run(self):
        self.prepare()
        self.execute()


def load_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run a CogNVS experiment."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML configuration.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Loaded experiment: {config['run_id']}")
    print(f"Fine-tuning steps: {config['fine_tuning_steps']}")
    print(f"Input sequence: {config['input_sequence']}")
    print(f"Checkpoint: {config['checkpoint']}")
    print(f"Output directory: {config['output_dir']}")

    runner = ExperimentRunner(config)
    runner.prepare()

    print("Experiment preparation completed.")
    print(f"Output directory: {runner.output_dir}")


if __name__ == "__main__":
    main()