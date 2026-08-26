import argparse
from pathlib import Path

import yaml

from src.experiments.output_manager import (
    create_output_directory,
    get_frames_dir,
    get_video_path,
    get_runtime_path,
)
from src.experiments.metadata import (
    create_metadata,
    save_metadata,
    save_config_copy,
    init_runtime_placeholder,
)


class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.output_dir = None
        self.metadata = None

        # Convenience paths for P1 to write into during execute(). These
        # match what P3's evaluate.py / evaluation_protocol.md expect to
        # find in each run's output directory.
        self.frames_dir = None
        self.video_path = None
        self.runtime_path = None

    def prepare(self):
        self.output_dir = create_output_directory(
        self.config["output_dir"]
        )

        self.frames_dir = get_frames_dir(self.output_dir)
        self.video_path = get_video_path(self.output_dir)
        self.runtime_path = get_runtime_path(self.output_dir)

        save_config_copy(self.config, self.output_dir)
        init_runtime_placeholder(self.output_dir)

        self.metadata = create_metadata(self.config)

        self.metadata_path = save_metadata(
        self.metadata,
        self.output_dir
        )
        self.metadata["status"] = "prepared"
        save_metadata(self.metadata, self.output_dir)

    def execute(self):
        """
        P1 implements this method to actually run CogNVS for this run's
        config once the inference interface (checkpoint loading, test-time
        fine-tuning, generation call) is confirmed.

        By the time execute() is called, prepare() has already created:
          - self.frames_dir  : write generated frames here as *.png
                                (or write a single video to self.video_path)
          - self.output_dir/config.yaml   : copy of self.config
          - self.runtime_path              : runtime.json placeholder to
                                              overwrite with real
                                              runtime_seconds / peak GPU
                                              memory / gpu_name once the
                                              run finishes

        Do not change these paths/filenames — P3's evaluate.py and
        aggregator.py are already wired to read from this exact layout.
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