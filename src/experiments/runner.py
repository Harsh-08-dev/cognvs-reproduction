"""
src/experiments/runner.py

Runs a full EXP01 (zero-shot angle-sweep) experiment from a YAML config:
config -> inference (src/inference/run_cognvs.py) -> standardized output
layout (frames/, output.mp4, metadata.json, config.yaml, runtime.json).

Usage (run from repo root, like every other script in this repo):
    python -m src.experiments.runner --config configs/exp01_davis_bear_angle030.yaml

Dry run (no GPU needed, just verifies config/output-layout logic):
    python -m src.experiments.runner --config configs/exp01_davis_bear_angle030.yaml --dry_run
"""
import argparse
import json
import shutil
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
from src.inference.run_cognvs import find_codebase_path, run_inference
from src.evaluation.video_utils import extract_frames


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
        Runs EXP01 (zero-shot angle-sweep) by calling into
        src/inference/run_cognvs.py's run_inference(), which owns all
        knowledge of how to drive the upstream cognvs-codebase (trajectory
        swap, data_gen, demo). This keeps P1's inference logic in one place
        instead of duplicating it here.

        By the time execute() is called, prepare() has already created:
          - self.frames_dir  : empty dir created by prepare(); this method
                                extracts self.video_path into *.png frames
                                here once the video is copied, so evaluate.py
                                --gen_dir has no manual step to run first
          - self.video_path  : where this method copies the generated video
          - self.output_dir/config.yaml   : copy of self.config
          - self.runtime_path              : runtime.json placeholder,
                                              overwritten below with the
                                              real runtime once the run
                                              finishes

        Required config keys: input_sequence (str, e.g. "davis_bear"),
        angle_deg (int, e.g. 30), seed (int — forwarded to demo.py's
        diffusion sampling; required for Best-of-K runs to actually
        produce K different outputs). Optional: codebase_path, dry_run.

        Do not change these paths/filenames — P3's evaluate.py and
        aggregator.py are already wired to read from this exact layout.
        """
        dry_run = bool(self.config.get("dry_run", False))
        codebase_path = find_codebase_path(self.config.get("codebase_path"))
        repo_root = Path(__file__).resolve().parents[2]

        if not codebase_path.exists():
            raise FileNotFoundError(
                f"codebase path not found at {codebase_path}; pass "
                f"'codebase_path' in the config if your layout differs "
                f"from the standard sibling-folder structure."
            )

        sequence = self.config["input_sequence"]
        angle = self.config["angle_deg"]
        seed = self.config.get("seed")

        elapsed, output_video = run_inference(
            codebase_path, repo_root, sequence, angle, dry_run, seed=seed
        )

        if dry_run:
            print(f"[ExperimentRunner] DRY RUN — would copy {output_video} -> {self.video_path}")
            print(f"[ExperimentRunner] DRY RUN — would extract {self.video_path} -> {self.frames_dir}")
        else:
            shutil.copy(str(output_video), str(self.video_path))
            extract_frames(str(self.video_path), str(self.frames_dir))

        runtime = {
            "status": "complete",
            "runtime_seconds": elapsed,
            "peak_gpu_memory_mb": None,
            "gpu_name": None,
            "notes": "dry_run" if dry_run else None,
        }
        with open(self.runtime_path, "w", encoding="utf-8") as f:
            json.dump(runtime, f, indent=2)

        self.metadata["status"] = "complete"
        save_metadata(self.metadata, self.output_dir)

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
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Verify config/output-layout logic without running GPU inference.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        config["dry_run"] = True

    print(f"Loaded experiment: {config['run_id']}")
    print(f"Input sequence: {config['input_sequence']}")
    print(f"Angle (deg): {config.get('angle_deg')}")
    print(f"Checkpoint: {config['checkpoint']}")
    print(f"Output directory: {config['output_dir']}")

    runner = ExperimentRunner(config)
    runner.run()

    print("Experiment run completed.")
    print(f"Output directory: {runner.output_dir}")


if __name__ == "__main__":
    main()