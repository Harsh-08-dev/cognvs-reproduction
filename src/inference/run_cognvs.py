"""
P1's inference wrapper for EXP01 (angle-sweep, zero-shot).

Given a sequence name and a target angle, this script:
  1. Backs up CMU's trajs/ folder
  2. Swaps in ONLY our target angle's trajectory file (deterministic output naming)
  3. Runs data_gen.py --mode eval (reconstruction/warping stage)
  4. Restores the original trajs/ folder
  5. Runs demo.py (neural inpainting / zero-shot inference stage)
  6. Copies outputs into a standardized results/ folder in THIS repo

Usage:
    python src/inference/run_cognvs.py --sequence davis_bear --angle 30

Dry run (no GPU needed, just verifies file logic):
    python src/inference/run_cognvs.py --sequence davis_bear --angle 30 --dry_run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_codebase_path(explicit_path: str = None) -> Path:
    """
    Locates the sibling cognvs-codebase folder.
    Assumes the standard R1/ layout: R1/cognvs-codebase and R1/cognvs-reproduction
    are siblings. Falls back to an explicit path if the layout differs.
    """
    if explicit_path:
        return Path(explicit_path).resolve()

    # this file lives at <repo>/src/inference/run_cognvs.py
    # repo root is 2 levels up, R1/ is one level above that
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root.parent / "cognvs-codebase"
    return candidate


def swap_trajectory(codebase_path: Path, angle: int, repo_root: Path):
    """
    Backs up trajs/, then replaces it with a single file for the target angle.
    Returns the backup path so it can be restored later.
    """
    trajs_dir = codebase_path / "trajs"
    backup_dir = codebase_path / "trajs_backup_tmp"
    source_traj_file = repo_root / "configs" / "trajectories" / f"traj_angle{angle:03d}.txt"

    if not source_traj_file.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {source_traj_file}\n"
            f"Run generate_trajectories.py first, or check the angle value."
        )

    if backup_dir.exists():
        raise RuntimeError(
            f"Backup dir already exists at {backup_dir} — a previous run may have "
            f"exited uncleanly. Please inspect and remove it manually before retrying."
        )

    shutil.move(str(trajs_dir), str(backup_dir))
    trajs_dir.mkdir()
    shutil.copy(str(source_traj_file), str(trajs_dir / f"traj_angle{angle:03d}.txt"))

    return backup_dir


def restore_trajectory(codebase_path: Path, backup_dir: Path):
    """Restores CMU's original trajs/ folder from backup."""
    trajs_dir = codebase_path / "trajs"
    if trajs_dir.exists():
        shutil.rmtree(trajs_dir)
    shutil.move(str(backup_dir), str(trajs_dir))


def run_data_gen(codebase_path: Path, sequence: str, dry_run: bool):
    """Runs data_gen.py --mode eval to produce the warped novel-view render."""
    cmd = [
        sys.executable, "data_gen.py",
        "--device", "cuda:0",
        "--data_path", f"demo_data/{sequence}",
        "--mode", "eval",
    ]
    print(f"[run_cognvs] data_gen command: {' '.join(cmd)}")
    if dry_run:
        print("[run_cognvs] DRY RUN — skipping actual execution")
        return
    subprocess.run(cmd, cwd=str(codebase_path), check=True)


def run_demo(codebase_path: Path, sequence: str, dry_run: bool):
    """Runs demo.py to perform zero-shot neural inpainting on the eval render."""
    cmd = [
        sys.executable, "demo.py",
        "--model_path", "checkpoints/CogVideoX-5b-I2V",
        "--cognvs_ckpt_path", "checkpoints/cognvs_ckpt_inpaint",
        "--data_path", f"demo_data/{sequence}",
        "--mp4_name", "eval_render1.mp4",
    ]
    print(f"[run_cognvs] demo command: {' '.join(cmd)}")
    if dry_run:
        print("[run_cognvs] DRY RUN — skipping actual execution")
        return
    subprocess.run(cmd, cwd=str(codebase_path), check=True)


def collect_outputs(codebase_path: Path, repo_root: Path, sequence: str, angle: int,
                     elapsed_seconds: float, dry_run: bool):
    """
    Copies the generated output video + metadata into this repo's standardized
    results/ folder, per the team's agreed output format:
        results/EXP01/<sequence>/angle_<NNN>/output.mp4, config.yaml, runtime.json, log.txt
    """
    src_output = codebase_path / "demo_data" / sequence / "outputs" / "eval_render1_out.mp4"
    dest_dir = repo_root / "results" / "EXP01" / sequence / f"angle_{angle:03d}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[run_cognvs] DRY RUN — would copy {src_output} -> {dest_dir / 'output.mp4'}")
    else:
        shutil.copy(str(src_output), str(dest_dir / "output.mp4"))

    config = {
        "experiment_id": "EXP01",
        "sequence_id": sequence,
        "angle_deg": angle,
        "checkpoint": "cognvs_ckpt_inpaint",
        "mode": "zero-shot",
    }
    with open(dest_dir / "config.yaml", "w") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    runtime = {
        "elapsed_seconds": elapsed_seconds,
        "dry_run": dry_run,
    }
    with open(dest_dir / "runtime.json", "w") as f:
        json.dump(runtime, f, indent=2)

    print(f"[run_cognvs] Output collected at: {dest_dir}")
    return dest_dir


def run_inference(codebase_path: Path, repo_root: Path, sequence: str, angle: int, dry_run: bool):
    """
    Runs the full zero-shot angle-sweep pipeline (swap trajectory -> data_gen
    -> restore trajectory -> demo) and returns (elapsed_seconds, output_video_path),
    where output_video_path is the raw video produced inside cognvs-codebase
    (not yet copied anywhere).

    This is the single orchestration entry point for EXP01. Both this script's
    own CLI (main(), below) and ExperimentRunner.execute() in
    src/experiments/runner.py call this function, so there is exactly one
    place that knows how to drive the upstream CogNVS codebase.
    """
    start_time = time.time()
    backup_dir = swap_trajectory(codebase_path, angle, repo_root)
    try:
        run_data_gen(codebase_path, sequence, dry_run)
    finally:
        # always restore, even if data_gen fails, so we never leave their repo altered
        restore_trajectory(codebase_path, backup_dir)

    run_demo(codebase_path, sequence, dry_run)
    elapsed = time.time() - start_time

    output_video = codebase_path / "demo_data" / sequence / "outputs" / "eval_render1_out.mp4"
    return elapsed, output_video


def main():
    parser = argparse.ArgumentParser(description="Run CogNVS zero-shot inference at a given novel-view angle")
    parser.add_argument("--sequence", type=str, required=True, help="e.g. davis_bear, sora_balloon")
    parser.add_argument("--angle", type=int, required=True, help="Target azimuth angle in degrees, e.g. 30")
    parser.add_argument("--codebase_path", type=str, default=None,
                         help="Path to cognvs-codebase, if not using the standard sibling-folder layout")
    parser.add_argument("--dry_run", action="store_true",
                         help="Verify file logic without actually running GPU inference")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    codebase_path = find_codebase_path(args.codebase_path)

    if not codebase_path.exists():
        print(f"ERROR: codebase path not found at {codebase_path}")
        print("Pass --codebase_path explicitly if your layout differs from the standard R1/ sibling structure.")
        sys.exit(1)

    print(f"[run_cognvs] Using codebase at: {codebase_path}")
    print(f"[run_cognvs] Sequence: {args.sequence}, Angle: {args.angle} deg, Dry run: {args.dry_run}")

    elapsed, _ = run_inference(codebase_path, repo_root, args.sequence, args.angle, args.dry_run)

    collect_outputs(codebase_path, repo_root, args.sequence, args.angle, elapsed, args.dry_run)


if __name__ == "__main__":
    main()