"""
Prerequisite fix for the planned "Best-of-K" experiment (Option B): running
CogNVS K times with different seeds only produces K different outputs if
the seed is actually forwarded to demo.py's diffusion sampling. Previously
`seed` was a required, validated config field that only ever made it into
metadata.json — run_demo()/run_inference() never accepted or used it, so
every run of the same angle/sequence was really seed-agnostic regardless
of what the config said.

These tests check the wiring at the cmd-construction level with --dry_run
(no GPU, no real cognvs-codebase checkout needed) rather than the actual
upstream demo.py behavior, since demo.py itself lives in the separate
cognvs-codebase repo we don't have here. If demo.py's real flag name isn't
--seed, only the flag string in run_demo() needs to change — these tests
would need the same update.
"""
from src.inference.run_cognvs import run_demo, run_inference
from pathlib import Path


def test_run_demo_dry_run_includes_seed_flag_when_given(capsys, tmp_path):
    run_demo(tmp_path, "davis_bear", dry_run=True, seed=7)
    printed = capsys.readouterr().out
    assert "--seed 7" in printed


def test_run_demo_dry_run_omits_seed_flag_when_not_given(capsys, tmp_path):
    run_demo(tmp_path, "davis_bear", dry_run=True)
    printed = capsys.readouterr().out
    assert "--seed" not in printed


def test_run_inference_forwards_seed_to_demo_command(capsys, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    codebase = tmp_path / "cognvs-codebase"
    (codebase / "trajs").mkdir(parents=True)

    run_inference(codebase, repo_root, "davis_bear", angle=30, dry_run=True, seed=123)

    printed = capsys.readouterr().out
    assert "--seed 123" in printed
    # trajectory swap/restore must still happen even though seed is new
    assert (codebase / "trajs").exists()


def test_different_seeds_produce_different_demo_commands(capsys, tmp_path):
    """Sanity check for Best-of-K: K runs with different seeds must not
    collapse into the same command — that would silently defeat the whole
    point of the experiment."""
    seen_commands = set()
    for seed in [0, 1, 2, 4, 8]:
        run_demo(tmp_path, "davis_bear", dry_run=True, seed=seed)
        printed = capsys.readouterr().out
        seen_commands.add(printed)

    assert len(seen_commands) == 5
