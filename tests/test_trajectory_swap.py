"""
Covers issue #7 (part 1): the trajectory-swap/restore logic in
src/inference/run_cognvs.py had no direct test coverage. swap_trajectory()
and restore_trajectory() are what temporarily replace the upstream
cognvs-codebase's trajs/ folder with a single angle-specific trajectory
file and put it back afterwards — get this wrong and we either corrupt the
sibling checkout or silently run inference against the wrong trajectory.

No GPU/codebase needed: everything here operates on plain tmp_path
directories standing in for cognvs-codebase and this repo's configs/, so
it runs the same on any machine (per docs/setup.md's "no GPU available"
constraint).
"""
import pytest

from src.inference.run_cognvs import (
    restore_trajectory,
    run_inference,
    swap_trajectory,
)


def make_codebase(tmp_path, existing_traj_files=("other_traj.txt",)):
    codebase = tmp_path / "cognvs-codebase"
    trajs_dir = codebase / "trajs"
    trajs_dir.mkdir(parents=True)
    for name in existing_traj_files:
        (trajs_dir / name).write_text("original CMU trajectory data")
    return codebase


def make_repo_root_with_traj(tmp_path, angle=30, contents="traj for angle 30"):
    repo_root = tmp_path / "cognvs-reproduction"
    traj_dir = repo_root / "configs" / "trajectories"
    traj_dir.mkdir(parents=True)
    (traj_dir / f"traj_angle{angle:03d}.txt").write_text(contents)
    return repo_root


# -- swap_trajectory: happy path -------------------------------------------

def test_swap_trajectory_backs_up_original_and_installs_target_angle(tmp_path):
    codebase = make_codebase(tmp_path)
    repo_root = make_repo_root_with_traj(tmp_path, angle=30, contents="angle030-data")

    backup_dir = swap_trajectory(codebase, angle=30, repo_root=repo_root)

    # original trajs/ moved intact to the backup location
    assert backup_dir == codebase / "trajs_backup_tmp"
    assert (backup_dir / "other_traj.txt").read_text() == "original CMU trajectory data"

    # trajs/ now contains ONLY the target angle's file, with correct contents
    trajs_dir = codebase / "trajs"
    assert [p.name for p in trajs_dir.iterdir()] == ["traj_angle030.txt"]
    assert (trajs_dir / "traj_angle030.txt").read_text() == "angle030-data"


def test_swap_trajectory_missing_source_file_raises(tmp_path):
    codebase = make_codebase(tmp_path)
    repo_root = tmp_path / "cognvs-reproduction"
    (repo_root / "configs" / "trajectories").mkdir(parents=True)
    # no traj_angle030.txt written

    with pytest.raises(FileNotFoundError):
        swap_trajectory(codebase, angle=30, repo_root=repo_root)

    # must not have touched trajs/ at all before validating the source file
    assert (codebase / "trajs" / "other_traj.txt").exists()
    assert not (codebase / "trajs_backup_tmp").exists()


# -- swap_trajectory: backup dir already exists -----------------------------

def test_swap_trajectory_refuses_when_backup_dir_already_exists(tmp_path):
    """A stale trajs_backup_tmp/ means a previous run exited uncleanly.
    swap_trajectory() must refuse rather than silently overwrite or nest
    into it, since either could lose the original trajectory data."""
    codebase = make_codebase(tmp_path)
    repo_root = make_repo_root_with_traj(tmp_path, angle=30)

    stale_backup = codebase / "trajs_backup_tmp"
    stale_backup.mkdir()
    (stale_backup / "leftover.txt").write_text("from a prior crashed run")

    with pytest.raises(RuntimeError, match="Backup dir already exists"):
        swap_trajectory(codebase, angle=30, repo_root=repo_root)

    # must not have touched the live trajs/ dir or the stale backup
    assert (codebase / "trajs" / "other_traj.txt").exists()
    assert (stale_backup / "leftover.txt").read_text() == "from a prior crashed run"


# -- restore_trajectory ------------------------------------------------------

def test_restore_trajectory_puts_original_back_and_removes_swapped_copy(tmp_path):
    codebase = make_codebase(tmp_path)
    repo_root = make_repo_root_with_traj(tmp_path, angle=30)

    backup_dir = swap_trajectory(codebase, angle=30, repo_root=repo_root)
    restore_trajectory(codebase, backup_dir)

    trajs_dir = codebase / "trajs"
    assert (trajs_dir / "other_traj.txt").read_text() == "original CMU trajectory data"
    assert not (trajs_dir / "traj_angle030.txt").exists()
    assert not backup_dir.exists()


def test_restore_trajectory_replaces_whatever_is_currently_at_trajs_dir(tmp_path):
    """restore_trajectory() must win even if something (e.g. a half-written
    swap) already occupies trajs_dir — it removes it before moving the
    backup back, rather than failing or moving the backup inside it."""
    codebase = tmp_path / "cognvs-codebase"
    (codebase / "trajs").mkdir(parents=True)
    (codebase / "trajs" / "leftover_swap_file.txt").write_text("should be discarded")

    backup_dir = codebase / "trajs_backup_tmp"
    backup_dir.mkdir()
    (backup_dir / "original.txt").write_text("the real original data")

    restore_trajectory(codebase, backup_dir)

    trajs_dir = codebase / "trajs"
    assert [p.name for p in trajs_dir.iterdir()] == ["original.txt"]
    assert not backup_dir.exists()


# -- run_inference: end-to-end swap -> restore, even on data_gen failure ----

def test_run_inference_restores_trajectory_even_if_data_gen_fails(tmp_path, monkeypatch):
    """run_inference() wraps run_data_gen() in try/finally specifically so
    a failed data_gen.py subprocess still leaves the upstream codebase's
    trajs/ folder intact. dry_run alone doesn't exercise this because
    run_data_gen() just returns early under dry_run without raising."""
    codebase = make_codebase(tmp_path)
    repo_root = make_repo_root_with_traj(tmp_path, angle=30)

    import src.inference.run_cognvs as run_cognvs_module

    def failing_data_gen(codebase_path, sequence, dry_run):
        raise subprocess_error

    subprocess_error = RuntimeError("data_gen.py exited non-zero")
    monkeypatch.setattr(run_cognvs_module, "run_data_gen", failing_data_gen)

    with pytest.raises(RuntimeError, match="data_gen.py exited non-zero"):
        run_inference(codebase, repo_root, sequence="davis_bear", angle=30, dry_run=False)

    # restore_trajectory must still have run despite the failure above
    assert (codebase / "trajs" / "other_traj.txt").exists()
    assert not (codebase / "trajs_backup_tmp").exists()


def test_run_inference_dry_run_still_swaps_and_restores_trajectory(tmp_path):
    """dry_run only skips the data_gen.py/demo.py subprocess calls — the
    trajectory swap/restore itself is real file I/O either way, so a dry
    run still needs a valid trajectory file and still leaves trajs/
    exactly as it found it."""
    codebase = make_codebase(tmp_path)
    repo_root = make_repo_root_with_traj(tmp_path, angle=30)

    run_inference(codebase, repo_root, sequence="davis_bear", angle=30, dry_run=True)

    assert (codebase / "trajs" / "other_traj.txt").exists()
    assert not (codebase / "trajs_backup_tmp").exists()
