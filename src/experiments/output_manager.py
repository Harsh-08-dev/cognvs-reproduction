from pathlib import Path


def create_output_directory(output_dir):
    """
    Creates the run's output directory plus the sub-paths P1 and P3 both
    rely on, so the folder layout is fixed before P1 ever runs CogNVS.

    Layout (matches docs/analysis/evaluation_protocol.md "Inputs required
    from P1" in the cognvs-reproduction repo):

        <output_dir>/
            frames/          <- P1 writes generated frames here (*.png),
                                for evaluate.py --gen_dir
            output.mp4       <- OR P1 writes a single video here instead
            metadata.json    <- written by metadata.py (P2)
            config.yaml      <- copy of the resolved run config (P2)
            runtime.json     <- runtime + GPU memory usage (P1 fills in
                                after execution; placeholder created here)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frames_dir = get_frames_dir(output_path)
    frames_dir.mkdir(parents=True, exist_ok=True)

    return output_path


def get_frames_dir(output_dir):
    return Path(output_dir) / "frames"


def get_video_path(output_dir):
    return Path(output_dir) / "output.mp4"


def get_config_path(output_dir):
    return Path(output_dir) / "config.yaml"


def get_runtime_path(output_dir):
    return Path(output_dir) / "runtime.json"