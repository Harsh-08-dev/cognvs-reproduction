"""
Generates custom camera trajectory files for the angle-sweep experiment (EXP01).

CogNVS's data_gen.py (eval mode) reads trajectory files in this exact format:
    line 1: phi_start phi_end      (azimuth angle deviation, in degrees)
    line 2: theta_start theta_end  (elevation angle deviation, in degrees)
    line 3: r_start r_end          (radius/zoom scale deviation, unitless)

For our angle sweep, we hold theta and r fixed (no elevation change, no zoom),
and vary only phi across our target angles.

Usage:
    python src/utils/generate_trajectories.py --angles 15 30 60 90 --out_dir configs/trajectories
"""

import argparse
import os


def write_trajectory_file(path: str, phi_end: float, theta_end: float = 0.0, r_end: float = 0.0):
    """
    Writes a single trajectory .txt file in CogNVS's expected format.

    phi_end: target azimuth angle in degrees (sweep goes from 0 -> phi_end)
    theta_end: target elevation angle in degrees (kept at 0 for a pure azimuth sweep)
    r_end: target radius/zoom deviation (kept at 0 for no zoom change)
    """
    with open(path, "w") as f:
        f.write(f"0 {phi_end}\n")
        f.write(f"0 {theta_end}\n")
        f.write(f"0. {r_end}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate angle-sweep trajectory files for EXP01")
    parser.add_argument("--angles", type=float, nargs="+", required=True,
                         help="List of target azimuth angles in degrees, e.g. --angles 15 30 60 90")
    parser.add_argument("--out_dir", type=str, required=True,
                         help="Directory to write trajectory files into")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for angle in args.angles:
        # Filename encodes the angle so it's unambiguous later, e.g. traj_angle015.txt
        filename = f"traj_angle{int(angle):03d}.txt"
        path = os.path.join(args.out_dir, filename)
        write_trajectory_file(path, phi_end=angle)
        print(f"Wrote {path}  (phi: 0 -> {angle} deg)")


if __name__ == "__main__":
    main()