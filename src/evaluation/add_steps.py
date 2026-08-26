"""
src/evaluation/add_steps.py

Adds a 'steps' field to each metrics.json based on filename pattern FT<steps>.json
e.g. FT000.json -> steps=0, FT050.json -> steps=50, FT100.json -> steps=100, FT200.json -> steps=200

Usage:
    python add_steps.py --metrics_dir ../../results/metrics
"""
import argparse
import json
import os
import glob
import re


def extract_steps_from_tag(tag: str) -> int:
    """
    Extracts the numeric step count from a tag like 'FT000', 'FT050', 'FT100', 'FT200'.
    Falls back to extracting any trailing digits if the FT prefix isn't present.
    """
    match = re.search(r'FT(\d+)', tag)
    if match:
        return int(match.group(1))
    # fallback: last underscore-separated numeric chunk, e.g. EXP01_100 -> 100
    match = re.search(r'(\d+)$', tag)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract step count from tag: {tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    args = parser.parse_args()

    for path in glob.glob(os.path.join(args.metrics_dir, "*.json")):
        with open(path) as f:
            data = json.load(f)
        steps = extract_steps_from_tag(data["tag"])
        data["steps"] = steps
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {path} -> tag={data['tag']}, steps={steps}")


if __name__ == "__main__":
    main()