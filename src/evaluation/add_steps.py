"""
src/evaluation/add_steps.py

Adds a 'steps' field to each metrics.json based on filename pattern FT<steps>.json
e.g. FT000.json -> steps=0, FT050.json -> steps=50, FT100.json -> steps=100, FT200.json -> steps=200

Usage (run from repo root, like every other script in this repo):
    python -m src.evaluation.add_steps --metrics_dir results/metrics
"""
import argparse
import json
import os
import glob
import re


def extract_steps_from_tag(tag: str) -> int:
    """
    Extracts the numeric step count from a tag like 'FT000', 'FT050', 'FT100', 'FT200'.

    Deliberately strict: earlier versions fell back to grabbing any trailing
    digits from the tag, which silently mislabeled angle-sweep tags (e.g.
    'ANGLE030' -> steps=30) instead of failing. Add angle-sweep metrics with
    add_angle.py, not this script.
    """
    match = re.search(r'^FT(\d+)$', tag)
    if match:
        return int(match.group(1))

    if re.search(r'ANGLE\d+', tag, re.IGNORECASE):
        raise ValueError(
            f"Tag '{tag}' looks like angle-sweep data (ANGLE### format), "
            f"not a fine-tuning-steps tag (FT###). Use add_angle.py for "
            f"angle-sweep metrics.json files instead."
        )

    raise ValueError(
        f"Could not extract step count from tag: {tag!r}. Expected FT### "
        f"format (e.g. 'FT000', 'FT100')."
    )


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