"""
src/evaluation/add_angle.py

Adds an 'angle' field to each metrics.json based on filename pattern ANGLE<degrees>.json
e.g. ANGLE015.json -> angle=15, ANGLE030.json -> angle=30, ANGLE090.json -> angle=90

Usage (run from repo root, like every other script in this repo):
    python -m src.evaluation.add_angle --metrics_dir results/metrics
"""
import argparse
import json
import os
import glob
import re


def extract_angle_from_tag(tag: str) -> int:
    """
    Extracts the numeric angle from a tag like 'ANGLE015', 'ANGLE030', 'ANGLE090'.

    Deliberately strict: earlier versions fell back to grabbing any trailing
    digits from the tag, which silently mislabeled fine-tuning-steps tags
    (e.g. 'FT030' -> angle=30) instead of failing. Add fine-tuning-steps
    metrics with add_steps.py, not this script.
    """
    match = re.search(r'^ANGLE(\d+)$', tag, re.IGNORECASE)
    if match:
        return int(match.group(1))

    if re.search(r'FT\d+', tag, re.IGNORECASE):
        raise ValueError(
            f"Tag '{tag}' looks like a fine-tuning-steps tag (FT### format), "
            f"not angle-sweep data (ANGLE###). Use add_steps.py for "
            f"fine-tuning-ablation metrics.json files instead."
        )

    raise ValueError(
        f"Could not extract angle from tag: {tag!r}. Expected ANGLE### "
        f"format (e.g. 'ANGLE015', 'ANGLE030')."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    args = parser.parse_args()

    for path in glob.glob(os.path.join(args.metrics_dir, "*.json")):
        with open(path) as f:
            data = json.load(f)
        angle = extract_angle_from_tag(data["tag"])
        data["angle"] = angle
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Updated {path} -> tag={data['tag']}, angle={angle}")


if __name__ == "__main__":
    main()