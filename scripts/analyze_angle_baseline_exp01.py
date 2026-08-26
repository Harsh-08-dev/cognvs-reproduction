"""
scripts/analyze_angle_baseline_exp01.py

Angle-aware counterpart to analyze_baseline_exp01.py, for EXP01's zero-shot
angle-sweep data (ANGLE015/030/060/090 tags, no paired ground truth).

The angle sweep has no FT000-equivalent baseline condition — there's no
"zero deviation" fine-tuned run to compare against — and no PSNR/SSIM/LPIPS,
since those require paired ground truth, which doesn't exist for in-the-wild
sequences at novel angles beyond the captured view (see
docs/analysis/evaluation_protocol.md's "No-GT mode" section; this matches
the paper's own Fig. 10, which is qualitative-only for the same reason). So
this script differs from analyze_baseline_exp01.py in two ways:
  - the baseline is the smallest-angle row actually run (closest to the
    source view), not a fixed "FT000" tag
  - only FID (relative %) and KID mean (absolute delta) are compared, since
    those are the only metrics EXP01 computes without GT; a metric missing
    for a row (e.g. KID skipped when <2 frames were available) is reported
    as null rather than raising

Usage (run from repo root):
    python -m scripts.analyze_angle_baseline_exp01 --csv results/final_metrics.csv --output results/relative_to_min_angle.csv
"""
import argparse
import csv
from pathlib import Path


PERCENT_METRIC_DIRECTIONS = {
    "fid": "lower",
}

DELTA_METRICS = {
    "kid_mean": "lower",
}


def load_metrics(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metrics CSV not found: {csv_path}"
        )

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(
            f"Metrics CSV is empty: {csv_path}"
        )

    return rows


def validate_columns(rows):
    required_columns = (
        {"tag", "angle"}
        | set(PERCENT_METRIC_DIRECTIONS)
        | set(DELTA_METRICS)
    )

    available_columns = set(rows[0].keys())
    missing = required_columns - available_columns

    if missing:
        if "angle" in missing and "steps" in available_columns:
            raise ValueError(
                "This CSV looks like fine-tuning-steps (EXP02) data — it "
                "has a 'steps' column but no 'angle' column. "
                "analyze_angle_baseline_exp01.py is for angle-sweep (EXP01) "
                "data only. Use scripts/analyze_baseline_exp01.py for "
                "FT###-based data instead."
            )
        raise ValueError(
            f"Metrics CSV is missing required columns: "
            f"{sorted(missing)}"
        )


def find_baseline(rows):
    """
    The angle sweep has no zero-angle / FT000-equivalent condition — use
    whichever angle was actually run smallest as the baseline instead.
    """
    try:
        return min(rows, key=lambda row: float(row["angle"]))
    except (KeyError, ValueError) as e:
        raise ValueError(
            f"Could not determine an angle-sweep baseline: {e}"
        )


def relative_percent(baseline_value, condition_value, direction):
    """
    Returns percentage improvement relative to baseline.

    Positive = improvement.
    Negative = degradation.
    """
    if baseline_value == 0:
        return None

    if direction == "higher":
        return (
            (condition_value - baseline_value)
            / baseline_value
            * 100
        )

    if direction == "lower":
        return (
            (baseline_value - condition_value)
            / baseline_value
            * 100
        )

    raise ValueError(
        f"Unknown metric direction: {direction}"
    )


def absolute_delta(baseline_value, condition_value):
    """
    Returns raw change relative to baseline.

    For lower-is-better metrics:
    negative = improvement
    positive = degradation.
    """
    return condition_value - baseline_value


def _parse_float_or_none(value):
    """CSV empty-string (written from a JSON null, e.g. KID skipped for
    <2 frames) means the metric wasn't computed for that row — treat as
    missing, not a parse error."""
    if value is None or value == "" or value == "None":
        return None
    return float(value)


def analyze(rows):
    validate_columns(rows)
    baseline = find_baseline(rows)
    baseline_angle = float(baseline["angle"])

    results = []

    for row in rows:
        output_row = {
            "tag": row["tag"],
            "angle": row["angle"],
        }

        for metric, direction in PERCENT_METRIC_DIRECTIONS.items():
            baseline_value = _parse_float_or_none(baseline[metric])
            condition_value = _parse_float_or_none(row[metric])

            if baseline_value is None or condition_value is None:
                output_row[f"{metric}_relative_percent"] = None
                continue

            output_row[f"{metric}_relative_percent"] = (
                relative_percent(
                    baseline_value,
                    condition_value,
                    direction,
                )
            )

        for metric in DELTA_METRICS:
            baseline_value = _parse_float_or_none(baseline[metric])
            condition_value = _parse_float_or_none(row[metric])

            if baseline_value is None or condition_value is None:
                output_row[f"{metric}_delta"] = None
                continue

            output_row[f"{metric}_delta"] = absolute_delta(
                baseline_value,
                condition_value,
            )

        results.append(output_row)

    return results, baseline_angle


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "tag",
        "angle",
    ]

    for metric in PERCENT_METRIC_DIRECTIONS:
        fieldnames.append(
            f"{metric}_relative_percent"
        )

    for metric in DELTA_METRICS:
        fieldnames.append(
            f"{metric}_delta"
        )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def print_summary(results, baseline_angle):
    print()
    print(f"EXP01 Angle-Sweep Baseline-Relative Analysis (baseline: {baseline_angle:g} deg)")
    print("=" * 90)

    for result in results:
        print(
            f"\n{result['tag']} "
            f"(angle={result['angle']} deg)"
        )

        for metric in PERCENT_METRIC_DIRECTIONS:
            value = result[
                f"{metric}_relative_percent"
            ]

            if value is None:
                print(
                    f"  {metric}: N/A "
                    f"(not computed for this row)"
                )
            else:
                print(
                    f"  {metric}: "
                    f"{value:+.2f}%"
                )

        for metric in DELTA_METRICS:
            value = result[f"{metric}_delta"]

            if value is None:
                print(f"  {metric}_delta: N/A (not computed for this row)")
            else:
                print(
                    f"  {metric}_delta: {value:+.6f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute baseline-relative EXP01 angle-sweep analysis "
            "from P3's final_metrics.csv, using the smallest angle "
            "actually run as the baseline."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to P3's final_metrics.csv (must have an 'angle' column, e.g. via add_angle.py)",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for baseline-relative CSV output",
    )

    args = parser.parse_args()

    rows = load_metrics(args.csv)
    results, baseline_angle = analyze(rows)

    save_results(results, args.output)
    print_summary(results, baseline_angle)

    print()
    print(f"Baseline-relative CSV saved to: {args.output}")


if __name__ == "__main__":
    main()
