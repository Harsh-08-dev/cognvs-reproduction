import argparse
import csv
from pathlib import Path


PERCENT_METRIC_DIRECTIONS = {
    "psnr": "higher",
    "ssim": "higher",
    "lpips": "lower",
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
        {"tag", "steps"}
        | set(PERCENT_METRIC_DIRECTIONS)
        | set(DELTA_METRICS)
    )

    available_columns = set(rows[0].keys())
    missing = required_columns - available_columns

    if missing:
        if "steps" in missing and "angle" in available_columns:
            raise ValueError(
                "This CSV looks like angle-sweep (EXP01) data — it has an "
                "'angle' column but no 'steps' column. analyze_baseline_exp01.py "
                "is for the FT###-based fine-tuning-steps ablation (EXP02) only; "
                "the angle sweep has no FT000-equivalent baseline. Use "
                "scripts/analyze_angle_baseline_exp01.py for angle-sweep "
                "baseline-relative analysis instead."
            )
        raise ValueError(
            f"Metrics CSV is missing required columns: "
            f"{sorted(missing)}"
        )


def find_baseline(rows):
    for row in rows:
        if row["tag"] == "FT000":
            return row

    raise ValueError(
        "Baseline FT000 was not found in the metrics CSV. If this is "
        "angle-sweep (EXP01) data, it has no FT000-equivalent baseline by "
        "design — use scripts/analyze_angle_baseline_exp01.py instead."
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


def analyze(rows):
    validate_columns(rows)
    baseline = find_baseline(rows)

    results = []

    for row in rows:
        output_row = {
            "tag": row["tag"],
            "steps": row["steps"],
        }

        for metric, direction in PERCENT_METRIC_DIRECTIONS.items():
            baseline_value = float(baseline[metric])
            condition_value = float(row[metric])

            output_row[f"{metric}_relative_percent"] = (
                relative_percent(
                    baseline_value,
                    condition_value,
                    direction,
                )
            )

        for metric in DELTA_METRICS:
            baseline_value = float(baseline[metric])
            condition_value = float(row[metric])

            output_row[f"{metric}_delta"] = absolute_delta(
                baseline_value,
                condition_value,
            )

        results.append(output_row)

    return results


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "tag",
        "steps",
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


def print_summary(results):
    print()
    print("EXP01 Baseline-Relative Analysis")
    print("=" * 90)

    for result in results:
        print(
            f"\n{result['tag']} "
            f"(steps={result['steps']})"
        )

        for metric in PERCENT_METRIC_DIRECTIONS:
            value = result[
                f"{metric}_relative_percent"
            ]

            if value is None:
                print(
                    f"  {metric}: N/A "
                    f"(baseline is zero)"
                )
            else:
                print(
                    f"  {metric}: "
                    f"{value:+.2f}%"
                )

        for metric in DELTA_METRICS:
            value = result[f"{metric}_delta"]

            print(
                f"  {metric}_delta: {value:+.6f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute baseline-relative EXP01 analysis "
            "from P3's final_metrics.csv."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to P3's final_metrics.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for baseline-relative CSV output",
    )

    args = parser.parse_args()

    rows = load_metrics(args.csv)
    results = analyze(rows)

    save_results(results, args.output)
    print_summary(results)

    print()
    print(f"Baseline-relative CSV saved to: {args.output}")


if __name__ == "__main__":
    main()