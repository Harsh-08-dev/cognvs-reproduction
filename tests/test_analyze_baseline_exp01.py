import pytest

from scripts.analyze_baseline_exp01 import (
    absolute_delta,
    analyze,
    find_baseline,
    relative_percent,
    validate_columns,
)


def make_rows():
    return [
        {
            "tag": "FT000",
            "steps": "0",
            "psnr": "20.0",
            "ssim": "0.90",
            "lpips": "0.10",
            "fid": "20.0",
            "kid_mean": "-0.008",
        },
        {
            "tag": "FT100",
            "steps": "100",
            "psnr": "25.0",
            "ssim": "0.95",
            "lpips": "0.05",
            "fid": "10.0",
            "kid_mean": "-0.013",
        },
    ]


def test_relative_percent_higher_is_better():
    result = relative_percent(
        baseline_value=20.0,
        condition_value=25.0,
        direction="higher",
    )

    assert result == pytest.approx(25.0)


def test_relative_percent_lower_is_better():
    result = relative_percent(
        baseline_value=20.0,
        condition_value=10.0,
        direction="lower",
    )

    assert result == pytest.approx(50.0)


def test_absolute_delta():
    result = absolute_delta(
        baseline_value=-0.008,
        condition_value=-0.013,
    )

    assert result == pytest.approx(-0.005)


def test_find_baseline():
    rows = make_rows()

    baseline = find_baseline(rows)

    assert baseline["tag"] == "FT000"


def test_find_baseline_rejects_missing_baseline():
    rows = make_rows()
    rows = [row for row in rows if row["tag"] != "FT000"]

    with pytest.raises(ValueError):
        find_baseline(rows)


def test_validate_columns_rejects_missing_metric():
    rows = make_rows()

    del rows[0]["fid"]

    with pytest.raises(ValueError):
        validate_columns(rows)


def test_analyze_produces_expected_results():
    results = analyze(make_rows())

    assert len(results) == 2

    baseline = results[0]
    ft100 = results[1]

    assert baseline["psnr_relative_percent"] == pytest.approx(0.0)
    assert baseline["kid_mean_delta"] == pytest.approx(0.0)

    assert ft100["psnr_relative_percent"] == pytest.approx(25.0)
    assert ft100["ssim_relative_percent"] == pytest.approx(
        (0.95 - 0.90) / 0.90 * 100
    )
    assert ft100["lpips_relative_percent"] == pytest.approx(50.0)
    assert ft100["fid_relative_percent"] == pytest.approx(50.0)
    assert ft100["kid_mean_delta"] == pytest.approx(-0.005)
