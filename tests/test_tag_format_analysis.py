"""
Covers issue #6: add_steps.py / add_angle.py used to fall back to grabbing
any trailing digits from a tag, which silently mislabeled the other
experiment's data (ANGLE030 -> steps=30, or FT030 -> angle=30) instead of
failing. analyze_baseline_exp01.py is hardcoded around FT###/FT000, which
doesn't exist for angle-sweep data, and previously only failed with a
generic missing-columns error instead of explaining why. Fixed by:
  - making both tag extractors strict, with a clear cross-pointer error
    when they see the other experiment's tag shape
  - giving analyze_baseline_exp01.py a specific, guided error for
    angle-sweep-shaped CSVs
  - adding scripts/analyze_angle_baseline_exp01.py, the angle-sweep
    counterpart docs/p2_analysis_workflow.md already promised
"""
import pytest

from src.evaluation.add_steps import extract_steps_from_tag
from src.evaluation.add_angle import extract_angle_from_tag
from scripts.analyze_baseline_exp01 import validate_columns as ft_validate_columns
from scripts.analyze_angle_baseline_exp01 import (
    analyze as angle_analyze,
    find_baseline as angle_find_baseline,
    validate_columns as angle_validate_columns,
)


# -- add_steps.py / add_angle.py: no more silent cross-mislabeling --------

def test_add_steps_accepts_ft_tag():
    assert extract_steps_from_tag("FT100") == 100


def test_add_steps_rejects_angle_tag_with_clear_error():
    with pytest.raises(ValueError, match="angle-sweep"):
        extract_steps_from_tag("ANGLE030")


def test_add_steps_rejects_unrecognized_tag():
    with pytest.raises(ValueError):
        extract_steps_from_tag("mystery_tag")


def test_add_angle_accepts_angle_tag():
    assert extract_angle_from_tag("ANGLE030") == 30


def test_add_angle_rejects_ft_tag_with_clear_error():
    with pytest.raises(ValueError, match="fine-tuning-steps"):
        extract_angle_from_tag("FT100")


def test_add_angle_rejects_unrecognized_tag():
    with pytest.raises(ValueError):
        extract_angle_from_tag("mystery_tag")


# -- analyze_baseline_exp01.py: guided error on angle-sweep data ----------

def make_angle_rows():
    return [
        {
            "tag": "ANGLE015", "angle": "15",
            "psnr": "", "ssim": "", "lpips": "",
            "fid": "20.0", "kid_mean": "-0.008",
        },
        {
            "tag": "ANGLE030", "angle": "30",
            "psnr": "", "ssim": "", "lpips": "",
            "fid": "25.0", "kid_mean": "-0.004",
        },
    ]


def test_ft_analyzer_rejects_angle_sweep_csv_with_guided_error():
    with pytest.raises(ValueError, match="analyze_angle_baseline_exp01"):
        ft_validate_columns(make_angle_rows())


# -- scripts/analyze_angle_baseline_exp01.py -------------------------------

def test_angle_analyzer_rejects_ft_csv_with_guided_error():
    ft_rows = [
        {
            "tag": "FT000", "steps": "0",
            "psnr": "20.0", "ssim": "0.9", "lpips": "0.1",
            "fid": "20.0", "kid_mean": "-0.008",
        },
    ]
    with pytest.raises(ValueError, match="analyze_baseline_exp01"):
        angle_validate_columns(ft_rows)


def test_angle_analyzer_finds_smallest_angle_as_baseline():
    baseline = angle_find_baseline(make_angle_rows())
    assert baseline["tag"] == "ANGLE015"


def test_angle_analyzer_produces_expected_results():
    results, baseline_angle = angle_analyze(make_angle_rows())

    assert baseline_angle == pytest.approx(15.0)
    assert len(results) == 2

    baseline_row, condition_row = results
    assert baseline_row["fid_relative_percent"] == pytest.approx(0.0)
    assert baseline_row["kid_mean_delta"] == pytest.approx(0.0)

    # fid is lower-is-better: condition (25.0) worse than baseline (20.0)
    # -> negative relative percent (degradation)
    assert condition_row["fid_relative_percent"] == pytest.approx(-25.0)
    assert condition_row["kid_mean_delta"] == pytest.approx(0.004)


def test_angle_analyzer_treats_missing_metric_as_null_not_error():
    rows = make_angle_rows()
    rows[1]["kid_mean"] = ""  # e.g. KID skipped for <2 frames

    results, _ = angle_analyze(rows)

    assert results[1]["kid_mean_delta"] is None
    # fid still computed even though kid_mean was missing for this row
    assert results[1]["fid_relative_percent"] is not None
