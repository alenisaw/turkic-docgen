from __future__ import annotations

import json
from pathlib import Path

import pytest

from turkicdocgen.page_planning.content.audit import generate_iteration5_reports


def test_expected_count_and_gates(tmp_path: Path) -> None:
    # 1. Setup mock raw manifest rows
    rows = []
    # Generate 100 rows
    for i in range(100):
        # We will use "kk" and "ky" for languages
        lang = "kk" if i % 2 == 0 else "ky"
        # We will use "book_page_single_column" for layout
        layout = "book_page_single_column"
        # Quality profile
        quality = "clean"
        # Simple zone structure
        zones = [
            {
                "zone_id": "body",
                "zone_type": "body",
                "reading_order": 1,
                "style": {
                    "font_family": "Liberation Serif",
                    "font_size_px": 24,
                },
                "text": "Тест хабарламасы. Екінші сөйлем.",
            }
        ]
        rows.append(
            {
                "image_path": f"shards/shard-0000{i // 50}/images/page_{i:06d}.png",
                "layout_id": layout,
                "language_mix": lang,
                "orientation": "portrait",
                "quality_profile": quality,
                "effect_chain": [],
                "layout_variant": "book_var_01",
                "layout_density": "standard",
                "zones": zones,
            }
        )

    # Run reports with a private profile ("visual_300") which should not fail on violations
    generate_iteration5_reports(tmp_path, rows, "visual_300")

    report_file = tmp_path / "reports" / "distribution_report.json"
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))

    # Verify expected-count calculation
    # Since total samples = 100, kk target = 0.38, expected count for kk is 38.0
    lang_rep = report["marginal_distributions"]["language"]["buckets"]
    assert lang_rep["kk"]["expected_count"] == 38.0
    assert lang_rep["kk"]["actual_count"] == 50

    # Verify deterministic report ordering:
    # Keys of marginal distributions should be deterministic
    assert list(report["marginal_distributions"].keys()) == [
        "language",
        "quality_profile",
        "layout_id",
        "layout_family",
        "orientation",
    ]

    # Shard drift detection:
    # Since shard-00000 has 50 samples and shard-00001 has 50 samples, and lang mix has same distribution, drift should be 0.0
    assert report["marginal_distributions"]["language"]["shard_level_drift"] == 0.0

    # Summary consistency
    summary_file = tmp_path / "reports" / "distribution_summary.md"
    assert summary_file.exists()
    summary_text = summary_file.read_text(encoding="utf-8")
    assert "# Multidimensional Distribution Diversity Report Summary" in summary_text


def test_hard_error_vs_warning(tmp_path: Path) -> None:
    # The final public profile must turn distribution violations into hard failures.
    rows = [
        {
            "image_path": "shards/shard-00000/images/page_000000.png",
            "layout_id": "book_page_single_column",
            "language_mix": "kk",
            "orientation": "portrait",
            "quality_profile": "clean",
            "effect_chain": [],
            "layout_variant": "book_var_01",
            "layout_density": "standard",
            "zones": [],
        }
    ]
    # This has 1 sample, which violates many gates (like Gate 5 orientations/columns, etc.)
    # Only large_100k is a publication profile; 25k/50k are local validation stages.
    with pytest.raises(ValueError) as excinfo:
        generate_iteration5_reports(tmp_path, rows, "large_100k")
    assert "Hard distribution gate violations encountered" in str(excinfo.value)


def test_sparse_bucket_and_unknown_category(tmp_path: Path) -> None:
    # Unknown layout or language categories should not crash the generator, they are audited safely
    rows = [
        {
            "image_path": "shards/shard-00000/images/page_000000.png",
            "layout_id": "unknown_layout_id",
            "language_mix": "unknown_lang_mix",
            "orientation": "portrait",
            "quality_profile": "clean",
            "effect_chain": [],
            "layout_variant": "unknown_variant",
            "layout_density": "standard",
            "zones": [],
        }
    ]
    # Running with private profile should compile reports safely without raising exceptions
    generate_iteration5_reports(tmp_path, rows, "visual_300")


def test_joint_report_includes_expected_but_absent_buckets(tmp_path: Path) -> None:
    rows = [
        {
            "page_id": f"p{i}",
            "layout_id": "book_page_single_column",
            "language_mix": "kk",
            "quality_profile": "clean",
            "effect_chain": [],
            "orientation": "portrait",
            "layout_variant": "book_var_01",
            "layout_density": "standard",
            "shard_id": f"shard-{i // 500:05d}",
            "zones": [],
        }
        for i in range(10000)
    ]
    with pytest.raises(ValueError, match="Hard distribution gate"):
        generate_iteration5_reports(tmp_path, rows, "visual_300")
    report = json.loads(
        (tmp_path / "reports" / "distribution_report.json").read_text(encoding="utf-8")
    )
    absent = [
        bucket
        for bucket in report["joint_distribution_layout_quality_effect_lang"]["buckets"]
        if bucket["actual_count"] == 0 and bucket["expected_count"] >= 25
    ]
    assert absent
    assert report["marginal_distributions"]["language"]["shard_level_drift"] == 0.0
    report_file = tmp_path / "reports" / "distribution_report.json"
    assert report_file.exists()
