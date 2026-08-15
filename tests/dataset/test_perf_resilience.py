from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from turkicdocgen.dataset import (
    GenerationOptions,
    _peak_rss_bytes,
    _publish_shard_directory,
    _verify_shard,
    build_run_signature,
    check_free_space,
    generate_dataset_from_options,
    parse_shard_range,
    run_single_benchmark_run,
    select_benchmark_run,
    shard_manifest_digest,
)


def test_free_space_check_triggers_io_error(tmp_path: Path) -> None:
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value.free = 50 * 1024 * 1024
        with pytest.raises(IOError, match="Disk space is low"):
            check_free_space(tmp_path)


def test_free_space_check_passes_on_sufficient_space(tmp_path: Path) -> None:
    with patch("shutil.disk_usage") as mock_usage:
        mock_usage.return_value.free = 200 * 1024 * 1024
        check_free_space(tmp_path)


def test_benchmark_mode_generates_report(tmp_path: Path) -> None:
    options = GenerationOptions(
        profile="visual_300",
        out=tmp_path,
        seed=42,
        count=1,
        force=True,
        benchmark_mode=True,
    )

    report_path = generate_dataset_from_options(options)

    assert report_path.exists()
    assert report_path.name == "performance_report.json"

    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "benchmark_runs" in report_data
    assert "recommendations" in report_data

    runs = report_data["benchmark_runs"]
    assert len(runs) > 0
    for run in runs:
        assert "worker_count" in run
        assert "throughput_pages_per_sec" in run
        assert "p50_latency_sec" in run
        assert "p95_latency_sec" in run
        assert "peak_rss_mb" in run
        assert "bytes_per_page" in run

    recs = report_data["recommendations"]
    assert "selected_worker_count" in recs
    assert "rationale" in recs
    assert recs["projected_pages"] == 100_000
    assert "projected_wall_time_hours_100k" in recs
    assert "projected_storage_gb_100k" in recs


def test_timing_and_memory_metrics_recorded(tmp_path: Path) -> None:
    options = GenerationOptions(
        profile="visual_300",
        out=tmp_path,
        seed=123,
        count=1,
        force=True,
        workers=1,
    )
    generate_dataset_from_options(options)

    shard_manifest_path = tmp_path / "shards" / "shard-00000" / "shard_manifest.json"
    assert shard_manifest_path.exists()

    manifest_data = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
    metrics = manifest_data.get("metrics", {})

    assert "timing" in metrics
    timing = metrics["timing"]
    assert "content_planning" in timing
    assert "layout_planning" in timing
    assert "glyph_measurement" in timing
    assert "rendering" in timing
    assert "effects" in timing
    assert "qa" in timing
    assert "encoding" in timing
    assert "serialization" in timing

    assert "page_latencies" in metrics
    assert len(metrics["page_latencies"]) == 1
    assert "peak_rss" in metrics
    assert metrics["peak_rss"] > 0


def test_benchmark_uses_production_retry_mode(tmp_path: Path) -> None:
    options = GenerationOptions(
        profile="visual_300",
        out=tmp_path,
        seed=42,
        count=1,
        force=True,
    )
    benchmark_out = tmp_path / "benchmark_w1"

    def fake_generate(run_options: GenerationOptions) -> Path:
        (benchmark_out / "manifest.jsonl").write_text(
            '{"page_id":"page-1"}\n', encoding="utf-8"
        )
        assert run_options.retry_rejected is True
        return benchmark_out

    with patch(
        "turkicdocgen.dataset.generate_dataset_from_options",
        side_effect=fake_generate,
    ):
        result = run_single_benchmark_run(
            1,
            1,
            options,
            42,
            "png",
            {},
            {},
        )

    assert result["failure_rate"] == 0.0


def test_peak_rss_is_available_for_current_process() -> None:
    assert _peak_rss_bytes() > 0


def test_publish_shard_retries_transient_windows_lock(tmp_path: Path) -> None:
    source = tmp_path / "shard.tmp"
    destination = tmp_path / "shard"
    source.mkdir()
    original_rename = Path.rename
    attempts = 0

    def flaky_rename(path: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient lock")
        return original_rename(path, target)

    with (
        patch("pathlib.Path.rename", new=flaky_rename),
        patch("turkicdocgen.dataset.time.sleep"),
    ):
        _publish_shard_directory(source, destination)

    assert attempts == 3
    assert destination.is_dir()


def test_shard_digest_ignores_only_volatile_telemetry() -> None:
    first = {
        "accepted_count": 1,
        "files": {"page.png": {"sha256": "abc", "size": 10}},
        "metrics": {
            "layouts": {"book": 1},
            "timing": {"rendering": 1.0},
            "page_latencies": [1.0],
            "peak_rss": 100,
        },
    }
    second = {
        **first,
        "metrics": {
            **first["metrics"],
            "timing": {"rendering": 9.0},
            "page_latencies": [9.0],
            "peak_rss": 900,
        },
    }
    assert shard_manifest_digest(first) == shard_manifest_digest(second)
    second["accepted_count"] = 2
    assert shard_manifest_digest(first) != shard_manifest_digest(second)


def test_run_signature_tracks_generator_implementation() -> None:
    with patch(
        "turkicdocgen.dataset.get_generator_implementation_hash",
        return_value="implementation-a",
    ):
        first = build_run_signature(
            profile="visual_300",
            master_seed=42,
            overrides={},
            image_format="png",
        )
    with patch(
        "turkicdocgen.dataset.get_generator_implementation_hash",
        return_value="implementation-b",
    ):
        second = build_run_signature(
            profile="visual_300",
            master_seed=42,
            overrides={},
            image_format="png",
        )

    assert first["generator_implementation_hash"] == "implementation-a"
    assert first != second


@pytest.mark.parametrize("value", ["3-1", "1-2-3", "-1", "9", "bad"])
def test_parse_shard_range_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_shard_range(value, num_shards=3)


def test_verify_shard_rejects_tampered_manifest_and_extra_file(
    tmp_path: Path,
) -> None:
    options = GenerationOptions(
        profile="visual_300",
        out=tmp_path,
        seed=123,
        count=1,
        force=True,
    )
    generate_dataset_from_options(options)
    shard_dir = tmp_path / "shards" / "shard-00000"
    assert _verify_shard(shard_dir, 0, 1)

    manifest_path = shard_dir / "shard_manifest.json"
    original = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original)
    manifest["accepted_count"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not _verify_shard(shard_dir, 0, 1)

    manifest_path.write_text(original, encoding="utf-8")
    (shard_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    assert not _verify_shard(shard_dir, 0, 1)


def test_partial_shard_range_cannot_publish_incomplete_root(tmp_path: Path) -> None:
    options = GenerationOptions(
        profile="visual_300",
        out=tmp_path,
        seed=123,
        count=2,
        force=True,
        shard_size=1,
        shard_range="0",
    )
    with pytest.raises(RuntimeError, match="dataset is incomplete"):
        generate_dataset_from_options(options)
    assert not (tmp_path / "run_manifest.json").exists()


def test_benchmark_selection_rejects_fast_failing_configuration() -> None:
    runs = [
        {
            "worker_count": 8,
            "throughput_pages_per_sec": 5.0,
            "p95_latency_sec": 2.0,
            "peak_rss_mb": 1000.0,
            "failure_rate": 0.0,
        },
        {
            "worker_count": 32,
            "throughput_pages_per_sec": 20.0,
            "p95_latency_sec": 10.0,
            "peak_rss_mb": 8000.0,
            "failure_rate": 0.1,
        },
    ]
    assert select_benchmark_run(runs)["worker_count"] == 8


def test_benchmark_selection_rejects_unbounded_memory_growth() -> None:
    runs = [
        {
            "worker_count": 8,
            "throughput_pages_per_sec": 5.0,
            "p95_latency_sec": 2.0,
            "peak_rss_mb": 1000.0,
            "failure_rate": 0.0,
        },
        {
            "worker_count": 32,
            "throughput_pages_per_sec": 6.0,
            "p95_latency_sec": 2.0,
            "peak_rss_mb": 5000.0,
            "failure_rate": 0.0,
        },
    ]
    assert select_benchmark_run(runs)["worker_count"] == 8
