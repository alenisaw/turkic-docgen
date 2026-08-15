from __future__ import annotations

import inspect
import json
import tarfile
from pathlib import Path
from typing import get_type_hints
from unittest import mock

import datasets
import pyarrow.parquet as pq
from typer.main import get_command
from typer.testing import CliRunner

from turkicdocgen.cli import OCR_CORE_PROFILES, app, generate, pipeline
from turkicdocgen.dataset import (
    GenerationOptions,
    generate_dataset,
    generate_dataset_from_options,
)

LEGACY_PROFILE_EXIT_CODE = 2
SMOKE_SAMPLE_COUNT = 4


def _file_hashes(root: Path) -> dict[str, bytes]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if path.name == "run_manifest.json":
                continue
            rel_name = path.relative_to(root).as_posix()
            if path.name == "shard_manifest.json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if "metrics" in data:
                    data["metrics"].pop("timing", None)
                    data["metrics"].pop("page_latencies", None)
                    data["metrics"].pop("peak_rss", None)
                hashes[rel_name] = json.dumps(data, sort_keys=True).encode("utf-8")
            else:
                hashes[rel_name] = path.read_bytes()
    return hashes


def test_generation_api_compatibility_and_runtime_types(tmp_path: Path) -> None:
    positional = tmp_path / "positional"
    keyword = tmp_path / "keyword"
    configured = tmp_path / "configured"

    generate_dataset("visual_300", positional, seed=17, count=1, force=True)
    generate_dataset(
        profile="visual_300",
        out=keyword,
        seed=17,
        count=1,
        force=True,
    )
    generate_dataset_from_options(
        GenerationOptions(
            profile="visual_300",
            out=configured,
            seed=17,
            count=1,
            force=True,
        )
    )

    assert _file_hashes(positional) == _file_hashes(keyword)
    assert _file_hashes(keyword) == _file_hashes(configured)
    assert get_type_hints(GenerationOptions)["out"] is Path


def test_cli_commands_have_honest_callable_signatures(tmp_path: Path) -> None:
    generate_parameters = inspect.signature(generate).parameters
    pipeline_parameters = inspect.signature(pipeline).parameters

    assert tuple(generate_parameters) == (
        "out",
        "profile",
        "count",
        "seed",
        "force",
        "quiet",
        "language",
        "layout",
        "effect",
        "workers",
        "shard_size",
        "resume",
        "shard_range",
        "retry_rejected",
        "verify_only",
        "benchmark_mode",
    )
    assert tuple(pipeline_parameters) == (
        "out",
        "profile",
        "count",
        "seed",
        "force",
        "language",
        "layout",
        "effect",
        "workers",
        "shard_size",
        "resume",
        "shard_range",
        "retry_rejected",
        "verify_only",
        "benchmark_mode",
    )

    direct_out = tmp_path / "direct"
    generate(direct_out, "visual_300", 1, 23, True, True, None, None, None)
    assert (direct_out / "manifest.jsonl").exists()


def test_generate_cli_reports_expected_runtime_failure(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "turkicdocgen.cli.generate_dataset_from_options",
        mock.Mock(side_effect=RuntimeError("worker pool stopped")),
    )

    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--out",
            str(tmp_path / "run"),
            "--profile",
            "visual_300",
            "--count",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "Generation refused" in result.output
    assert "worker pool stopped" in result.output


def test_only_dataset_profiles_are_active() -> None:
    assert set(OCR_CORE_PROFILES) == {
        "visual_300",
        "internal_10k",
        "tiny_25k",
        "medium_50k",
        "large_100k",
    }
    result = CliRunner().invoke(app, ["profiles"])
    assert result.exit_code == 0
    assert "visual_300" in result.stdout
    assert "quality_gate" not in result.stdout


def test_generate_error_escapes_non_ascii_for_legacy_windows_console(
    tmp_path: Path,
) -> None:
    with mock.patch(
        "turkicdocgen.cli.generate_dataset_from_options",
        side_effect=ValueError("bad \u0406"),
    ):
        result = CliRunner().invoke(
            app,
            ["generate", "--out", str(tmp_path / "failed")],
        )

    assert result.exit_code == 1
    assert "bad \\u0406" in result.stdout


def test_legacy_profile_is_rejected_with_migration_text(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--profile",
            "benchmark_5k",
            "--out",
            str(tmp_path / "legacy"),
        ],
    )
    assert result.exit_code == LEGACY_PROFILE_EXIT_CODE
    assert "Legacy profile removed" in result.stdout
    assert "visual_300" in result.stdout


def test_generate_validate_and_pipeline_smoke(tmp_path: Path) -> None:
    out = tmp_path / "visual"
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--profile",
            "visual_300",
            "--count",
            "8",
            "--seed",
            "42",
            "--out",
            str(out),
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert (out / "manifest.jsonl").exists()

    validate = CliRunner().invoke(app, ["validate", str(out)])
    assert validate.exit_code == 0
    assert "Validation passed" in validate.stdout

    pipe_out = tmp_path / "pipe"
    pipeline = CliRunner().invoke(
        app,
        [
            "pipeline",
            "--profile",
            "visual_300",
            "--count",
            "4",
            "--seed",
            "42",
            "--out",
            str(pipe_out),
            "--force",
        ],
    )
    assert pipeline.exit_code == 0
    summary = json.loads((pipe_out / "reports" / "pipeline_summary.json").read_text())
    assert summary["rows"] == SMOKE_SAMPLE_COUNT
    assert summary["validation_errors"] == []


def test_registered_public_commands_are_dataset_surface() -> None:
    commands = get_command(app).commands
    for name in [
        "generate",
        "pipeline",
        "validate",
        "web",
        "profiles",
        "export-det",
        "export-rec",
        "export-release",
        "validate-release",
        "publish-release",
        "cleanup-runs",
        "core-summary",
        "core-split-manifest",
        "core-special-char-stats",
    ]:
        assert name in commands
    removed_terminal_launcher = "tu" + "i"
    assert removed_terminal_launcher not in commands
    assert "generator" not in commands


def test_export_release_writes_local_release_folder(tmp_path: Path) -> None:
    out = tmp_path / "visual"
    release = tmp_path / "release"
    generated = CliRunner().invoke(
        app,
        [
            "generate",
            "--profile",
            "visual_300",
            "--count",
            "4",
            "--seed",
            "42",
            "--out",
            str(out),
            "--force",
        ],
    )
    assert generated.exit_code == 0
    security_report = json.loads(
        (out / "reports" / "security_report.json").read_text(encoding="utf-8")
    )
    assert security_report["unresolved_high_severity"] == 0

    exported = CliRunner().invoke(
        app,
        [
            "export-release",
            "--input",
            str(out),
            "--out",
            str(release),
            "--hf-card",
        ],
    )
    assert exported.exit_code == 0
    assert (release / "README.md").exists()
    assert (release / "CITATION.cff").exists()
    assert (release / "family_index.json").exists()
    assert (release / "dataset_info.json").exists()
    assert (release / "provenance.json").exists()
    assert (release / "checksums.sha256").exists()
    assert (release / "indexes" / "samples.parquet").exists()
    assert (release / "data").exists()

    report = json.loads((release / "provenance.json").read_text(encoding="utf-8"))
    assert report["release_id"] == release.name

    card = (release / "README.md").read_text(encoding="utf-8")
    assert (
        "TurkicDocGen Synthetic Cyrillic Sample" in card
        or "TurkicDocGen Synthetic Cyrillic" in card
    )
    assert "Image format: JPEG" in card
    assert "PaddleOCR" not in card
    family_index = json.loads(
        (release / "family_index.json").read_text(encoding="utf-8")
    )
    for config_name in family_index["configs"]:
        loaded = datasets.load_dataset(str(release), name=config_name)
        expected_counts = family_index["split_counts"][config_name]
        assert {split: len(loaded[split]) for split in loaded} == {
            split: count for split, count in expected_counts.items() if count > 0
        }
        streamed = datasets.load_dataset(str(release), name=config_name, streaming=True)
        assert set(streamed) == {
            split for split, count in expected_counts.items() if count > 0
        }

    validated = CliRunner().invoke(
        app,
        ["validate-release", "--input", str(release)],
    )
    assert validated.exit_code == 0

    # Let's delete a file to make validate fail
    (release / "CITATION.cff").unlink()
    invalid = CliRunner().invoke(
        app,
        ["validate-release", "--input", str(release)],
    )
    assert invalid.exit_code != 0
    assert "missing CITATION.cff" in invalid.stdout


def test_export_release_without_hf_card_is_valid(tmp_path: Path) -> None:
    out = tmp_path / "visual"
    release = tmp_path / "release"
    generated = CliRunner().invoke(
        app,
        [
            "generate",
            "--profile",
            "visual_300",
            "--count",
            "1",
            "--seed",
            "42",
            "--out",
            str(out),
            "--force",
        ],
    )
    assert generated.exit_code == 0

    exported = CliRunner().invoke(
        app,
        [
            "export-release",
            "--input",
            str(out),
            "--out",
            str(release),
            "--no-hf-card",
        ],
    )
    assert exported.exit_code == 0
    assert not (release / "README.md").exists()

    validated = CliRunner().invoke(
        app,
        ["validate-release", "--input", str(release)],
    )
    assert validated.exit_code == 0


def test_export_release_filters_rejected_samples(tmp_path: Path) -> None:
    out = tmp_path / "visual"
    release = tmp_path / "release"
    generated = CliRunner().invoke(
        app,
        [
            "generate",
            "--profile",
            "visual_300",
            "--count",
            "3",
            "--seed",
            "42",
            "--out",
            str(out),
            "--force",
        ],
    )
    assert generated.exit_code == 0

    manifest_path = out / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    rejected = rows[0]
    rows[0] = {**rejected, "qa_ok": False, "qa_issues": ["forced rejection"]}
    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (out / "rejected_samples.jsonl").write_text(
        json.dumps(rows[0], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    exported = CliRunner().invoke(
        app,
        ["export-release", "--input", str(out), "--out", str(release), "--hf-card"],
    )
    assert exported.exit_code == 0

    table = pq.read_table(release / "indexes" / "samples.parquet")
    release_manifest = table.to_pylist()

    assert {row["page_id"] for row in release_manifest} == {
        row["page_id"] for row in rows[1:]
    }

    tar_files = list((release / "data").rglob("*.tar"))
    assert len(tar_files) > 0
    for tar_path in tar_files:
        with tarfile.open(tar_path, "r") as tar:
            names = tar.getnames()
            for name in names:
                assert rejected["page_id"] not in name

    assert (release / "rejected_samples.jsonl").exists()

    validated = CliRunner().invoke(
        app,
        ["validate-release", "--input", str(release)],
    )
    assert validated.exit_code == 0


def test_publish_release_dry_run_prints_hf_cli_plan(
    monkeypatch, tmp_path: Path
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    called = {}

    def fake_publish(release_dir, repo_id, **kwargs):
        called.update({"release_dir": release_dir, "repo_id": repo_id, **kwargs})
        return [
            ["auth", "whoami"],
            ["repos", "create", repo_id, "--type", "dataset", "--exist-ok"],
            [
                "upload-large-folder",
                repo_id,
                release_dir.as_posix(),
            ],
        ]

    monkeypatch.setattr("turkicdocgen.cli.publish_hf_release", fake_publish)

    result = CliRunner().invoke(
        app,
        [
            "publish-release",
            "--input",
            str(release),
            "--repo-id",
            "alenisaw/turkic-docgen",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert called["release_dir"] == release
    assert called["repo_id"] == "alenisaw/turkic-docgen"
    assert called["dry_run"] is True
    assert "Dry run only" in result.stdout
    assert "hf auth whoami" in result.stdout


def test_cleanup_runs_dry_run_lists_generated_roots(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "old").mkdir(parents=True)
    (tmp_path / "release" / "old").mkdir(parents=True)
    (tmp_path / "data" / "synthetic" / "old").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    result = CliRunner().invoke(app, ["cleanup-runs", "--dry-run"])
    assert result.exit_code == 0
    assert "outputs" in result.stdout
    assert "release" in result.stdout
    assert "data" not in result.stdout
    assert (tmp_path / "src").exists()


def test_readme_describes_dataset_path_not_legacy() -> None:
    root = Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "Zone ground truth" in readme
    assert "docs/assets/brand/turkicdocgen-banner.png" in readme
    assert "Release and Publication" not in readme
    assert "publish-release" not in readme
    assert "Hugging Face-" + "ready" not in readme
    assert "quality_gate" not in readme


def test_generate_refuses_protected_output() -> None:
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--profile",
            "visual_300",
            "--count",
            "1",
            "--out",
            "src/bad-output",
            "--force",
        ],
    )
    assert result.exit_code == 1
    assert "Generation refused" in result.stdout


def test_agent_folder_is_ignored_locally() -> None:
    root = Path(__file__).resolve().parent.parent
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".agent/" in gitignore
