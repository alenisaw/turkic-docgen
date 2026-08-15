from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from turkicdocgen.hf.release import (
    export_hf_release,
    publish_hf_release,
    sha256_file,
    validate_hf_release,
)
from turkicdocgen.release_config import RELEASE_CONFIG_TARGETS


def _write_minimal_valid_release(release_dir: Path):
    release_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write metadata files
    (release_dir / "README.md").write_text(
        """---
license: cc-by-4.0
configs:
  - config_name: tiny
    data_files:
      - split: train
        path: indexes/tiny.parquet
  - config_name: medium
    data_files:
      - split: train
        path: indexes/medium.parquet
  - config_name: large
    data_files:
      - split: train
        path: indexes/large.parquet
---
# TurkicDocGen Release
""",
        encoding="utf-8",
    )

    (release_dir / "CITATION.cff").write_text(
        """cff-version: 1.2.0
message: "citation message"
doi: "pending"
""",
        encoding="utf-8",
    )

    (release_dir / "family_index.json").write_text(
        json.dumps(
            {
                "family_name": "TurkicDocGen Synthetic Cyrillic",
                "configs": ["tiny", "medium", "large"],
                "nested": {"tiny": "medium", "medium": "large"},
            }
        ),
        encoding="utf-8",
    )

    (release_dir / "dataset_info.json").write_text(
        json.dumps({"description": "info desc"}), encoding="utf-8"
    )
    reports_dir = release_dir / "reports"
    reports_dir.mkdir()
    (reports_dir / "security_report.json").write_text(
        json.dumps({"status": "passed", "unresolved_high_severity": 0}),
        encoding="utf-8",
    )

    # 2. Write indexes
    indexes_dir = release_dir / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    sample_data = {
        "page_id": "p1",
        "layout_id": "simple_prose_page",
        "split": "train",
        "subsets": ["tiny", "medium", "large"],
        "tar_path": "data/train/train-000000.tar",
        "image_filename": "p1.png",
        "ocr_text": "hello",
        "ocr_bboxes": [[0, 0, 10, 10]],
        "ocr_labels": ["hello"],
        "ocr_region_ids": ["r1"],
        "ocr_json": "[]",
        "zones_json": "[]",
        "sft_json": "{}",
    }
    table = pa.Table.from_pylist([sample_data])
    pq.write_table(table, indexes_dir / "samples.parquet")
    pq.write_table(table, indexes_dir / "tiny.parquet")
    pq.write_table(table, indexes_dir / "medium.parquet")
    pq.write_table(table, indexes_dir / "large.parquet")

    # 3. Write data TAR
    data_train_dir = release_dir / "data" / "train"
    data_train_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_train_dir / "train-000000.tar"
    with tarfile.open(tar_path, "w") as tar:
        # page_id.png
        png_buffer = io.BytesIO()
        Image.new("RGB", (2, 2), "white").save(png_buffer, format="PNG")
        png_data = png_buffer.getvalue()
        tarinfo = tarfile.TarInfo(name="p1.png")
        tarinfo.size = len(png_data)
        tar.addfile(tarinfo, io.BytesIO(png_data))

        # page_id.json
        json_data = json.dumps({"page_id": "p1"}).encode("utf-8")
        tarinfo_json = tarfile.TarInfo(name="p1.json")
        tarinfo_json.size = len(json_data)
        tar.addfile(tarinfo_json, io.BytesIO(json_data))

    # We also need empty validation / test dirs or mock data to avoid validation errors
    (release_dir / "data" / "validation").mkdir(parents=True, exist_ok=True)
    (release_dir / "data" / "test").mkdir(parents=True, exist_ok=True)

    # Let's compute provenance.json
    inventory_hashes = {}
    for path in sorted(release_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(release_dir).as_posix()
            inventory_hashes[rel] = sha256_file(path)

    provenance_data = {
        "release_id": release_dir.name,
        "git_commit": "mock_commit",
        "configurations_hash": "mock_config_hash",
        "inventory_hashes": inventory_hashes,
        "metadata": {"created_at": "2026-06-13T09:38:34-07:00", "generator": "mock"},
    }
    (release_dir / "provenance.json").write_text(
        json.dumps(provenance_data), encoding="utf-8"
    )

    # checksums.sha256
    checksum_lines = []
    for path in sorted(release_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(release_dir).as_posix()
            if rel != "checksums.sha256":
                h = sha256_file(path)
                checksum_lines.append(f"{h}  {rel}")
    (release_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


def test_valid_release_passes_validation(tmp_path: Path):
    release_dir = tmp_path / "valid_release"
    _write_minimal_valid_release(release_dir)
    errors = validate_hf_release(release_dir)
    assert not errors, f"Expected no errors, got: {errors}"


def test_validation_detects_missing_metadata(tmp_path: Path):
    release_dir = tmp_path / "missing_meta"
    _write_minimal_valid_release(release_dir)
    (release_dir / "CITATION.cff").unlink()
    errors = validate_hf_release(release_dir)
    assert any("missing CITATION.cff" in err for err in errors)


def test_checksum_covers_all_files(tmp_path: Path):
    release_dir = tmp_path / "checksum_test"
    _write_minimal_valid_release(release_dir)

    (release_dir / "extra.txt").write_text("extra content")
    errors = validate_hf_release(release_dir)
    assert any(
        "file not listed in checksums.sha256: extra.txt" in err for err in errors
    )


def test_local_loading_configs(tmp_path: Path):
    release_dir = tmp_path / "loading_test"
    _write_minimal_valid_release(release_dir)

    # Load each config in normal and streaming mode
    for config_name in ["tiny", "medium", "large"]:
        # normal mode
        dataset = datasets.load_dataset(str(release_dir), name=config_name)
        assert len(dataset["train"]) == 1
        assert dataset["train"][0]["page_id"] == "p1"
        assert dataset["train"][0]["ocr_text"] == "hello"

        # streaming mode
        dataset_stream = datasets.load_dataset(
            str(release_dir), name=config_name, streaming=True
        )
        samples = list(dataset_stream["train"].take(1))
        assert len(samples) == 1
        assert samples[0]["page_id"] == "p1"
        assert samples[0]["ocr_text"] == "hello"


def test_validation_rejects_fake_image_payload(tmp_path: Path) -> None:
    release_dir = tmp_path / "fake_image"
    _write_minimal_valid_release(release_dir)
    tar_path = release_dir / "data" / "train" / "train-000000.tar"
    with tarfile.open(tar_path, "w") as tar:
        png_data = b"\x89PNG\r\n\x1a\nfake_png_data"
        png_info = tarfile.TarInfo(name="p1.png")
        png_info.size = len(png_data)
        tar.addfile(png_info, io.BytesIO(png_data))
        json_data = json.dumps({"page_id": "p1"}).encode("utf-8")
        json_info = tarfile.TarInfo(name="p1.json")
        json_info.size = len(json_data)
        tar.addfile(json_info, io.BytesIO(json_data))
    assert any("invalid image" in error for error in validate_hf_release(release_dir))


def test_export_rejects_missing_image_and_unsafe_page_id(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "page_id": "valid-page",
                "qa_ok": True,
                "image": "images/missing.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        export_hf_release(input_dir, tmp_path / "missing-release")
    except ValueError as exc:
        assert "missing source image" in str(exc)
    else:
        raise AssertionError("Missing source image was replaced instead of rejected")
    assert not (tmp_path / "missing-release").exists()

    (input_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "page_id": "../escape",
                "qa_ok": True,
                "image": "images/page.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        export_hf_release(input_dir, tmp_path / "unsafe-release")
    except ValueError as exc:
        assert "unsafe page_id" in str(exc)
    else:
        raise AssertionError("Unsafe page_id was accepted")


def test_release_is_not_published_after_partial_write_failure(
    tmp_path: Path, monkeypatch
) -> None:
    import turkicdocgen.hf.release as release_module

    input_dir = tmp_path / "input"
    image_dir = input_dir / "images"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(image_dir / "page.png")
    (input_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "page_id": "page",
                "qa_ok": True,
                "image": "images/page.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        release_module.pq.ParquetWriter,
        "write_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    final_release = tmp_path / "release"
    with pytest.raises(OSError, match="disk full"):
        export_hf_release(input_dir, final_release)
    assert not final_release.exists()


def test_release_sidecars_may_be_in_different_page_order(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    image_dir = input_dir / "images"
    image_dir.mkdir(parents=True)
    rows = []
    for index, page_id in enumerate(("page-a", "page-b")):
        Image.new("RGB", (8, 8), "white").save(image_dir / f"{page_id}.png")
        rows.append(
            {
                "page_id": page_id,
                "qa_ok": True,
                "image": f"images/{page_id}.png",
                "layout_id": "simple_prose_page",
                "split": "train",
                "nested_rank": index,
                "in_tiny": True,
                "in_medium": True,
                "in_large": True,
                "subsets": ["tiny", "medium", "large"],
            }
        )
    for name, payload in {
        "manifest.jsonl": rows,
        "zone_gt.jsonl": [
            {"page_id": "page-b", "zones": [{"text": "zone-b"}]},
            {"page_id": "page-a", "zones": [{"text": "zone-a"}]},
        ],
        "sft.jsonl": [
            {"page_id": "page-b", "prompt": "b", "response": ["b"]},
            {"page_id": "page-a", "prompt": "a", "response": ["a"]},
        ],
        "ocr_det.jsonl": [
            {"page_id": "page-a", "region_id": "r-a", "bbox": [0, 0, 1, 1]},
            {"page_id": "page-b", "region_id": "r-b", "bbox": [1, 1, 2, 2]},
        ],
        "ocr_rec.jsonl": [
            {"page_id": "page-b", "region_id": "r-b", "text": "text-b"},
            {"page_id": "page-a", "region_id": "r-a", "text": "text-a"},
        ],
    }.items():
        with (input_dir / name).open("w", encoding="utf-8") as handle:
            for row in payload:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    release_dir = export_hf_release(input_dir, tmp_path / "release")

    table = pq.read_table(release_dir / "indexes" / "samples.parquet")
    by_page = {row["page_id"]: row for row in table.to_pylist()}
    assert by_page["page-a"]["ocr_text"] == "text-a"
    assert by_page["page-b"]["ocr_text"] == "text-b"
    assert not (release_dir / ".release_index.sqlite3").exists()


def test_validation_rejects_special_tar_members_and_checksum_traversal(
    tmp_path: Path,
) -> None:
    release_dir = tmp_path / "malicious_release"
    _write_minimal_valid_release(release_dir)
    tar_path = release_dir / "data" / "train" / "train-000000.tar"
    with tarfile.open(tar_path, "w") as tar:
        link = tarfile.TarInfo(name="p1.png")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        tar.addfile(link)
        payload = json.dumps({"page_id": "p1"}).encode("utf-8")
        info = tarfile.TarInfo(name="p1.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    checksum_path = release_dir / "checksums.sha256"
    checksum_path.write_text(
        checksum_path.read_text(encoding="utf-8") + f"{'0' * 64}  ../outside.txt\n",
        encoding="utf-8",
    )

    errors = validate_hf_release(release_dir)
    assert any("non-regular TAR member" in error for error in errors)
    assert any("unsafe checksum path" in error for error in errors)


def test_validation_rejects_legacy_readme_config_names(tmp_path: Path) -> None:
    release_dir = tmp_path / "legacy_readme_configs"
    _write_minimal_valid_release(release_dir)
    readme = release_dir / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            "  - config_name: tiny",
            "  - config_name: compact-25k\n"
            "    data_files:\n"
            "      - split: train\n"
            "        path: indexes/tiny.parquet\n"
            "  - config_name: tiny",
        ),
        encoding="utf-8",
    )

    errors = validate_hf_release(release_dir)
    assert any("unexpected configs" in error for error in errors)


def test_validation_rejects_wrong_release_split_counts(tmp_path: Path) -> None:
    release_dir = tmp_path / "wrong_split_counts"
    _write_minimal_valid_release(release_dir)

    family_index = json.loads(
        (release_dir / "family_index.json").read_text(encoding="utf-8")
    )
    family_index["nested"] = {"tiny": "medium", "medium": "large"}
    family_index["split_counts"] = {
        config: {
            "train": RELEASE_CONFIG_TARGETS[config],
            "validation": 0,
            "test": 0,
        }
        for config in ("tiny", "medium", "large")
    }
    (release_dir / "family_index.json").write_text(
        json.dumps(family_index), encoding="utf-8"
    )
    (release_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "description": "info desc",
                "configs": {
                    "tiny": {
                        "num_examples": RELEASE_CONFIG_TARGETS["tiny"],
                        "splits": family_index["split_counts"]["tiny"],
                    },
                    "medium": {
                        "num_examples": RELEASE_CONFIG_TARGETS["medium"],
                        "splits": family_index["split_counts"]["medium"],
                    },
                    "large": {
                        "num_examples": RELEASE_CONFIG_TARGETS["large"],
                        "splits": family_index["split_counts"]["large"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    errors = validate_hf_release(release_dir)
    assert any("split_counts for tiny" in error for error in errors)
    assert any("splits for large" in error for error in errors)


def test_publish_hf_release_validates_and_uses_hf_cli(
    tmp_path: Path, monkeypatch
) -> None:
    release_dir = tmp_path / "publish_release"
    _write_minimal_valid_release(release_dir)
    calls = []

    def fake_run_hf_cli(args):
        calls.append(args)

    monkeypatch.setattr("turkicdocgen.hf.release._run_hf_cli", fake_run_hf_cli)

    plan = publish_hf_release(
        release_dir,
        "alenisaw/turkic-docgen",
        private=True,
        num_workers=2,
    )

    assert calls == [
        ["auth", "whoami"],
        [
            "repos",
            "create",
            "alenisaw/turkic-docgen",
            "--type",
            "dataset",
            "--exist-ok",
            "--private",
        ],
        [
            "upload-large-folder",
            "alenisaw/turkic-docgen",
            str(release_dir),
            "--type",
            "dataset",
            "--num-workers",
            "2",
        ],
    ]
    assert plan == calls
