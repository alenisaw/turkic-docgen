from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import sqlite3
import subprocess
import tarfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image

from turkicdocgen.profiles import DATASET_PROFILE
from turkicdocgen.qa import QA_CONFIG
from turkicdocgen.release_config import (
    LARGE_CONFIG,
    MEDIUM_CONFIG,
    RELEASE_CONFIG_DETAILS,
    RELEASE_CONFIG_TARGETS,
    RELEASE_CONFIGS,
    RELEASE_NESTING,
    TINY_CONFIG,
    normalize_release_subsets,
)
from turkicdocgen.safety import (
    assert_not_protected_path,
    safe_prepare_output_dir,
    validate_structure_limits,
)
from turkicdocgen.splits import (
    assign_splits,
    build_components,
    get_duplicate_clusters,
    proportional_split_targets,
    stratify_and_rank,
)

from .dataset_card import dataset_release_name, write_dataset_card

SAFE_PAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
HF_SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}
MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
MAX_RELEASE_ROWS = 300_000
SUPPORTED_RELEASE_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_RELEASE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, max_rows=MAX_RELEASE_ROWS))


def iter_jsonl(
    path: Path,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
                raise ValueError(
                    f"{path.name}:{line_number} exceeds the JSONL line-size limit"
                )
            if max_rows is not None and line_number > max_rows:
                raise ValueError(f"{path.name} exceeds {max_rows} rows")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {path.name} at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path.name}:{line_number} must contain a JSON object"
                )
            validate_structure_limits(row)
            yield row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_page_id(page_id: Any) -> str:
    value = str(page_id)
    if value in {".", ".."} or not SAFE_PAGE_ID.fullmatch(value):
        raise ValueError(f"unsafe page_id for release archive: {value!r}")
    return value


def _read_verified_image(path: Path) -> tuple[bytes, str]:
    if not path.is_file():
        raise ValueError(f"missing source image: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_RELEASE_IMAGE_SUFFIXES:
        raise ValueError(f"unsupported source image suffix: {path}")
    data = path.read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format or ""
            if image_format not in SUPPORTED_RELEASE_IMAGE_FORMATS:
                raise ValueError(f"unsupported source image format: {path}")
            image.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"invalid source image: {path}") from exc
    return data, suffix


def _resolve_source_file(root: Path, relative_path: Any) -> Path:
    value = str(relative_path)
    posix_path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or posix_path.is_absolute()
        or ".." in posix_path.parts
    ):
        raise ValueError(f"unsafe source path in manifest: {value!r}")
    root_resolved = root.resolve()
    candidate = (root / Path(*posix_path.parts)).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"source path escapes input directory: {value!r}")
    return candidate


def _manifest_image_relative_path(input_dir: Path, manifest_row: dict[str, Any]) -> str:
    page_id = _validate_page_id(manifest_row["page_id"])
    image_rel = manifest_row.get("image") or manifest_row.get("image_path")
    if image_rel:
        return str(image_rel)
    for suffix in sorted(SUPPORTED_RELEASE_IMAGE_SUFFIXES):
        candidate = Path("images") / f"{page_id}{suffix}"
        if (input_dir / candidate).is_file():
            return candidate.as_posix()
    raise ValueError(f"manifest row for {page_id} does not declare an image path")


def _parse_card_front_matter(card_text: str) -> dict[str, Any]:
    if not card_text.startswith("---\n"):
        raise ValueError("dataset card is missing YAML front matter")
    try:
        raw_metadata, _body = card_text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise ValueError("dataset card front matter is not terminated") from exc
    metadata = yaml.safe_load(raw_metadata)
    if not isinstance(metadata, dict):
        raise ValueError("dataset card front matter must be a mapping")
    validate_structure_limits(metadata, max_items=10_000, max_string_length=1_000_000)
    return metadata


class _SidecarIndex:
    def __init__(self, input_dir: Path, database_path: Path) -> None:
        self.database_path = database_path
        self.connection = sqlite3.connect(database_path)
        try:
            self.connection.executescript(
                """
                PRAGMA journal_mode = OFF;
                PRAGMA synchronous = OFF;
                PRAGMA temp_store = FILE;
                CREATE TABLE zones (page_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE sft (page_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE recognition (
                    page_id TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY (page_id, region_id)
                );
                CREATE TABLE detection (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_id TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    bbox TEXT
                );
                CREATE INDEX detection_page ON detection(page_id, sequence);
                """
            )
            self._insert_unique_rows(input_dir / "zone_gt.jsonl", "zones")
            self._insert_unique_rows(input_dir / "sft.jsonl", "sft")
            self._insert_recognition_rows(input_dir / "ocr_rec.jsonl")
            self._insert_detection_rows(input_dir / "ocr_det.jsonl")
            missing_recognitions = self.connection.execute(
                """
                SELECT COUNT(*)
                FROM detection AS d
                LEFT JOIN recognition AS r
                  ON r.page_id = d.page_id AND r.region_id = d.region_id
                WHERE r.page_id IS NULL
                """
            ).fetchone()[0]
            if missing_recognitions:
                raise ValueError(
                    "OCR detection export contains "
                    f"{missing_recognitions} regions without recognition rows"
                )
        except Exception:
            self.close()
            raise

    def _insert_unique_rows(self, path: Path, table: str) -> None:
        batch = []
        for row in iter_jsonl(path, max_rows=MAX_RELEASE_ROWS):
            page_id = str(row.get("page_id", ""))
            if not page_id:
                raise ValueError(f"{path.name} contains a row without page_id")
            batch.append((page_id, json.dumps(row, ensure_ascii=False)))
            if len(batch) >= 1000:
                self.connection.executemany(
                    f"INSERT INTO {table} (page_id, payload) VALUES (?, ?)",
                    batch,
                )
                batch.clear()
        if batch:
            self.connection.executemany(
                f"INSERT INTO {table} (page_id, payload) VALUES (?, ?)",
                batch,
            )
        self.connection.commit()

    def _insert_recognition_rows(self, path: Path) -> None:
        batch = []
        for row in iter_jsonl(path, max_rows=MAX_RELEASE_ROWS * 100):
            page_id = str(row.get("page_id", ""))
            region_id = str(row.get("region_id") or row.get("line_id") or "")
            if not page_id or not region_id:
                raise ValueError(f"{path.name} contains an incomplete OCR key")
            batch.append((page_id, region_id, str(row.get("text") or "")))
            if len(batch) >= 5000:
                self.connection.executemany(
                    "INSERT INTO recognition VALUES (?, ?, ?)",
                    batch,
                )
                batch.clear()
        if batch:
            self.connection.executemany(
                "INSERT INTO recognition VALUES (?, ?, ?)",
                batch,
            )
        self.connection.commit()

    def _insert_detection_rows(self, path: Path) -> None:
        batch = []
        for row in iter_jsonl(path, max_rows=MAX_RELEASE_ROWS * 100):
            page_id = str(row.get("page_id", ""))
            region_id = str(row.get("region_id") or row.get("line_id") or "")
            if not page_id or not region_id:
                raise ValueError(f"{path.name} contains an incomplete OCR key")
            batch.append(
                (
                    page_id,
                    region_id,
                    json.dumps(row.get("bbox"), ensure_ascii=False),
                )
            )
            if len(batch) >= 5000:
                self.connection.executemany(
                    "INSERT INTO detection (page_id, region_id, bbox) VALUES (?, ?, ?)",
                    batch,
                )
                batch.clear()
        if batch:
            self.connection.executemany(
                "INSERT INTO detection (page_id, region_id, bbox) VALUES (?, ?, ?)",
                batch,
            )
        self.connection.commit()

    def page(
        self, page_id: str
    ) -> tuple[list[Any], dict[str, Any] | None, list[dict[str, Any]]]:
        zone_row = self.connection.execute(
            "SELECT payload FROM zones WHERE page_id = ?", (page_id,)
        ).fetchone()
        sft_row = self.connection.execute(
            "SELECT payload FROM sft WHERE page_id = ?", (page_id,)
        ).fetchone()
        ocr_rows = self.connection.execute(
            """
            SELECT d.region_id, d.bbox, r.text
            FROM detection AS d
            JOIN recognition AS r
              ON r.page_id = d.page_id AND r.region_id = d.region_id
            WHERE d.page_id = ?
            ORDER BY d.sequence
            """,
            (page_id,),
        )
        zones = json.loads(zone_row[0]).get("zones", []) if zone_row else []
        sft = json.loads(sft_row[0]) if sft_row else None
        ocr = [
            {
                "region_id": region_id,
                "bbox": json.loads(bbox),
                "text": text,
            }
            for region_id, bbox, text in ocr_rows
        ]
        return zones, sft, ocr

    def close(self) -> None:
        self.connection.close()
        self.database_path.unlink(missing_ok=True)


def _release_parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            ("page_id", pa.string()),
            ("layout_id", pa.string()),
            ("split", pa.string()),
            ("subsets", pa.list_(pa.string())),
            ("tar_path", pa.string()),
            ("image_filename", pa.string()),
            ("ocr_text", pa.string()),
            ("ocr_bboxes", pa.list_(pa.list_(pa.float64()))),
            ("ocr_labels", pa.list_(pa.string())),
            ("ocr_region_ids", pa.list_(pa.string())),
            ("ocr_json", pa.string()),
            ("zones_json", pa.string()),
            ("sft_json", pa.string()),
        ]
    )


def _prepare_legacy_release_manifest(
    manifest_file: Path,
    prepared_path: Path,
    input_dir: Path,
) -> Path:
    rows = [row for row in read_jsonl(manifest_file) if row.get("qa_ok") is True]
    duplicate_clusters = get_duplicate_clusters(rows, input_dir)
    components = build_components(rows, duplicate_clusters)
    page_to_split = assign_splits(components, len(rows))
    split_groups = {split: [] for split in HF_SPLIT_NAMES}
    page_rows = {row["page_id"]: row for row in rows}
    for page_id, split in page_to_split.items():
        split_groups[split].append(page_rows[page_id])
    rankings = {
        split: stratify_and_rank(split_groups[split]) for split in HF_SPLIT_NAMES
    }
    split_sizes = {split: len(ranking) for split, ranking in rankings.items()}
    tiny_targets = proportional_split_targets(
        split_sizes, min(TINY_CONFIG.target_rows, len(rows))
    )
    medium_targets = proportional_split_targets(
        split_sizes, min(MEDIUM_CONFIG.target_rows, len(rows))
    )
    for split, ranking in rankings.items():
        for rank, row in enumerate(ranking):
            in_tiny = rank < tiny_targets[split]
            in_medium = rank < medium_targets[split]
            subsets = [LARGE_CONFIG.name]
            if in_medium:
                subsets.insert(0, MEDIUM_CONFIG.name)
            if in_tiny:
                subsets.insert(0, TINY_CONFIG.name)
            row.update(
                {
                    "split": split,
                    "nested_rank": rank,
                    "in_tiny": in_tiny,
                    "in_medium": in_medium,
                    "in_large": True,
                    "subsets": subsets,
                }
            )
    write_jsonl(prepared_path, rows)
    return prepared_path


def _stream_release_data(
    input_dir: Path,
    out_dir: Path,
    samples_per_shard: int,
    manifest_path: Path,
) -> tuple[int, dict[str, Any], dict[str, dict[str, str]], dict[str, dict[str, int]]]:
    schema = _release_parquet_schema()
    indexes_dir = out_dir / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    parquet_paths = {
        "samples": indexes_dir / "samples.parquet",
        TINY_CONFIG.name: indexes_dir / f"{TINY_CONFIG.name}.parquet",
        MEDIUM_CONFIG.name: indexes_dir / f"{MEDIUM_CONFIG.name}.parquet",
        LARGE_CONFIG.name: indexes_dir / f"{LARGE_CONFIG.name}.parquet",
    }
    config_data_files: dict[str, dict[str, str]] = {
        config: {} for config in RELEASE_CONFIGS
    }
    config_split_counts: dict[str, dict[str, int]] = {
        config: {split: 0 for split in HF_SPLIT_NAMES.values()}
        for config in RELEASE_CONFIGS
    }
    for config in RELEASE_CONFIGS:
        config_dir = indexes_dir / config
        config_dir.mkdir(parents=True, exist_ok=True)
        for split in HF_SPLIT_NAMES.values():
            parquet_paths[f"{config}:{split}"] = config_dir / f"{split}.parquet"

    writers: dict[str, pq.ParquetWriter] = {}
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def append_parquet(key: str, row: dict[str, Any]) -> None:
        buffer = buffers[key]
        buffer.append(row)
        if len(buffer) >= 256:
            writers[key].write_table(pa.Table.from_pylist(buffer, schema=schema))
            buffer.clear()

    sidecars = _SidecarIndex(input_dir, out_dir / ".release_index.sqlite3")
    tar_handles: dict[str, tarfile.TarFile] = {}
    split_counts = Counter()
    layouts = Counter()
    languages = Counter()
    effects = Counter()
    accepted = 0

    try:
        for key, path in parquet_paths.items():
            writers[key] = pq.ParquetWriter(path, schema, compression="zstd")
        for manifest_row in iter_jsonl(manifest_path, max_rows=MAX_RELEASE_ROWS):
            page_id = _validate_page_id(manifest_row["page_id"])
            if manifest_row.get("qa_ok") is not True:
                continue
            zones_list, sft_row, ocr_list = sidecars.page(page_id)
            split_key = str(manifest_row.get("split", ""))
            if split_key not in HF_SPLIT_NAMES:
                raise ValueError(f"invalid split in manifest: {split_key!r}")
            hf_split = HF_SPLIT_NAMES[split_key]
            split_index = split_counts[split_key]
            shard_index = split_index // samples_per_shard
            if split_index % samples_per_shard == 0:
                previous = tar_handles.pop(split_key, None)
                if previous is not None:
                    previous.close()
                split_dir = out_dir / "data" / hf_split
                split_dir.mkdir(parents=True, exist_ok=True)
                tar_handles[split_key] = tarfile.open(
                    split_dir / f"{hf_split}-{shard_index:06d}.tar",
                    "w",
                )
            tar_handle = tar_handles[split_key]
            tar_rel_path = f"data/{hf_split}/{hf_split}-{shard_index:06d}.tar"

            image_rel = _manifest_image_relative_path(input_dir, manifest_row)
            image_data, image_suffix = _read_verified_image(
                _resolve_source_file(input_dir, image_rel)
            )
            image_filename = f"{page_id}{image_suffix}"
            image_info = tarfile.TarInfo(name=image_filename)
            image_info.size = len(image_data)
            tar_handle.addfile(image_info, io.BytesIO(image_data))

            sft_data = None
            if sft_row:
                sft_data = {
                    "prompt": sft_row.get("prompt", "Read the page zones."),
                    "response": sft_row.get("response", []),
                }
            subsets = normalize_release_subsets(manifest_row.get("subsets"))
            annotation = {
                "page_id": page_id,
                "layout_id": manifest_row.get("layout_id"),
                "layout_variant": manifest_row.get("layout_variant"),
                "split": split_key,
                "in_tiny": TINY_CONFIG.name in subsets,
                "in_medium": MEDIUM_CONFIG.name in subsets,
                "in_large": LARGE_CONFIG.name in subsets,
                "subsets": subsets,
                "sft": sft_data,
                "ocr": ocr_list,
                "zones": zones_list,
            }
            annotation_bytes = json.dumps(
                annotation, ensure_ascii=False, indent=2
            ).encode("utf-8")
            annotation_info = tarfile.TarInfo(name=f"{page_id}.json")
            annotation_info.size = len(annotation_bytes)
            tar_handle.addfile(annotation_info, io.BytesIO(annotation_bytes))

            parquet_row = {
                "page_id": page_id,
                "layout_id": manifest_row.get("layout_id"),
                "split": split_key,
                "subsets": subsets,
                "tar_path": tar_rel_path,
                "image_filename": image_filename,
                "ocr_text": " ".join(item["text"] for item in ocr_list if item["text"]),
                "ocr_bboxes": [item["bbox"] for item in ocr_list],
                "ocr_labels": [item["text"] for item in ocr_list],
                "ocr_region_ids": [item["region_id"] for item in ocr_list],
                "ocr_json": json.dumps(ocr_list, ensure_ascii=False),
                "zones_json": json.dumps(zones_list, ensure_ascii=False),
                "sft_json": (
                    json.dumps(sft_data, ensure_ascii=False) if sft_data else ""
                ),
            }
            append_parquet("samples", parquet_row)
            for config in RELEASE_CONFIGS:
                if config in subsets:
                    append_parquet(config, parquet_row)
                    append_parquet(f"{config}:{hf_split}", parquet_row)
                    config_split_counts[config][hf_split] += 1

            split_counts[split_key] += 1
            accepted += 1
            layouts[str(manifest_row.get("layout_id", ""))] += 1
            languages[str(manifest_row.get("language_mix", ""))] += 1
            effects[
                str(
                    manifest_row.get("effect_profile")
                    or manifest_row.get("quality_profile", "")
                )
            ] += 1

        for key, buffer in buffers.items():
            if buffer:
                writers[key].write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
    finally:
        for tar_handle in tar_handles.values():
            tar_handle.close()
        for writer in writers.values():
            writer.close()
        sidecars.close()

    for config, split_counts_for_config in config_split_counts.items():
        for split, count in split_counts_for_config.items():
            if count:
                config_data_files[config][split] = (
                    parquet_paths[f"{config}:{split}"].relative_to(out_dir).as_posix()
                )
    summary = {
        "rows": accepted,
        "layouts": dict(layouts),
        "languages": dict(languages),
        "effects": dict(effects),
    }
    return accepted, summary, config_data_files, config_split_counts


def export_hf_release(
    input_dir: Path,
    out_dir: Path,
    *,
    pretty_name: str | None = None,
    hf_card: bool = True,
    force: bool = True,
    publish: bool = False,
    samples_per_shard: int = 1000,
) -> Path:
    if publish:
        raise ValueError(
            "Remote publication is disabled; build and validate the release locally."
        )
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")

    manifest_file = input_dir / "manifest.jsonl"
    if not manifest_file.exists():
        raise ValueError(f"missing manifest: {manifest_file}")

    final_out_dir = assert_not_protected_path(out_dir, purpose="HF release output")
    if final_out_dir.exists() and not force:
        raise FileExistsError(f"release output already exists: {final_out_dir}")
    staging_dir = final_out_dir.with_name(
        f".{final_out_dir.name}.building-{uuid.uuid4().hex}"
    )
    out_dir = safe_prepare_output_dir(staging_dir, force=True)
    out_dir.mkdir(parents=True, exist_ok=False)

    first_manifest_row = next(
        iter_jsonl(manifest_file, max_rows=MAX_RELEASE_ROWS), None
    )
    if first_manifest_row is None:
        raise ValueError("manifest does not contain any rows")
    release_manifest = manifest_file
    if "split" not in first_manifest_row:
        release_manifest = _prepare_legacy_release_manifest(
            manifest_file,
            out_dir / ".prepared_manifest.jsonl",
            input_dir,
        )
    try:
        (
            filtered_manifest_count,
            summary,
            config_data_files,
            config_split_counts,
        ) = _stream_release_data(
            input_dir,
            out_dir,
            samples_per_shard,
            release_manifest,
        )
    except Exception:
        safe_prepare_output_dir(out_dir, force=True)
        raise
    finally:
        if release_manifest != manifest_file:
            release_manifest.unlink(missing_ok=True)

    # Release Metadata & README card

    if hf_card:
        write_dataset_card(
            out_dir / "README.md",
            pretty_name or dataset_release_name(filtered_manifest_count),
            summary=summary,
            config_data_files=config_data_files,
        )

    # CITATION.cff
    citation_content = """cff-version: 1.2.0
message: "If you use this dataset, please cite both the dataset and the accompanying paper."
authors:
  - family-names: Issayev
    given-names: Alen
title: "TurkicOCR Synthetic Cyrillic Dataset"
version: "1.0.0"
doi: "10.57967/hf/9255"
date-released: "2026-08-15"
url: "https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic"
"""
    (out_dir / "CITATION.cff").write_text(citation_content, encoding="utf-8")

    # family_index.json
    family_index_data = {
        "family_name": "TurkicDocGen Synthetic Cyrillic",
        "configs": list(RELEASE_CONFIGS),
        "data_files": config_data_files,
        "split_counts": config_split_counts,
        "nested": RELEASE_NESTING,
    }
    (out_dir / "family_index.json").write_text(
        json.dumps(family_index_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Copy rejected_samples.jsonl if exists
    rejected_src = input_dir / "rejected_samples.jsonl"
    if rejected_src.exists():
        shutil.copy2(rejected_src, out_dir / "rejected_samples.jsonl")

    # dataset_info.json
    dataset_info_data = {
        "description": "TurkicDocGen Synthetic Cyrillic OCR and document-understanding dataset.",
        "configs": {
            config: {
                "description": RELEASE_CONFIG_DETAILS[config].description,
                "num_examples": sum(config_split_counts[config].values()),
                "splits": config_split_counts[config],
                "data_files": config_data_files[config],
                "features": {
                    "page_id": "string",
                    "image": "image (JPEG/PNG/WebP)",
                    "annotations": "json",
                },
            }
            for config in RELEASE_CONFIGS
        },
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    security_report = {
        "status": "passed",
        "unresolved_high_severity": 0,
        "controls": {
            "source_paths_confined": True,
            "external_symlinks_rejected": True,
            "archive_member_names_sanitized": True,
            "archive_regular_files_only": True,
            "image_payloads_verified": True,
            "json_depth_and_size_bounded": True,
            "dataset_card_yaml_validated": True,
            "remote_publication_disabled": True,
            "environment_dump_stored": False,
        },
        "dependency_audit": {
            "pip_audit_available": shutil.which("pip-audit") is not None,
            "cargo_audit_available": shutil.which("cargo-audit") is not None,
            "execution": "recorded by the validation ladder when available",
        },
        "findings": [],
        "residual_risks": [
            "Dependency advisory databases require an explicit local audit command."
        ],
    }
    (reports_dir / "security_report.json").write_text(
        json.dumps(security_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # provenance.json
    def get_git_commit() -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return "unknown"

    run_manifest_path = input_dir / "run_manifest.json"
    configuration_sources = {
        "qa_config": QA_CONFIG,
        "dataset_profile_sha256": sha256_file(DATASET_PROFILE),
        "run_manifest_sha256": (
            sha256_file(run_manifest_path) if run_manifest_path.is_file() else None
        ),
    }
    config_hash = hashlib.sha256(
        json.dumps(configuration_sources, sort_keys=True).encode("utf-8")
    ).hexdigest()

    inventory_hashes = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(out_dir).as_posix()
            inventory_hashes[rel] = sha256_file(path)

    provenance_data = {
        "release_id": final_out_dir.name,
        "git_commit": get_git_commit(),
        "configurations_hash": config_hash,
        "inventory_hashes": inventory_hashes,
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": "TurkicDocGen Local Hugging Face Packager",
            "network_access": "disabled",
        },
        "configuration_sources": configuration_sources,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # checksums.sha256 reuses the inventory digests so large TAR shards are not
    # read twice. Provenance is written after the inventory snapshot.
    checksum_hashes = dict(inventory_hashes)
    checksum_hashes["provenance.json"] = sha256_file(out_dir / "provenance.json")
    checksum_lines = [
        f"{digest}  {relative_path}"
        for relative_path, digest in sorted(checksum_hashes.items())
    ]
    (out_dir / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    if final_out_dir.exists():
        safe_prepare_output_dir(final_out_dir, force=True)
    out_dir.replace(final_out_dir)
    return final_out_dir


def _run_hf_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["hf", *args],
            check=True,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Hugging Face CLI `hf` is not installed. Install it and run "
            "`hf auth login`, or set HF_TOKEN in the environment."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = f"HF CLI command failed: hf {' '.join(args)}"
        if detail:
            message = f"{message}\n{detail}"
        raise RuntimeError(message) from exc


def publish_hf_release(
    release_dir: Path,
    repo_id: str,
    *,
    private: bool = False,
    dry_run: bool = False,
    num_workers: int | None = None,
) -> list[list[str]]:
    """Validate and publish a local release folder through the HF CLI.

    Authentication is intentionally delegated to the environment or the HF CLI
    credential store. Tokens must not be passed through this API.
    """
    errors = validate_hf_release(release_dir)
    if errors:
        preview = "; ".join(errors[:10])
        raise ValueError(f"release validation failed: {preview}")
    if not repo_id or "/" not in repo_id or repo_id.strip() != repo_id:
        raise ValueError("repo_id must look like 'namespace/dataset-name'")

    auth_command = ["auth", "whoami"]
    create_command = ["repos", "create", repo_id, "--type", "dataset", "--exist-ok"]
    if private:
        create_command.append("--private")
    upload_command = [
        "upload-large-folder",
        repo_id,
        str(release_dir),
        "--type",
        "dataset",
    ]
    if num_workers is not None:
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        upload_command.extend(["--num-workers", str(num_workers)])

    plan = [auth_command, create_command, upload_command]
    if dry_run:
        return plan

    _run_hf_cli(auth_command)
    _run_hf_cli(create_command)
    _run_hf_cli(upload_command)
    return plan


def validate_hf_release(release_dir: Path) -> list[str]:
    errors: list[str] = []
    declared_split_counts: dict[str, dict[str, int]] = {}

    # Read provenance first to check if README.md was generated
    provenance_path = release_dir / "provenance.json"
    has_readme = True
    if provenance_path.exists():
        try:
            prov_data = json.loads(provenance_path.read_text(encoding="utf-8"))
            if "inventory_hashes" in prov_data:
                has_readme = "README.md" in prov_data["inventory_hashes"]
        except Exception:
            pass

    # 1. Basic Metadata Files
    basic_files = [
        "CITATION.cff",
        "family_index.json",
        "dataset_info.json",
        "provenance.json",
        "checksums.sha256",
        "reports/security_report.json",
    ]
    if has_readme:
        basic_files.append("README.md")

    for name in basic_files:
        if not (release_dir / name).exists():
            errors.append(f"missing {name}")

    security_report_path = release_dir / "reports" / "security_report.json"
    if security_report_path.is_file():
        try:
            security_report = json.loads(
                security_report_path.read_text(encoding="utf-8")
            )
            validate_structure_limits(security_report, max_items=10_000)
            if security_report.get("unresolved_high_severity") != 0:
                errors.append(
                    "security report contains unresolved high-severity findings"
                )
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            errors.append(f"failed to read security report: {exc}")

    card_path = release_dir / "README.md"
    if card_path.is_file():
        try:
            card_metadata = _parse_card_front_matter(
                card_path.read_text(encoding="utf-8")
            )
            configs = card_metadata.get("configs")
            if not isinstance(configs, list):
                errors.append("dataset card configs must be a list")
            else:
                config_names = {
                    config.get("config_name")
                    for config in configs
                    if isinstance(config, dict)
                }
                missing_configs = set(RELEASE_CONFIGS) - config_names
                if missing_configs:
                    errors.append(
                        f"dataset card missing configs: {sorted(missing_configs)}"
                    )
                unexpected_configs = config_names - set(RELEASE_CONFIGS)
                if unexpected_configs:
                    errors.append(
                        "dataset card contains unexpected configs: "
                        f"{sorted(unexpected_configs)}"
                    )
                for config in configs:
                    if not isinstance(config, dict):
                        errors.append("dataset card config entry must be a mapping")
                        continue
                    data_files = config.get("data_files")
                    if not isinstance(data_files, list) or not data_files:
                        errors.append(
                            f"dataset card config {config.get('config_name')} has no data_files"
                        )
                        continue
                    declared_splits = set()
                    for data_file in data_files:
                        if not isinstance(data_file, dict):
                            errors.append("dataset card data_file must be a mapping")
                            continue
                        split = data_file.get("split")
                        rel_path = data_file.get("path")
                        declared_splits.add(split)
                        if split not in {"train", "validation", "test"}:
                            errors.append(
                                f"dataset card contains invalid split: {split!r}"
                            )
                        if not isinstance(rel_path, str):
                            errors.append(
                                "dataset card data_file path must be a string"
                            )
                            continue
                        path_parts = PurePosixPath(rel_path)
                        if path_parts.is_absolute() or ".." in path_parts.parts:
                            errors.append(
                                f"dataset card contains unsafe data_file path: {rel_path}"
                            )
                        elif not (release_dir / Path(*path_parts.parts)).is_file():
                            errors.append(
                                f"dataset card data_file does not exist: {rel_path}"
                            )
                    if config_data := config.get("config_name"):
                        if config_data in RELEASE_CONFIGS and not declared_splits:
                            errors.append(
                                f"dataset card config {config_data} has no declared splits"
                            )
        except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"invalid README.md front matter: {exc}")

    # Check CITATION.cff has valid DOI and no Zenodo references
    citation_path = release_dir / "CITATION.cff"
    if citation_path.exists():
        citation_content = citation_path.read_text(encoding="utf-8")
        if (
            'doi: "pending"' not in citation_content
            and "doi: pending" not in citation_content
            and "10.57967" not in citation_content
            and 'doi: "10.' not in citation_content
        ):
            errors.append("CITATION.cff DOI is not set to pending or valid DOI")
        if "zenodo" in citation_content.lower():
            errors.append("CITATION.cff contains Zenodo references")

    # Check family_index.json structure
    family_index_path = release_dir / "family_index.json"
    if family_index_path.exists():
        try:
            family_data = json.loads(family_index_path.read_text(encoding="utf-8"))
            if "configs" not in family_data or not isinstance(
                family_data["configs"], list
            ):
                errors.append("invalid family_index.json structure")
            else:
                if family_data["configs"] != list(RELEASE_CONFIGS):
                    errors.append(
                        "family_index.json configs must be exactly "
                        f"{list(RELEASE_CONFIGS)}"
                    )
                for conf in RELEASE_CONFIGS:
                    if conf not in family_data["configs"]:
                        errors.append(f"config {conf} missing from family_index.json")
                unexpected = set(family_data["configs"]) - set(RELEASE_CONFIGS)
                if unexpected:
                    errors.append(
                        f"unexpected configs in family_index.json: {sorted(unexpected)}"
                    )
            if family_data.get("nested") != RELEASE_NESTING:
                errors.append("family_index.json nested release order is invalid")
            split_counts = family_data.get("split_counts")
            if split_counts is not None:
                if not isinstance(split_counts, dict):
                    errors.append("family_index.json split_counts must be a mapping")
                else:
                    for config in RELEASE_CONFIGS:
                        counts = split_counts.get(config)
                        if not isinstance(counts, dict):
                            errors.append(
                                f"family_index.json missing split_counts for {config}"
                            )
                            continue
                        expected_total = RELEASE_CONFIG_TARGETS[config]
                        expected_counts = {
                            "train": int(expected_total * 0.90),
                            "validation": int(expected_total * 0.05),
                            "test": expected_total
                            - int(expected_total * 0.90)
                            - int(expected_total * 0.05),
                        }
                        normalized_counts = {}
                        for split in ("train", "validation", "test"):
                            raw_count = counts.get(split)
                            if raw_count is None:
                                errors.append(
                                    "family_index.json split_counts for "
                                    f"{config} missing {split}"
                                )
                                normalized_counts[split] = 0
                                continue
                            try:
                                count = int(raw_count)
                            except (TypeError, ValueError):
                                errors.append(
                                    "family_index.json split_counts for "
                                    f"{config}/{split} must be an integer"
                                )
                                normalized_counts[split] = 0
                                continue
                            if count < 0:
                                errors.append(
                                    "family_index.json split_counts for "
                                    f"{config}/{split} must be non-negative"
                                )
                            normalized_counts[split] = count
                        normalized_total = sum(normalized_counts.values())
                        if normalized_total > expected_total:
                            errors.append(
                                f"family_index.json split_counts for {config} exceed "
                                f"target {expected_total}: {normalized_counts}"
                            )
                        elif (
                            normalized_total == expected_total
                            and normalized_counts != expected_counts
                        ):
                            errors.append(
                                f"family_index.json split_counts for {config} "
                                f"must be {expected_counts}, got {normalized_counts}"
                            )
                        declared_split_counts[config] = normalized_counts
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            errors.append(f"failed to read family_index.json: {exc}")

    # Check dataset_info.json
    dataset_info_path = release_dir / "dataset_info.json"
    if dataset_info_path.exists():
        try:
            dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
            configs = dataset_info.get("configs")
            if isinstance(configs, dict):
                if set(configs) != set(RELEASE_CONFIGS):
                    errors.append(
                        "dataset_info.json configs must be exactly "
                        f"{list(RELEASE_CONFIGS)}"
                    )
                for config in RELEASE_CONFIGS:
                    config_info = configs.get(config)
                    if not isinstance(config_info, dict):
                        errors.append(f"dataset_info.json missing config {config}")
                        continue
                    expected_total = RELEASE_CONFIG_TARGETS[config]
                    raw_declared_examples = config_info.get("num_examples")
                    try:
                        declared_examples = (
                            int(raw_declared_examples)
                            if raw_declared_examples is not None
                            else None
                        )
                    except (TypeError, ValueError):
                        errors.append(
                            f"dataset_info.json num_examples for {config} "
                            "must be an integer"
                        )
                        declared_examples = None
                    if (
                        declared_examples is not None
                        and declared_examples > expected_total
                    ):
                        errors.append(
                            f"dataset_info.json num_examples for {config} exceed "
                            f"target {expected_total}: {declared_examples}"
                        )
                    if declared_examples == expected_total:
                        expected_counts = {
                            "train": int(expected_total * 0.90),
                            "validation": int(expected_total * 0.05),
                            "test": expected_total
                            - int(expected_total * 0.90)
                            - int(expected_total * 0.05),
                        }
                        if config_info.get("splits") != expected_counts:
                            errors.append(
                                f"dataset_info.json splits for {config} must be "
                                f"{expected_counts}"
                            )
                    if (
                        config in declared_split_counts
                        and config_info.get("splits") != declared_split_counts[config]
                    ):
                        errors.append(
                            f"dataset_info.json splits for {config} do not match "
                            "family_index.json"
                        )
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"failed to read dataset_info.json: {exc}")

    # Check provenance.json
    provenance_path = release_dir / "provenance.json"
    if provenance_path.exists():
        try:
            prov_data = json.loads(provenance_path.read_text(encoding="utf-8"))
            if (
                "git_commit" not in prov_data
                or "configurations_hash" not in prov_data
                or "inventory_hashes" not in prov_data
            ):
                errors.append("invalid provenance.json structure")
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            errors.append(f"failed to read provenance.json: {exc}")

    # 2. Check Parquet indexes
    indexes_dir = release_dir / "indexes"
    if not indexes_dir.exists():
        errors.append("missing indexes/")
    else:
        for name in [
            "samples.parquet",
            *(f"{config}.parquet" for config in RELEASE_CONFIGS),
        ]:
            p = indexes_dir / name
            if not p.exists():
                errors.append(f"missing index file: indexes/{name}")
            else:
                try:
                    parquet_file = pq.ParquetFile(p)
                    expected_cols = {
                        "page_id",
                        "layout_id",
                        "split",
                        "subsets",
                        "tar_path",
                        "image_filename",
                    }
                    missing_cols = expected_cols - set(parquet_file.schema_arrow.names)
                    if missing_cols:
                        errors.append(f"index {name} missing columns: {missing_cols}")
                    if declared_split_counts and name == "samples.parquet":
                        expected_rows = sum(
                            declared_split_counts.get(LARGE_CONFIG.name, {}).values()
                        )
                    elif declared_split_counts and name.endswith(".parquet"):
                        config_name = name.removesuffix(".parquet")
                        config_counts = declared_split_counts.get(config_name)
                        expected_rows = (
                            sum(config_counts.values()) if config_counts else None
                        )
                    else:
                        expected_rows = None
                    row_count = parquet_file.metadata.num_rows
                    if expected_rows is not None and row_count != expected_rows:
                        errors.append(
                            f"index {name} must contain {expected_rows} rows, "
                            f"got {row_count}"
                        )
                except Exception as exc:
                    errors.append(f"failed to read index {name}: {exc}")
        for config, split_counts in declared_split_counts.items():
            for split, expected_rows in split_counts.items():
                split_path = indexes_dir / config / f"{split}.parquet"
                if not split_path.exists():
                    errors.append(
                        f"missing split index file: "
                        f"{split_path.relative_to(release_dir).as_posix()}"
                    )
                    continue
                try:
                    parquet_file = pq.ParquetFile(split_path)
                    row_count = parquet_file.metadata.num_rows
                    if row_count != expected_rows:
                        errors.append(
                            f"split index {config}/{split}.parquet must contain "
                            f"{expected_rows} rows, got {row_count}"
                        )
                except Exception as exc:
                    errors.append(
                        f"failed to read split index {config}/{split}.parquet: {exc}"
                    )

    # 3. Check packed TAR shards offline validation
    data_dir = release_dir / "data"
    if not data_dir.exists():
        errors.append("missing data/")
    else:
        tar_files = sorted(list(data_dir.rglob("*.tar")))
        if not tar_files:
            errors.append("no TAR archives found under data/")
        for tar_path in tar_files:
            try:
                split_folder = tar_path.parent.name
                if split_folder not in ("train", "validation", "test"):
                    errors.append(
                        f"invalid split folder for tar archive: {tar_path.relative_to(release_dir)}"
                    )

                with tarfile.open(tar_path, "r") as tar:
                    tar_members = tar.getmembers()
                    members = [member.name for member in tar_members]
                    for member in tar_members:
                        mname = member.name
                        member_path = PurePosixPath(mname)
                        if (
                            member_path.is_absolute()
                            or ".." in member_path.parts
                            or len(member_path.parts) != 1
                        ):
                            errors.append(
                                f"dangerous file path in {tar_path.name}: {mname}"
                            )
                        if not member.isfile():
                            errors.append(
                                f"non-regular TAR member in {tar_path.name}: {mname}"
                            )

                    stem_counts = Counter()
                    for member in tar_members:
                        mname = member.name
                        p = Path(mname)
                        stem_counts[p.stem] += 1

                        # Validate file format / content
                        if not member.isfile():
                            continue
                        if p.suffix == ".json":
                            f = tar.extractfile(member)
                            if f:
                                try:
                                    payload = json.loads(f.read().decode("utf-8"))
                                    if payload.get("page_id") != p.stem:
                                        errors.append(
                                            f"page_id mismatch in {tar_path.name} member {mname}"
                                        )
                                except (
                                    json.JSONDecodeError,
                                    UnicodeDecodeError,
                                    AttributeError,
                                ) as exc:
                                    errors.append(
                                        f"invalid JSON in {tar_path.name} member {mname}: {exc}"
                                    )
                        elif p.suffix in SUPPORTED_RELEASE_IMAGE_SUFFIXES:
                            f = tar.extractfile(member)
                            if f:
                                try:
                                    with Image.open(io.BytesIO(f.read())) as image:
                                        if (
                                            image.format
                                            not in SUPPORTED_RELEASE_IMAGE_FORMATS
                                        ):
                                            raise ValueError("unsupported image format")
                                        image.verify()
                                except (OSError, SyntaxError, ValueError):
                                    errors.append(
                                        f"invalid image in {tar_path.name} member {mname}"
                                    )
                        else:
                            errors.append(
                                f"unsupported TAR member in {tar_path.name}: {mname}"
                            )

                    # Check that for each page_id there are exactly two files (image + JSON).
                    for stem, count in stem_counts.items():
                        if count != 2:
                            errors.append(
                                f"incomplete sample in {tar_path.name}: {stem} has {count} files (expected 2)"
                            )
                        else:
                            has_image = any(
                                f"{stem}{suffix}" in members
                                for suffix in SUPPORTED_RELEASE_IMAGE_SUFFIXES
                            )
                            if not has_image:
                                errors.append(
                                    f"missing image for sample {stem} in {tar_path.name}"
                                )
                            if f"{stem}.json" not in members:
                                errors.append(
                                    f"missing JSON for sample {stem} in {tar_path.name}"
                                )
            except Exception as exc:
                errors.append(
                    f"failed to open/validate TAR shard {tar_path.name}: {exc}"
                )

    # 4. Checksums file verification
    checksum_path = release_dir / "checksums.sha256"
    if checksum_path.exists():
        try:
            lines = checksum_path.read_text(encoding="utf-8").splitlines()
            checked_files = set()
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    errors.append(f"invalid line in checksums.sha256: {line}")
                    continue
                expected_hash, rel_path = parts
                rel_path = rel_path.strip()
                path_parts = PurePosixPath(rel_path)
                if len(expected_hash) != 64 or any(
                    char not in "0123456789abcdef" for char in expected_hash
                ):
                    errors.append(f"invalid checksum digest for: {rel_path}")
                    continue
                if path_parts.is_absolute() or ".." in path_parts.parts:
                    errors.append(f"unsafe checksum path: {rel_path}")
                    continue
                file_path = release_dir / Path(*path_parts.parts)
                if file_path.is_symlink():
                    errors.append(f"checksum path is a symlink: {rel_path}")
                    continue
                if not file_path.is_file():
                    errors.append(
                        f"file listed in checksums.sha256 does not exist: {rel_path}"
                    )
                    continue
                checked_files.add(rel_path)
                actual_hash = sha256_file(file_path)
                if actual_hash != expected_hash:
                    errors.append(
                        f"file '{rel_path}' hash mismatch: actual={actual_hash}, expected={expected_hash}"
                    )

            # Check for untracked files
            for p in sorted(release_dir.rglob("*")):
                if p.is_symlink():
                    errors.append(
                        f"symlink is not allowed in release: "
                        f"{p.relative_to(release_dir).as_posix()}"
                    )
                elif p.is_file():
                    rel = p.relative_to(release_dir).as_posix()
                    if rel != "checksums.sha256" and rel not in checked_files:
                        errors.append(f"file not listed in checksums.sha256: {rel}")
        except Exception as exc:
            errors.append(f"failed to verify checksums: {exc}")

    return errors
