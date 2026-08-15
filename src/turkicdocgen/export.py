from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from turkicdocgen.languages import canonical_language_mix

from .qa import QA_CONFIG
from .schema import PagePlan, QAReport

_ = (Path, PagePlan, QAReport)


@dataclass(frozen=True)
class OCRRecordConfig:
    image_rel: str
    page_id: str
    region_id: str
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[int, int]]
    text: str | None
    min_width: int
    min_height: int


def _font_category(family: str, path: str | None) -> str:
    lowered = f"{family} {path or ''}".lower()
    if "mono" in lowered:
        return "mono_or_table_safe"
    if any(token in lowered for token in ("serif", "times", "ptserif", "freeserif")):
        return "serif"
    return "sans"


def _font_source(path: str | None) -> str:
    if not path:
        return "fallback"
    normalized = path.replace("\\", "/").lower()
    if ".cache/turkicdocgen/fonts" in normalized:
        return "custom_user"
    if "windows/fonts" in normalized or "/usr/share/fonts" in normalized:
        return "system"
    return "pil_lookup"


def _selected_fonts(plan: PagePlan) -> list[dict[str, str | bool | None]]:
    seen: set[tuple[str, str | None]] = set()
    fonts: list[dict[str, str | bool | None]] = []
    for zone in plan.zones:
        key = (zone.style.font_family, zone.style.font_path)
        if key in seen:
            continue
        seen.add(key)
        fonts.append(
            {
                "family": zone.style.font_family,
                "path": zone.style.font_path,
                "source": _font_source(zone.style.font_path),
                "category": _font_category(
                    zone.style.font_family, zone.style.font_path
                ),
                "coverage_language": canonical_language_mix(plan.language_mix),
                "coverage_ok": True,
            }
        )
    return fonts


def _corpus_metadata(plan: PagePlan) -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for zone in plan.zones:
        record_id = zone.metadata.get("corpus_record_id")
        if not record_id:
            continue
        records[str(record_id)] = {
            "corpus_record_id": record_id,
            "language_mix": zone.metadata.get("language_mix"),
            "domain": zone.metadata.get("domain"),
            "source_type": zone.metadata.get("source_type"),
            "license_note": zone.metadata.get("license_note"),
            "recommended_layouts": zone.metadata.get("recommended_layouts", []),
        }
    return list(records.values())


def _metadata_groups(
    plan: PagePlan,
    qa: QAReport,
    image_rel: str,
) -> dict[str, object]:
    effect_meta = plan.metadata.get("effect_result", {})
    return {
        "identity": {"id": plan.page_id, "page_id": plan.page_id, "image": image_rel},
        "generation": {
            "profile": plan.metadata.get("profile"),
            "seed": plan.metadata.get("seed"),
            "index": plan.metadata.get("index"),
            "regenerate_command": (
                "python -m turkicdocgen generate "
                f"--profile {plan.metadata.get('profile')} "
                f"--seed {plan.metadata.get('seed')} --count 1 "
                f"--language {plan.language_mix} --layout {plan.layout_id} "
                f"--effect {plan.quality_profile} --out outputs/regenerated"
            ),
        },
        "language": {
            "language_mix": plan.language_mix,
            "primary_language": plan.metadata.get("primary_language"),
            "secondary_language": plan.metadata.get("secondary_language"),
            "mixing_features": plan.metadata.get("mixing_features", []),
            "language_mix_ratio": plan.metadata.get("language_mix_ratio", {}),
        },
        "layout": {
            "layout_id": plan.layout_id,
            "shard_id": plan.metadata.get("shard_id"),
            "orientation": plan.metadata.get("orientation", "portrait"),
            "zone_count": len(plan.zones),
            "layout_density": plan.metadata.get("layout_density"),
            "content_height_ratio": plan.metadata.get("content_height_ratio"),
            "date_roles": [
                {"zone": zone, "role": role}
                for zone, role in plan.metadata.get("date_roles", {}).items()
            ],
            "signature_role": plan.metadata.get("signature_role"),
        },
        "corpus": {"records": _corpus_metadata(plan)},
        "render": {
            "quality_profile": plan.quality_profile,
            "paper_base": plan.metadata.get("paper_base"),
        },
        "effects": effect_meta,
        "fonts": {"selected_fonts": _selected_fonts(plan)},
        "qa": {
            "qa_ok": qa.ok,
            "qa_issues": [asdict(issue) for issue in qa.issues],
            "qa_flags": [issue.code for issue in qa.issues],
        },
        "review": {"visual_qa_status": "pending_manual"},
        "release": {},
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl_records(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_zone_jsonl(path: Path, plan: PagePlan, qa: QAReport) -> None:
    append_jsonl(path, {"page": asdict(plan), "qa": asdict(qa)})


def _ocr_record_pair(config: OCRRecordConfig) -> tuple[dict, dict] | None:
    text = config.text
    if text is None or not text.strip():
        return None
    x1, y1, x2, y2 = config.bbox
    if x2 - x1 < config.min_width or y2 - y1 < config.min_height:
        return None
    normalized_polygon = config.polygon or [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]
    shared = {
        "image": config.image_rel,
        "page_id": config.page_id,
        "region_id": config.region_id,
        "line_id": config.region_id,
        "text": text,
    }
    return (
        {**shared, "bbox": config.bbox, "polygon": normalized_polygon},
        shared,
    )


def export_page(out_dir: Path, plan: PagePlan, qa: QAReport, image_rel: str) -> None:
    min_w = QA_CONFIG.get("ocr", {}).get("min_width_px", 4)
    min_h = QA_CONFIG.get("ocr", {}).get("min_height_px", 6)

    zone_records = []
    det_records = []
    rec_records = []
    for zone in plan.zones:
        zone_dict = asdict(zone)
        zone_dict["role"] = zone.metadata.get("role", zone.zone_type)
        zone_dict["is_ocr_target"] = zone.zone_type not in {
            "stamp",
            "decorative_non_text",
        }
        zone_records.append(zone_dict)

        if zone.zone_type in {"stamp", "decorative_non_text"}:
            continue

        for line in zone.lines:
            pair = _ocr_record_pair(
                OCRRecordConfig(
                    image_rel=image_rel,
                    page_id=plan.page_id,
                    region_id=line.line_id,
                    bbox=line.bbox,
                    polygon=line.polygon,
                    text=line.text,
                    min_width=min_w,
                    min_height=min_h,
                )
            )
            if pair is None:
                continue
            det_record, rec_record = pair
            det_records.append(det_record)
            rec_records.append(rec_record)

        for cell in zone.cells:
            cell_line_id = f"{zone.zone_id}_cell_{cell.row}_{cell.col}"
            pair = _ocr_record_pair(
                OCRRecordConfig(
                    image_rel=image_rel,
                    page_id=plan.page_id,
                    region_id=cell_line_id,
                    bbox=cell.bbox,
                    polygon=cell.polygon,
                    text=cell.text,
                    min_width=min_w,
                    min_height=min_h,
                )
            )
            if pair is None:
                continue
            det_record, rec_record = pair
            det_records.append(det_record)
            rec_records.append(rec_record)

    append_jsonl(
        out_dir / "zone_gt.jsonl",
        {"page_id": plan.page_id, "image": image_rel, "zones": zone_records},
    )
    append_jsonl_records(out_dir / "ocr_det.jsonl", det_records)
    append_jsonl_records(out_dir / "ocr_rec.jsonl", rec_records)
    append_jsonl(
        out_dir / "sft.jsonl",
        {
            "page_id": plan.page_id,
            "prompt": "Read the page zones.",
            "response": zone_records,
        },
    )
    metadata_groups = _metadata_groups(plan, qa, image_rel)
    append_jsonl(
        out_dir / "manifest.jsonl",
        {
            "id": plan.page_id,
            "page_id": plan.page_id,
            "image": image_rel,
            "image_path": image_rel,
            "layout_id": plan.layout_id,
            "shard_id": plan.metadata.get("shard_id"),
            "language_mix": plan.language_mix,
            "orientation": plan.metadata.get("orientation", "portrait"),
            "primary_language": plan.metadata.get("primary_language"),
            "secondary_language": plan.metadata.get("secondary_language"),
            "mixing_features": plan.metadata.get("mixing_features", []),
            "language_mix_ratio": plan.metadata.get("language_mix_ratio", {}),
            "quality_profile": plan.quality_profile,
            "effect_metadata": plan.metadata.get("effect_result", {}),
            "effect_profile": plan.metadata.get("effect_result", {}).get(
                "effect_profile", plan.quality_profile
            ),
            "effect_chain": plan.metadata.get("effect_result", {}).get(
                "effect_chain", []
            ),
            "effect_seed": plan.metadata.get("effect_seed"),
            "selected_fonts": _selected_fonts(plan),
            "metadata_groups": metadata_groups,
            "corpus_metadata": _corpus_metadata(plan),
            "content_schema_id": plan.metadata.get("content_schema_id"),
            "layout_variant": plan.metadata.get("layout_variant"),
            "layout_density": plan.metadata.get("layout_density"),
            "date_roles": plan.metadata.get("date_roles", {}),
            "signature_role": plan.metadata.get("signature_role"),
            "content_height_ratio": plan.metadata.get("content_height_ratio"),
            "content_record_ids": plan.metadata.get("content_record_ids", []),
            "generator_schema_version": plan.metadata.get("generator_schema_version"),
            "qa_ok": qa.ok,
            "qa_issues": [asdict(issue) for issue in qa.issues],
            "qa_flags": [issue.code for issue in qa.issues],
            "regenerate_command": (
                "python -m turkicdocgen generate "
                f"--profile {plan.metadata.get('profile')} "
                f"--seed {plan.metadata.get('seed')} --count 1 "
                f"--language {plan.language_mix} --layout {plan.layout_id} "
                f"--effect {plan.quality_profile} --out outputs/regenerated"
            ),
            "zones": zone_records,
        },
    )
    append_jsonl(
        out_dir / "metadata.jsonl",
        {
            "page_id": plan.page_id,
            "profile": plan.metadata.get("profile"),
            "layout_id": plan.layout_id,
            "language_mix": plan.language_mix,
            "orientation": plan.metadata.get("orientation", "portrait"),
            "primary_language": plan.metadata.get("primary_language"),
            "secondary_language": plan.metadata.get("secondary_language"),
            "mixing_features": plan.metadata.get("mixing_features", []),
            "language_mix_ratio": plan.metadata.get("language_mix_ratio", {}),
            "quality_profile": plan.quality_profile,
            "effect_metadata": plan.metadata.get("effect_result", {}),
            "effect_profile": plan.metadata.get("effect_result", {}).get(
                "effect_profile", plan.quality_profile
            ),
            "selected_fonts": _selected_fonts(plan),
            "corpus_metadata": _corpus_metadata(plan),
            "zone_count": len(plan.zones),
            "content_schema_id": plan.metadata.get("content_schema_id"),
            "layout_variant": plan.metadata.get("layout_variant"),
            "layout_density": plan.metadata.get("layout_density"),
            "date_roles": [
                {"zone": zone, "role": role}
                for zone, role in plan.metadata.get("date_roles", {}).items()
            ],
            "signature_role": plan.metadata.get("signature_role"),
            "content_height_ratio": plan.metadata.get("content_height_ratio"),
            "content_record_ids": plan.metadata.get("content_record_ids", []),
            "generator_schema_version": plan.metadata.get("generator_schema_version"),
        },
    )


def write_metadata_parquet_if_available(out_dir: Path) -> Path:
    rows = [
        json.loads(line)
        for line in (out_dir / "metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    try:
        path = out_dir / "metadata.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return path
    except (ImportError, OSError, ValueError):
        return out_dir / "metadata.jsonl"


def write_batch_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_shard_pages(
    out_dir: Path, batch: list[tuple[PagePlan, QAReport, str]]
) -> None:
    min_w = QA_CONFIG.get("ocr", {}).get("min_width_px", 4)
    min_h = QA_CONFIG.get("ocr", {}).get("min_height_px", 6)

    zone_gt_records = []
    det_records = []
    rec_records = []
    sft_records = []
    manifest_records = []
    metadata_records = []

    for plan, qa, image_rel in batch:
        zone_records = []
        for zone in plan.zones:
            zone_dict = asdict(zone)
            zone_dict["role"] = zone.metadata.get("role", zone.zone_type)
            zone_dict["is_ocr_target"] = zone.zone_type not in {
                "stamp",
                "decorative_non_text",
            }
            zone_records.append(zone_dict)

            if zone.zone_type in {"stamp", "decorative_non_text"}:
                continue

            for line in zone.lines:
                pair = _ocr_record_pair(
                    OCRRecordConfig(
                        image_rel=image_rel,
                        page_id=plan.page_id,
                        region_id=line.line_id,
                        bbox=line.bbox,
                        polygon=line.polygon,
                        text=line.text,
                        min_width=min_w,
                        min_height=min_h,
                    )
                )
                if pair is not None:
                    det_records.append(pair[0])
                    rec_records.append(pair[1])

            for cell in zone.cells:
                cell_line_id = f"{zone.zone_id}_cell_{cell.row}_{cell.col}"
                pair = _ocr_record_pair(
                    OCRRecordConfig(
                        image_rel=image_rel,
                        page_id=plan.page_id,
                        region_id=cell_line_id,
                        bbox=cell.bbox,
                        polygon=cell.polygon,
                        text=cell.text,
                        min_width=min_w,
                        min_height=min_h,
                    )
                )
                if pair is not None:
                    det_records.append(pair[0])
                    rec_records.append(pair[1])

        zone_gt_records.append(
            {"page_id": plan.page_id, "image": image_rel, "zones": zone_records}
        )
        sft_records.append(
            {
                "page_id": plan.page_id,
                "prompt": "Read the page zones.",
                "response": zone_records,
            }
        )

        metadata_groups = _metadata_groups(plan, qa, image_rel)
        manifest_records.append(
            {
                "id": plan.page_id,
                "page_id": plan.page_id,
                "image": image_rel,
                "image_path": image_rel,
                "layout_id": plan.layout_id,
                "language_mix": plan.language_mix,
                "orientation": plan.metadata.get("orientation", "portrait"),
                "primary_language": plan.metadata.get("primary_language"),
                "secondary_language": plan.metadata.get("secondary_language"),
                "mixing_features": plan.metadata.get("mixing_features", []),
                "language_mix_ratio": plan.metadata.get("language_mix_ratio", {}),
                "quality_profile": plan.quality_profile,
                "effect_metadata": plan.metadata.get("effect_result", {}),
                "effect_profile": plan.metadata.get("effect_result", {}).get(
                    "effect_profile", plan.quality_profile
                ),
                "effect_chain": plan.metadata.get("effect_result", {}).get(
                    "effect_chain", []
                ),
                "effect_seed": plan.metadata.get("effect_seed"),
                "selected_fonts": _selected_fonts(plan),
                "metadata_groups": metadata_groups,
                "corpus_metadata": _corpus_metadata(plan),
                "content_schema_id": plan.metadata.get("content_schema_id"),
                "layout_variant": plan.metadata.get("layout_variant"),
                "layout_density": plan.metadata.get("layout_density"),
                "date_roles": plan.metadata.get("date_roles", {}),
                "signature_role": plan.metadata.get("signature_role"),
                "content_height_ratio": plan.metadata.get("content_height_ratio"),
                "content_record_ids": plan.metadata.get("content_record_ids", []),
                "generator_schema_version": plan.metadata.get(
                    "generator_schema_version"
                ),
                "qa_ok": qa.ok,
                "qa_issues": [asdict(issue) for issue in qa.issues],
                "qa_flags": [issue.code for issue in qa.issues],
                "regenerate_command": (
                    "python -m turkicdocgen generate "
                    f"--profile {plan.metadata.get('profile')} "
                    f"--seed {plan.metadata.get('seed')} --count 1 "
                    f"--language {plan.language_mix} --layout {plan.layout_id} "
                    f"--effect {plan.quality_profile} --out outputs/regenerated"
                ),
                "zones": zone_records,
            }
        )

        metadata_records.append(
            {
                "page_id": plan.page_id,
                "profile": plan.metadata.get("profile"),
                "layout_id": plan.layout_id,
                "language_mix": plan.language_mix,
                "orientation": plan.metadata.get("orientation", "portrait"),
                "primary_language": plan.metadata.get("primary_language"),
                "secondary_language": plan.metadata.get("secondary_language"),
                "mixing_features": plan.metadata.get("mixing_features", []),
                "language_mix_ratio": plan.metadata.get("language_mix_ratio", {}),
                "quality_profile": plan.quality_profile,
                "effect_metadata": plan.metadata.get("effect_result", {}),
                "effect_profile": plan.metadata.get("effect_result", {}).get(
                    "effect_profile", plan.quality_profile
                ),
                "selected_fonts": _selected_fonts(plan),
                "corpus_metadata": _corpus_metadata(plan),
                "zone_count": len(plan.zones),
                "content_schema_id": plan.metadata.get("content_schema_id"),
                "layout_variant": plan.metadata.get("layout_variant"),
                "layout_density": plan.metadata.get("layout_density"),
                "date_roles": [
                    {"zone": zone, "role": role}
                    for zone, role in plan.metadata.get("date_roles", {}).items()
                ],
                "signature_role": plan.metadata.get("signature_role"),
                "content_height_ratio": plan.metadata.get("content_height_ratio"),
                "content_record_ids": plan.metadata.get("content_record_ids", []),
                "generator_schema_version": plan.metadata.get(
                    "generator_schema_version"
                ),
            }
        )

    write_batch_jsonl(out_dir / "zone_gt.jsonl", zone_gt_records)
    write_batch_jsonl(out_dir / "ocr_det.jsonl", det_records)
    write_batch_jsonl(out_dir / "ocr_rec.jsonl", rec_records)
    write_batch_jsonl(out_dir / "sft.jsonl", sft_records)
    write_batch_jsonl(out_dir / "manifest.jsonl", manifest_records)
    write_batch_jsonl(out_dir / "metadata.jsonl", metadata_records)
