from __future__ import annotations

import importlib.resources
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import yaml

from .config_validation import validate_qa_config
from .languages import FORBIDDEN_LATIN_FALLBACK_TOKENS, canonical_language_mix
from .schema import PagePlan, QAIssue, QAReport

QA_PROFILE_PATH = (
    importlib.resources.files("turkicdocgen") / "configs" / "qa_profile.yaml"
)


def _load_qa_config() -> dict[str, Any]:
    raw = yaml.safe_load(QA_PROFILE_PATH.read_text(encoding="utf-8"))
    return validate_qa_config(raw)


QA_CONFIG = _load_qa_config()["qa"]

SAFE_EFFECT_WARNINGS = {
    "perspective_profile_limited_to_non_geometric_proxy",
    "stamp_used_default_safe_corner",
}

REQUIRED_LAYOUT_ZONES = {
    "official_statement_page": {
        "recipient",
        "applicant",
        "doc_number",
        "title",
        "body",
        "attachment_note",
        "date",
        "signature",
        "stamp_safe",
    },
    "certificate_page": {
        "organization",
        "title",
        "reference",
        "recipient",
        "body",
        "signature",
        "stamp_safe",
    },
    "official_letter_page": {
        "applicant",
        "doc_number",
        "date",
        "recipient",
        "title",
        "body",
        "attachment_note",
        "signature",
        "stamp_safe",
    },
    "memo_page": {
        "sender",
        "recipient",
        "title",
        "subject",
        "body",
        "signature",
    },
    "meeting_minutes_page": {
        "organization",
        "title",
        "meeting_metadata",
        "participants",
        "agenda",
        "decisions",
        "signature",
    },
    "archival_notice_page": {
        "archive_code",
        "title",
        "catalog_metadata",
        "body",
        "archive_footer",
    },
    "lecture_notes_page": {"course", "title", "outline", "notes", "summary"},
    "glossary_page": {"title", "range", "entries_left", "entries_right"},
    "dictionary_entry_page": {"title", "range", "entries_left", "entries_right"},
    "index_page": {"title", "range", "entries_left", "entries_right"},
    "simple_form_page": {"title", "fields", "date", "signature"},
    "application_form_page": {"title", "fields", "date", "signature"},
    "exam_sheet_page": {"title", "fields", "date", "signature"},
    "worksheet_page": {"title", "fields", "date", "signature"},
    "receipt_like_page": {"title", "fields", "date", "signature"},
    "simple_table_page": {"title", "table", "note"},
    "registry_extract_page": {"title", "table", "note"},
    "syllabus_page": {"title", "table", "note"},
    "catalog_entry_page": {"title", "table", "note"},
    "invoice_like_page": {"title", "table", "note"},
    "schedule_table_page": {"title", "table", "note"},
    "exam_register_page": {"title", "table", "note"},
    "inventory_sheet_page": {"title", "table", "note"},
    "attendance_sheet_page": {"title", "table", "note"},
    "wide_schedule_page": {"title", "table", "note"},
    "bulletin_or_newspaper_page": {
        "masthead",
        "issue_metadata",
        "lead_story",
        "column_left",
        "column_right",
        "footer_note",
    },
    "historical_newspaper_page": {
        "masthead",
        "issue_metadata",
        "column_1",
        "column_2",
        "column_3",
    },
}

ABBREVIATION_TOKENS = (
    "ЖСН / ИИН",
    "БСН / БИН",
    "Жеке номер / ИИН",
    "ID",
    "QR",
    "PDF",
    "GPA",
)


BBOX_LENGTH = 4
MIN_POLYGON_POINTS = 3
MAX_DUPLICATE_COUNT = 4
MIN_DUPLICATE_LEN = 3
MIN_BODY_DUPLICATE_LEN = 32
MAX_ALLOWED_OVERLAP = 0.35
SUM_CHECK_EPSILON = 0.01

DEFAULT_MIN_TITLE_FONT = 32
DEFAULT_MIN_BODY_FONT = 20
DEFAULT_MIN_TABLE_FONT = 18
DEFAULT_MIN_METADATA_FONT = 16
TITLE_METADATA_PATTERNS = (
    re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b"),
    re.compile(r"(?:дата выдачи|берилген күн|берілген күні)", re.IGNORECASE),
)
LOW_CARDINALITY_TABLE_TYPES = {
    "sequence",
    "date",
    "amount",
    "department",
    "score",
    "status",
}


def _valid_bbox(bbox: object, width: int, height: int) -> bool:
    if not isinstance(bbox, tuple | list) or len(bbox) != BBOX_LENGTH:
        return False
    x1, y1, x2, y2 = bbox
    return (
        all(isinstance(v, int | float) for v in bbox)
        and 0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    )


def _bbox_intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def _polygon_axes(polygon: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    axes = []
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        edge_x = float(next_point[0]) - float(point[0])
        edge_y = float(next_point[1]) - float(point[1])
        axes.append((-edge_y, edge_x))
    return axes


def _project_polygon(
    polygon: Sequence[Sequence[float]],
    axis: tuple[float, float],
) -> tuple[float, float]:
    values = [
        float(point[0]) * axis[0] + float(point[1]) * axis[1] for point in polygon
    ]
    return min(values), max(values)


def _polygons_intersect_with_area(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    *,
    tolerance: float = 0.5,
) -> bool:
    if len(first) < 3 or len(second) < 3:
        return False
    for axis in [*_polygon_axes(first), *_polygon_axes(second)]:
        first_min, first_max = _project_polygon(first, axis)
        second_min, second_max = _project_polygon(second, axis)
        if min(first_max, second_max) - max(first_min, second_min) <= tolerance:
            return False
    return True


def _check_required_zones(plan: PagePlan, issues: list[QAIssue]) -> None:
    seen_zone_ids = {zone.zone_id for zone in plan.zones}
    required = REQUIRED_LAYOUT_ZONES.get(plan.layout_id, set())
    for zone_id in sorted(required - seen_zone_ids):
        issues.append(
            QAIssue(
                "missing_required_zone",
                "error",
                f"Required zone is missing for {plan.layout_id}",
                zone_id,
            )
        )


def _check_zone_font_size(zone: Any, issues: list[QAIssue]) -> None:
    min_title_font = QA_CONFIG.get("min_title_font_px", DEFAULT_MIN_TITLE_FONT)
    if zone.zone_type == "title" and zone.style.font_size_px < min_title_font:
        issues.append(
            QAIssue(
                "tiny_title_font",
                "error",
                f"Title font below {min_title_font}px",
                zone.zone_id,
            )
        )

    min_body_font = int(
        zone.metadata.get(
            "min_render_font_px",
            QA_CONFIG.get("min_body_font_px", DEFAULT_MIN_BODY_FONT),
        )
    )
    if (
        zone.zone_type in {"paragraph", "body"}
        and zone.style.font_size_px < min_body_font
    ):
        issues.append(
            QAIssue(
                "tiny_body_font",
                "error",
                f"Body font below {min_body_font}px",
                zone.zone_id,
            )
        )

    min_table_font = QA_CONFIG.get("min_table_font_px", DEFAULT_MIN_TABLE_FONT)
    if zone.zone_type == "table" and zone.style.font_size_px < min_table_font:
        issues.append(
            QAIssue(
                "table_cell_font_too_small",
                "error",
                f"Table font below {min_table_font}px",
                zone.zone_id,
            )
        )

    min_metadata_font = QA_CONFIG.get("min_metadata_font_px", DEFAULT_MIN_METADATA_FONT)
    if (
        zone.zone_type
        in {
            "header",
            "footer",
            "date",
            "signature",
            "recipient",
            "applicant",
            "doc_number",
        }
        and zone.style.font_size_px < min_metadata_font
    ):
        issues.append(
            QAIssue(
                "tiny_metadata_font",
                "error",
                f"Metadata font below {min_metadata_font}px",
                zone.zone_id,
            )
        )


def _check_zone_date_signature(
    zone: Any,
    date_values: dict[str, list[str]],
    signature_zones: list[Any],
) -> None:
    date_role = zone.metadata.get("date_role")
    if date_role:
        dates = re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b", zone.text)
        for date in dates:
            date_values.setdefault(date, []).append(str(date_role))
    if zone.metadata.get("signature_role"):
        signature_zones.append(zone)


def _check_zone_latin_fallback(
    zone: Any,
    language_mix: str,
    issues: list[QAIssue],
) -> None:
    if language_mix in {"kk", "ky"}:
        upper_text = zone.text.upper()
        for token in FORBIDDEN_LATIN_FALLBACK_TOKENS:
            if token in upper_text:
                issues.append(
                    QAIssue(
                        "latin_fallback_text",
                        "error",
                        f"Forbidden Latin fallback token: {token}",
                        zone.zone_id,
                    )
                )


class _ZoneValidationState:
    def __init__(self, issues: list[QAIssue]) -> None:
        self.seen_orders: set[int] = set()
        self.date_values: dict[str, list[str]] = {}
        self.signature_zones: list[Any] = []
        self.issues: list[QAIssue] = issues


def _check_single_zone_properties(
    plan: PagePlan,
    zone: Any,
    language_mix: str,
    state: _ZoneValidationState,
) -> tuple[int, bool]:
    if not _valid_bbox(zone.bbox, plan.width, plan.height):
        state.issues.append(
            QAIssue(
                "invalid_zone_box",
                "error",
                f"Invalid bbox {zone.bbox}",
                zone.zone_id,
            )
        )
        return 0, False

    if zone.reading_order in state.seen_orders:
        state.issues.append(
            QAIssue(
                "invalid_reading_order",
                "error",
                "Duplicate reading order",
                zone.zone_id,
            )
        )
    state.seen_orders.add(zone.reading_order)

    if zone.style.align == "justify":
        state.issues.append(
            QAIssue(
                "body_justify",
                "error",
                "Justify is forbidden in dataset pages",
                zone.zone_id,
            )
        )

    _check_zone_font_size(zone, state.issues)
    _check_zone_date_signature(zone, state.date_values, state.signature_zones)
    _check_zone_latin_fallback(zone, language_mix, state.issues)

    if zone.zone_type in {"body", "form"}:
        return len(zone.text.strip()), True
    return 0, True


def _check_zone_lines(
    plan: PagePlan,
    zone: Any,
    min_w: int,
    min_h: int,
    issues: list[QAIssue],
) -> None:
    for line in zone.lines:
        if not line.text or not line.text.strip():
            issues.append(
                QAIssue(
                    "empty_ocr_label",
                    "warning",
                    "Line text is empty or blank",
                    zone.zone_id,
                )
            )
        lx1, ly1, lx2, ly2 = line.bbox
        if (lx2 - lx1) < min_w or (ly2 - ly1) < min_h:
            issues.append(
                QAIssue(
                    "tiny_ocr_box",
                    "warning",
                    f"Line bbox {(lx2 - lx1)}x{(ly2 - ly1)} is below threshold {min_w}x{min_h}",
                    zone.zone_id,
                )
            )
        if not _valid_bbox(line.bbox, plan.width, plan.height):
            issues.append(
                QAIssue(
                    "invalid_line_box",
                    "error",
                    f"Line bbox {line.bbox} is invalid or outside page",
                    zone.zone_id,
                )
            )
        if line.polygon:
            if len(line.polygon) < MIN_POLYGON_POINTS:
                issues.append(
                    QAIssue(
                        "invalid_polygon_geometry",
                        "error",
                        f"Line polygon has less than {MIN_POLYGON_POINTS} points: {line.polygon}",
                        zone.zone_id,
                    )
                )
            elif any(
                not (0 <= pt[0] <= plan.width and 0 <= pt[1] <= plan.height)
                for pt in line.polygon
            ):
                issues.append(
                    QAIssue(
                        "invalid_polygon_geometry",
                        "error",
                        f"Line polygon points outside page: {line.polygon}",
                        zone.zone_id,
                    )
                )


def _check_zone_cells(
    plan: PagePlan,
    zone: Any,
    min_w: int,
    min_h: int,
    min_table_font: int,
    issues: list[QAIssue],
) -> None:
    if not zone.cells:
        return
    x1, y1, x2, y2 = zone.bbox
    for cell in zone.cells:
        if not _valid_bbox(cell.bbox, plan.width, plan.height):
            issues.append(
                QAIssue(
                    "invalid_table_cell",
                    "error",
                    "Invalid table cell bbox",
                    zone.zone_id,
                )
            )
            continue
        cx1, cy1, cx2, cy2 = cell.bbox
        if not (x1 <= cx1 and cx2 <= x2 and y1 <= cy1 and cy2 <= y2):
            issues.append(
                QAIssue(
                    "invalid_table_cell",
                    "error",
                    "Table cell outside table zone",
                    zone.zone_id,
                )
            )
        if not cell.text or not cell.text.strip():
            issues.append(
                QAIssue(
                    "empty_ocr_label",
                    "warning",
                    "Table cell text is empty or blank",
                    zone.zone_id,
                )
            )
        if (cx2 - cx1) < min_w or (cy2 - cy1) < min_h:
            issues.append(
                QAIssue(
                    "tiny_ocr_box",
                    "warning",
                    f"Table cell bbox {(cx2 - cx1)}x{(cy2 - cy1)} is below threshold {min_w}x{min_h}",
                    zone.zone_id,
                )
            )
        rendered_font_size = cell.metadata.get("rendered_font_size")
        if isinstance(rendered_font_size, int) and rendered_font_size < min_table_font:
            issues.append(
                QAIssue(
                    "table_cell_font_too_small",
                    "error",
                    f"Rendered table cell font below {min_table_font}px: {rendered_font_size}px",
                    zone.zone_id,
                )
            )
        rendered_lines = cell.metadata.get("rendered_lines") or []
        normalized_source = re.sub(r"\s+", "", cell.text)
        normalized_rendered = re.sub(r"\s+", "", "".join(rendered_lines))
        if rendered_lines and normalized_source != normalized_rendered:
            issues.append(
                QAIssue(
                    "table_cell_text_truncated",
                    "error",
                    "Rendered table cell text was truncated",
                    zone.zone_id,
                )
            )
        if cell.polygon:
            if len(cell.polygon) < MIN_POLYGON_POINTS:
                issues.append(
                    QAIssue(
                        "invalid_polygon_geometry",
                        "error",
                        f"Cell polygon has less than {MIN_POLYGON_POINTS} points: {cell.polygon}",
                        zone.zone_id,
                    )
                )
            elif any(
                not (0 <= pt[0] <= plan.width and 0 <= pt[1] <= plan.height)
                for pt in cell.polygon
            ):
                issues.append(
                    QAIssue(
                        "invalid_polygon_geometry",
                        "error",
                        f"Cell polygon points outside page: {cell.polygon}",
                        zone.zone_id,
                    )
                )


def _check_zone_structures(
    plan: PagePlan,
    min_w: int,
    min_h: int,
    min_table_font: int,
    language_mix: str,
    issues: list[QAIssue],
) -> tuple[int, dict[str, list[str]], list[Any]]:
    state = _ZoneValidationState(issues)
    visible_chars = 0

    for zone in plan.zones:
        zone_chars, valid_zone = _check_single_zone_properties(
            plan,
            zone,
            language_mix,
            state,
        )
        visible_chars += zone_chars
        if not valid_zone:
            continue
        _check_zone_lines(plan, zone, min_w, min_h, issues)
        _check_zone_cells(plan, zone, min_w, min_h, min_table_font, issues)

    return visible_chars, state.date_values, state.signature_zones


def _check_duplicate_texts(plan: PagePlan, issues: list[QAIssue]) -> None:
    all_texts: list[tuple[str, str | None]] = []
    for zone in plan.zones:
        for line in zone.lines:
            if line.text and line.text.strip():
                value_type = (
                    "body_line" if zone.zone_type in {"body", "paragraph"} else "text"
                )
                all_texts.append((line.text.strip(), value_type))
        for cell in zone.cells:
            if cell.text and cell.text.strip():
                all_texts.append(
                    (cell.text.strip(), str(cell.metadata.get("value_type")))
                )
    if all_texts:
        counts = Counter(all_texts)
        excessive_dupes = [
            f"{txt} [{value_type or 'text'}]"
            for (txt, value_type), count in counts.items()
            if count > MAX_DUPLICATE_COUNT
            and len(txt)
            > (
                MIN_BODY_DUPLICATE_LEN
                if value_type == "body_line"
                else MIN_DUPLICATE_LEN
            )
            and value_type not in LOW_CARDINALITY_TABLE_TYPES
        ]
        if excessive_dupes:
            issues.append(
                QAIssue(
                    "excessive_duplicates",
                    "warning",
                    f"Excessive duplicate text lines found: {excessive_dupes[:3]}",
                )
            )


def _check_rendered_text_outside_cell(plan: PagePlan, issues: list[QAIssue]) -> None:
    for zone in plan.zones:
        if zone.zone_type != "table":
            continue
        for cell in zone.cells:
            rendered_inside = cell.metadata.get("rendered_inside_cell")
            if rendered_inside is False:
                issues.append(
                    QAIssue(
                        "rendered_text_outside_cell",
                        "error",
                        "Rendered glyph bounds extend outside the cell box",
                        zone.zone_id,
                    )
                )
                continue
            if rendered_inside is True:
                continue
            rendered_bbox = cell.metadata.get("rendered_bbox")
            if (
                isinstance(rendered_bbox, list | tuple)
                and len(rendered_bbox) == 4
                and not (
                    cell.bbox[0] <= rendered_bbox[0] <= rendered_bbox[2] <= cell.bbox[2]
                    and cell.bbox[1]
                    <= rendered_bbox[1]
                    <= rendered_bbox[3]
                    <= cell.bbox[3]
                )
            ):
                issues.append(
                    QAIssue(
                        "rendered_text_outside_cell",
                        "error",
                        "Rendered glyph bounds extend outside the cell box",
                        zone.zone_id,
                    )
                )
                continue
            text_y = cell.metadata.get("rendered_text_y")
            line_height = cell.metadata.get("rendered_line_height")
            rendered_lines = cell.metadata.get("rendered_lines") or []
            if not isinstance(text_y, int) or not isinstance(line_height, int):
                continue
            top = text_y
            bottom = text_y + len(rendered_lines) * line_height
            if top < cell.bbox[1] or bottom > cell.bbox[3]:
                issues.append(
                    QAIssue(
                        "rendered_text_outside_cell",
                        "error",
                        "Rendered text extends outside the cell box",
                        zone.zone_id,
                    )
                )


def _check_form_field_overlap(plan: PagePlan, issues: list[QAIssue]) -> None:
    for zone in plan.zones:
        if zone.zone_type != "form":
            continue
        for rendered_field in zone.metadata.get("rendered_fields") or []:
            label_bbox = rendered_field.get("label_bbox")
            value_bbox = rendered_field.get("value_bbox")
            if not label_bbox or not value_bbox:
                continue
            overlaps = max(label_bbox[0], value_bbox[0]) < min(
                label_bbox[2], value_bbox[2]
            ) and max(label_bbox[1], value_bbox[1]) < min(label_bbox[3], value_bbox[3])
            if overlaps or not rendered_field.get("rendered_complete", False):
                issues.append(
                    QAIssue(
                        "form_field_overlap",
                        "error",
                        "Form label and value overlap or do not fit their row",
                        zone.zone_id,
                    )
                )
                break


def _check_title_semantic_mismatch(plan: PagePlan, issues: list[QAIssue]) -> None:
    titles = [
        zone for zone in plan.zones if zone.zone_type == "title" and zone.text.strip()
    ]
    if not titles:
        return
    invalid_title = next(
        (
            zone
            for zone in titles
            if any(pattern.search(zone.text) for pattern in TITLE_METADATA_PATTERNS)
        ),
        None,
    )
    if invalid_title is not None:
        issues.append(
            QAIssue(
                "title_semantic_mismatch",
                "warning",
                "Title contains metadata that belongs in a date or metadata zone",
                invalid_title.zone_id,
            )
        )


def _check_zone_overlaps(plan: PagePlan, issues: list[QAIssue]) -> None:
    for i in range(len(plan.zones)):
        for j in range(i + 1, len(plan.zones)):
            z1 = plan.zones[i]
            z2 = plan.zones[j]
            if z1.zone_type in {"stamp", "decorative_non_text"} or z2.zone_type in {
                "stamp",
                "decorative_non_text",
            }:
                continue
            bbox1 = z1.metadata.get("source_bbox") or z1.bbox
            bbox2 = z2.metadata.get("source_bbox") or z2.bbox
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            if xi1 < xi2 and yi1 < yi2:
                inter_area = (xi2 - xi1) * (yi2 - yi1)
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                min_area = min(area1, area2)
                if min_area > 0 and (inter_area / min_area) > MAX_ALLOWED_OVERLAP:
                    issues.append(
                        QAIssue(
                            "excessive_overlap",
                            "error",
                            f"Excessive overlap between {z1.zone_id} ({z1.zone_type}) and {z2.zone_id} ({z2.zone_type})",
                            z1.zone_id,
                        )
                    )


def _check_empty_page(plan: PagePlan, issues: list[QAIssue]) -> None:
    total_lines_or_cells = sum(
        len(z.lines) + len(z.cells)
        for z in plan.zones
        if z.zone_type not in {"stamp", "decorative_non_text"}
    )
    if total_lines_or_cells == 0:
        issues.append(
            QAIssue(
                "empty_page",
                "error",
                "Page has no useful OCR text lines or cells",
            )
        )


def _check_dates_and_signatures(
    plan: PagePlan,
    date_values: dict[str, list[str]],
    signature_zones: list[Any],
    issues: list[QAIssue],
) -> None:
    for date, roles in date_values.items():
        if len(roles) > 1 and len(set(roles)) == 1:
            issues.append(
                QAIssue(
                    "duplicate_structural_date",
                    "error",
                    f"Date {date} repeats with the same role {roles[0]}",
                )
            )
    if len(signature_zones) > 1:
        issues.append(
            QAIssue(
                "multiple_final_signatures",
                "error",
                f"Found {len(signature_zones)} final signature zones",
            )
        )
    for signature in signature_zones:
        source_bbox = signature.metadata.get("source_bbox", signature.bbox)
        if source_bbox[1] < plan.height * 0.70:
            issues.append(
                QAIssue(
                    "signature_too_high",
                    "error",
                    "Final signature must be in the lower 30% of the page",
                    signature.zone_id,
                )
            )


def _check_required_zones_sparsity(plan: PagePlan, issues: list[QAIssue]) -> None:
    required = REQUIRED_LAYOUT_ZONES.get(plan.layout_id, set())
    for zone in plan.zones:
        if zone.zone_id in required:
            if zone.zone_type in {"stamp", "decorative_non_text"}:
                continue
            if zone.zone_type == "table":
                # Check cell text instead of zone text
                has_content = any(cell.text.strip() for cell in zone.cells)
                if not has_content:
                    issues.append(
                        QAIssue(
                            "required_zone_empty",
                            "error",
                            f"Required table zone '{zone.zone_id}' has no cell text",
                            zone.zone_id,
                        )
                    )
                continue
            text_stripped = zone.text.strip()
            if not text_stripped:
                issues.append(
                    QAIssue(
                        "required_zone_empty",
                        "error",
                        f"Required zone '{zone.zone_id}' is empty",
                        zone.zone_id,
                    )
                )
                continue
            if zone.zone_type == "title" and len(text_stripped) < 3:
                issues.append(
                    QAIssue(
                        "required_zone_sparse",
                        "warning",
                        f"Required title zone '{zone.zone_id}' is implausibly sparse (length {len(text_stripped)})",
                        zone.zone_id,
                    )
                )
            elif zone.zone_type in {"body", "paragraph"} and len(text_stripped) < 15:
                issues.append(
                    QAIssue(
                        "required_zone_sparse",
                        "warning",
                        f"Required body/paragraph zone '{zone.zone_id}' is implausibly sparse (length {len(text_stripped)})",
                        zone.zone_id,
                    )
                )
            elif zone.zone_type == "form" and not zone.metadata.get("rendered_fields"):
                issues.append(
                    QAIssue(
                        "required_zone_sparse",
                        "warning",
                        f"Required form zone '{zone.zone_id}' has no rendered fields",
                        zone.zone_id,
                    )
                )
            elif zone.zone_type == "table" and not zone.cells:
                issues.append(
                    QAIssue(
                        "required_zone_sparse",
                        "warning",
                        f"Required table zone '{zone.zone_id}' has no cells",
                        zone.zone_id,
                    )
                )


def _check_glyph_intersections(plan: PagePlan, issues: list[QAIssue]) -> None:
    for zone in plan.zones:
        zone_r_bbox = zone.metadata.get("rendered_bbox")
        if isinstance(zone_r_bbox, list | tuple) and len(zone_r_bbox) == 4:
            if (
                zone_r_bbox[0] < -3
                or zone_r_bbox[1] < -3
                or zone_r_bbox[2] > plan.width + 3
                or zone_r_bbox[3] > plan.height + 3
            ):
                issues.append(
                    QAIssue(
                        "glyph_intersection_page_boundary",
                        "error",
                        f"Zone rendered bbox {zone_r_bbox} goes outside page boundaries",
                        zone.zone_id,
                    )
                )

        if zone.zone_type == "table":
            for i, cell_a in enumerate(zone.cells):
                r_bbox_a = cell_a.metadata.get("rendered_bbox")
                if not r_bbox_a:
                    continue
                if (
                    r_bbox_a[0] < -3
                    or r_bbox_a[1] < -3
                    or r_bbox_a[2] > plan.width + 3
                    or r_bbox_a[3] > plan.height + 3
                ):
                    issues.append(
                        QAIssue(
                            "glyph_intersection_page_boundary",
                            "error",
                            f"Cell rendered bbox {r_bbox_a} goes outside page boundaries",
                            zone.zone_id,
                        )
                    )
                cx1, cy1, cx2, cy2 = cell_a.bbox
                if (
                    r_bbox_a[0] < cx1 - 3
                    or r_bbox_a[2] > cx2 + 3
                    or r_bbox_a[1] < cy1 - 3
                    or r_bbox_a[3] > cy2 + 3
                ):
                    issues.append(
                        QAIssue(
                            "glyph_intersection_rule",
                            "error",
                            f"Cell rendered bbox {r_bbox_a} extends outside cell boundary {cell_a.bbox}",
                            zone.zone_id,
                        )
                    )
                for j, cell_b in enumerate(zone.cells):
                    if i >= j:
                        continue
                    r_bbox_b = cell_b.metadata.get("rendered_bbox")
                    if not r_bbox_b:
                        continue
                    polygon_a = cell_a.metadata.get("rendered_bbox_polygon") or []
                    polygon_b = cell_b.metadata.get("rendered_bbox_polygon") or []
                    intersects = (
                        _polygons_intersect_with_area(polygon_a, polygon_b)
                        if polygon_a and polygon_b
                        else _bbox_intersects(r_bbox_a, r_bbox_b)
                    )
                    if intersects:
                        issues.append(
                            QAIssue(
                                "glyph_intersection_neighboring_cells",
                                "error",
                                f"Cell {i} rendered bbox intersects with cell {j} rendered bbox",
                                zone.zone_id,
                            )
                        )

        elif zone.zone_type == "form":
            for field in zone.metadata.get("rendered_fields") or []:
                l_bbox = field.get("label_bbox")
                v_bbox = field.get("value_bbox")
                row_bbox = field.get("row_bbox")

                if l_bbox:
                    if (
                        l_bbox[0] < -3
                        or l_bbox[1] < -3
                        or l_bbox[2] > plan.width + 3
                        or l_bbox[3] > plan.height + 3
                    ):
                        issues.append(
                            QAIssue(
                                "glyph_intersection_page_boundary",
                                "error",
                                f"Form label rendered bbox {l_bbox} goes outside page boundaries",
                                zone.zone_id,
                            )
                        )
                    if row_bbox and (
                        l_bbox[1] < row_bbox[1] - 3 or l_bbox[3] > row_bbox[3] + 3
                    ):
                        issues.append(
                            QAIssue(
                                "glyph_intersection_rule",
                                "error",
                                f"Form label rendered bbox {l_bbox} extends outside row boundary {row_bbox}",
                                zone.zone_id,
                            )
                        )
                if v_bbox:
                    if (
                        v_bbox[0] < -3
                        or v_bbox[1] < -3
                        or v_bbox[2] > plan.width + 3
                        or v_bbox[3] > plan.height + 3
                    ):
                        issues.append(
                            QAIssue(
                                "glyph_intersection_page_boundary",
                                "error",
                                f"Form value rendered bbox {v_bbox} goes outside page boundaries",
                                zone.zone_id,
                            )
                        )
                    if row_bbox and (
                        v_bbox[1] < row_bbox[1] - 3 or v_bbox[3] > row_bbox[3] + 3
                    ):
                        issues.append(
                            QAIssue(
                                "glyph_intersection_rule",
                                "error",
                                f"Form value rendered bbox {v_bbox} extends outside row boundary {row_bbox}",
                                zone.zone_id,
                            )
                        )
                if l_bbox and v_bbox:
                    if max(l_bbox[0], v_bbox[0]) < min(l_bbox[2], v_bbox[2]) and max(
                        l_bbox[1], v_bbox[1]
                    ) < min(l_bbox[3], v_bbox[3]):
                        issues.append(
                            QAIssue(
                                "glyph_intersection_label_value",
                                "error",
                                f"Form label bbox {l_bbox} intersects with value bbox {v_bbox}",
                                zone.zone_id,
                            )
                        )
                if field.get("kind") != "field":
                    continue

                label_width = zone.metadata.get("form_label_width")
                separator = zone.metadata.get("form_separator_polygon")
                if (
                    isinstance(separator, list)
                    and len(separator) == 2
                    and all(
                        isinstance(point, list | tuple) and len(point) == 2
                        for point in separator
                    )
                ):
                    (sep_x1, sep_y1), (sep_x2, sep_y2) = separator

                    def separator_x_at(
                        y: float,
                        x1: float = sep_x1,
                        y1: float = sep_y1,
                        x2: float = sep_x2,
                        y2: float = sep_y2,
                    ) -> float:
                        if y2 == y1:
                            return (x1 + x2) / 2
                        ratio = (y - y1) / (y2 - y1)
                        return x1 + ratio * (x2 - x1)

                    label_polygon = field.get("label_bbox_polygon") or []
                    value_polygon = field.get("value_bbox_polygon") or []
                    if label_polygon and any(
                        point[0] > separator_x_at(point[1]) + 3
                        for point in label_polygon
                    ):
                        issues.append(
                            QAIssue(
                                "glyph_intersection_rule",
                                "error",
                                f"Form label polygon {label_polygon} crossed separator "
                                f"rule {separator}",
                                zone.zone_id,
                            )
                        )
                    if value_polygon and any(
                        point[0] < separator_x_at(point[1]) - 3
                        for point in value_polygon
                    ):
                        issues.append(
                            QAIssue(
                                "glyph_intersection_rule",
                                "error",
                                f"Form value polygon {value_polygon} crossed separator "
                                f"rule {separator}",
                                zone.zone_id,
                            )
                        )
                elif label_width:
                    x1, _, _, _ = zone.bbox
                    sep_x = x1 + label_width
                    if l_bbox and l_bbox[2] > sep_x + 3:
                        issues.append(
                            QAIssue(
                                "glyph_intersection_rule",
                                "error",
                                f"Form label rendered bbox {l_bbox} crossed separator rule at x={sep_x}",
                                zone.zone_id,
                            )
                        )
                    if v_bbox and v_bbox[0] < sep_x - 3:
                        issues.append(
                            QAIssue(
                                "glyph_intersection_rule",
                                "error",
                                f"Form value rendered bbox {v_bbox} crossed separator rule at x={sep_x}",
                                zone.zone_id,
                            )
                        )


def _check_ratios_and_densities(
    plan: PagePlan,
    visible_chars: int,
    issues: list[QAIssue],
) -> None:
    from turkicdocgen.page_planning.layouts.registry import LAYOUT_FAMILIES

    minimum_height_ratio = {
        "simple_form_page": 0.70,
        "application_form_page": 0.70,
        "exam_sheet_page": 0.70,
        "worksheet_page": 0.70,
        "receipt_like_page": 0.70,
        "simple_table_page": 0.65,
        "registry_extract_page": 0.65,
        "syllabus_page": 0.65,
        "catalog_entry_page": 0.65,
        "invoice_like_page": 0.65,
        "schedule_table_page": 0.65,
        "exam_register_page": 0.65,
        "inventory_sheet_page": 0.65,
        "attendance_sheet_page": 0.65,
        "wide_schedule_page": 0.65,
        "official_statement_page": 0.62,
        "certificate_page": 0.62,
        "official_letter_page": 0.62,
        "memo_page": 0.62,
        "meeting_minutes_page": 0.62,
        "archival_notice_page": 0.62,
    }.get(plan.layout_id)
    content_height_ratio = float(plan.metadata.get("content_height_ratio", 0.0))
    if minimum_height_ratio is not None and content_height_ratio < minimum_height_ratio:
        issues.append(
            QAIssue(
                "low_content_height_ratio",
                "error",
                f"Content height ratio {content_height_ratio:.3f} below {minimum_height_ratio:.2f}",
            )
        )
    for zone in plan.zones:
        if zone.metadata.get("text_truncated"):
            issues.append(
                QAIssue(
                    "rendered_text_truncated",
                    "error",
                    "Renderer did not fit all planned text",
                    zone.zone_id,
                )
            )

    min_chars = QA_CONFIG.get("min_visible_chars", {}).get(plan.layout_id, 0)
    if visible_chars < min_chars:
        issues.append(
            QAIssue(
                "low_text_density",
                "error",
                f"Visible chars {visible_chars} below {min_chars}",
            )
        )

    family = LAYOUT_FAMILIES.get(plan.layout_id, "other")
    if family == "form":
        field_count = sum(
            zone.metadata.get("rendered_entry_count", zone.text.count(":"))
            for zone in plan.zones
            if zone.zone_type == "form"
        )
        min_fields = (
            QA_CONFIG.get("form_table_density", {})
            .get("simple_form_page", {})
            .get("min_filled_fields", 15)
        )
        if field_count < min_fields:
            issues.append(
                QAIssue(
                    "low_form_density", "error", f"Only {field_count} filled fields"
                )
            )
    elif family == "table":
        cell_count = sum(
            len(zone.cells) for zone in plan.zones if zone.zone_type == "table"
        )
        min_cells = (
            QA_CONFIG.get("form_table_density", {})
            .get("simple_table_page", {})
            .get("min_filled_cells", 30)
        )
        if cell_count < min_cells:
            issues.append(
                QAIssue("low_table_density", "error", f"Only {cell_count} cells")
            )


def _check_language_mixing(
    plan: PagePlan,
    language_mix: str,
    mixing_features: list[str],
    issues: list[QAIssue],
) -> None:
    if language_mix in {"ru_kk", "ru_ky"}:
        marked_zones = [
            zone
            for zone in plan.zones
            if str(zone.language).startswith("bilingual_")
            or zone.language == "ru"
            or zone.metadata.get("mixing_feature")
            or zone.metadata.get("mixing_features")
        ]
        marked_cells = [
            cell
            for zone in plan.zones
            for cell in zone.cells
            if str(cell.language).startswith("bilingual_")
            or cell.language == "ru"
            or cell.metadata.get("mixing_feature")
            or cell.metadata.get("mixing_features")
        ]
        if not marked_zones and not marked_cells:
            issues.append(
                QAIssue(
                    "mixed_language_has_no_secondary_content",
                    "warning",
                    "Mixed-language sample has no bilingual or secondary-language zone",
                )
            )

        declared_without_zone = [
            feature
            for feature in mixing_features
            if not any(
                feature
                in {
                    zone.metadata.get("mixing_feature"),
                    *zone.metadata.get("mixing_features", []),
                }
                for zone in plan.zones
            )
            and not any(
                feature
                in {
                    cell.metadata.get("mixing_feature"),
                    *cell.metadata.get("mixing_features", []),
                }
                for zone in plan.zones
                for cell in zone.cells
            )
        ]
        if declared_without_zone:
            issues.append(
                QAIssue(
                    "missing_zone_language_metadata",
                    "warning",
                    "Declared mixing features have no matching zone metadata: "
                    + ", ".join(declared_without_zone),
                )
            )

        if "field_level" in mixing_features and not any(
            zone.zone_type == "form"
            and (
                str(zone.language).startswith("bilingual_")
                or zone.metadata.get("mixing_feature") == "field_level"
                or "field_level" in zone.metadata.get("mixing_features", [])
            )
            for zone in plan.zones
        ):
            issues.append(
                QAIssue(
                    "declared_mixing_feature_not_present",
                    "warning",
                    "field_level is declared but no mixed form zone is present",
                )
            )

        if "table_level" in mixing_features and not any(
            zone.zone_type == "table"
            and (
                zone.metadata.get("mixing_feature") == "table_level"
                or "table_level" in zone.metadata.get("mixing_features", [])
                or any(
                    cell.metadata.get("mixing_feature") == "table_level"
                    or "table_level" in cell.metadata.get("mixing_features", [])
                    for cell in zone.cells
                )
            )
            for zone in plan.zones
        ):
            issues.append(
                QAIssue(
                    "declared_mixing_feature_not_present",
                    "warning",
                    "table_level is declared but no mixed table zone is present",
                )
            )

        if "abbreviation_level" in mixing_features:
            all_text = "\n".join(
                [zone.text for zone in plan.zones]
                + [cell.text for zone in plan.zones for cell in zone.cells]
            )
            if not any(token in all_text for token in ABBREVIATION_TOKENS):
                issues.append(
                    QAIssue(
                        "declared_mixing_feature_not_present",
                        "warning",
                        "abbreviation_level is declared but no abbreviation token is present",
                    )
                )


DEFAULT_MAX_PEN_CROSSED = 2
DEFAULT_OCR_MIN_WIDTH = 4
DEFAULT_OCR_MIN_HEIGHT = 6


def _check_extreme_geometry_matrices(transform: dict, issues: list[QAIssue]) -> None:
    if not transform.get("forward") or not transform.get("inverse"):
        issues.append(
            QAIssue(
                "missing_extreme_transform",
                "error",
                "Extreme geometry requires forward and inverse matrices",
            )
        )
    else:
        try:
            forward = np.asarray(transform["forward"], dtype=float)
            inverse = np.asarray(transform["inverse"], dtype=float)
            matrices_match = np.allclose(forward @ inverse, np.eye(3), atol=1e-5)
        except (TypeError, ValueError):
            matrices_match = False
        if not matrices_match:
            issues.append(
                QAIssue(
                    "invalid_inverse_transform",
                    "error",
                    "Forward and inverse geometry matrices are inconsistent",
                )
            )


def _check_extreme_geometry_zones(plan: PagePlan, issues: list[QAIssue]) -> None:
    for zone in plan.zones:
        if not zone.polygon:
            issues.append(
                QAIssue(
                    "missing_polygon",
                    "error",
                    "Extreme geometry is applied but zone polygon is missing",
                    zone.zone_id,
                )
            )
        for line in zone.lines:
            if not line.polygon:
                issues.append(
                    QAIssue(
                        "missing_polygon",
                        "error",
                        "Extreme geometry is applied but line polygon is missing",
                        zone.zone_id,
                    )
                )
        for cell in zone.cells:
            if not cell.polygon:
                issues.append(
                    QAIssue(
                        "missing_polygon",
                        "error",
                        "Extreme geometry is applied but table cell polygon is missing",
                        zone.zone_id,
                    )
                )
        if any(
            not (0 <= point[0] < plan.width and 0 <= point[1] < plan.height)
            for point in zone.polygon
        ):
            issues.append(
                QAIssue(
                    "extreme_geometry_outside_canvas",
                    "error",
                    "Extreme geometry moved a zone outside the canvas",
                    zone.zone_id,
                )
            )


def _check_geometry_transforms(
    plan: PagePlan,
    effect_result: dict,
    issues: list[QAIssue],
) -> None:
    geometry_tier = effect_result.get("geometry_tier")
    transform = effect_result.get("transform", {})
    if geometry_tier == "extreme":
        _check_extreme_geometry_matrices(transform, issues)
        _check_extreme_geometry_zones(plan, issues)


def _check_pen_artifacts(
    plan: PagePlan,
    effect_result: dict,
    issues: list[QAIssue],
) -> None:
    pen_artifacts: list[dict] = []
    exact_parameters = effect_result.get("exact_parameters", {})
    for name in ("signature_marks", "sparse_pen_marks", "underlines_checks"):
        parameters = exact_parameters.get(name, {})
        if isinstance(parameters, dict):
            pen_artifacts.extend(parameters.get("artifacts", []))
    for artifact in pen_artifacts:
        required = {
            "type",
            "color_rgb",
            "opacity",
            "stroke_width",
            "target_zone",
            "polygon",
            "bbox",
        }
        if not required.issubset(artifact):
            issues.append(
                QAIssue(
                    "incomplete_pen_metadata",
                    "error",
                    "Pen artifact metadata is incomplete",
                )
            )

    body_targets = {
        zone.zone_id for zone in plan.zones if zone.zone_type in {"body", "paragraph"}
    }
    max_crossed = QA_CONFIG.get("effects", {}).get(
        "max_pen_crossed_body_lines", DEFAULT_MAX_PEN_CROSSED
    )
    if (
        sum(artifact.get("target_zone") in body_targets for artifact in pen_artifacts)
        > max_crossed
    ):
        issues.append(
            QAIssue(
                "pen_body_crossing_budget",
                "error",
                f"Pen artifacts cross more than {max_crossed} body regions",
            )
        )


CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LATIN_RE = re.compile(r"[a-zA-Z]")
KAZAKH_ONLY_CHARS = set("ӘәҒғҚқҰұҺһІі")


def _has_latin_confusable(word: str) -> bool:
    if CYRILLIC_RE.search(word) and LATIN_RE.search(word):
        if "@" in word:
            return False
        if (
            "http" in word
            or "www." in word
            or any(
                word.endswith(ext)
                for ext in (".com", ".kz", ".kg", ".ru", ".org", ".net")
            )
        ):
            return False
        if any(c.isdigit() for c in word):
            return False
        return True
    return False


def _check_latin_confusables(plan: PagePlan, issues: list[QAIssue]) -> None:
    for zone in plan.zones:
        if zone.text:
            for word in zone.text.split():
                if _has_latin_confusable(word):
                    issues.append(
                        QAIssue(
                            "latin_confusable_detected",
                            "error",
                            f"Latin confusable detected in Cyrillic word: {word}",
                            zone.zone_id,
                        )
                    )
                    break
        for cell in zone.cells:
            if cell.text:
                for word in cell.text.split():
                    if _has_latin_confusable(word):
                        issues.append(
                            QAIssue(
                                "latin_confusable_detected",
                                "error",
                                f"Latin confusable detected in Cyrillic word: {word}",
                                zone.zone_id,
                            )
                        )
                        break


def _check_character_inventories(plan: PagePlan, issues: list[QAIssue]) -> None:
    lang = canonical_language_mix(plan.language_mix)
    is_kyrgyz = lang in {"ky", "ru_ky"}
    for zone in plan.zones:
        if zone.text and is_kyrgyz:
            bad_chars = [c for c in zone.text if c in KAZAKH_ONLY_CHARS]
            if bad_chars:
                issues.append(
                    QAIssue(
                        "unsupported_character_detected",
                        "error",
                        f"Kazakh character '{bad_chars[0]}' is not allowed in Kyrgyz page",
                        zone.zone_id,
                    )
                )
        for cell in zone.cells:
            if cell.text and is_kyrgyz:
                bad_chars = [c for c in cell.text if c in KAZAKH_ONLY_CHARS]
                if bad_chars:
                    issues.append(
                        QAIssue(
                            "unsupported_character_detected",
                            "error",
                            f"Kazakh character '{bad_chars[0]}' is not allowed in Kyrgyz page",
                            zone.zone_id,
                        )
                    )


def _check_nfc_normalization(plan: PagePlan, issues: list[QAIssue]) -> None:
    for zone in plan.zones:
        if zone.text and unicodedata.normalize("NFC", zone.text) != zone.text:
            issues.append(
                QAIssue(
                    "text_not_nfc_normalized",
                    "error",
                    "Zone text is not NFC normalized",
                    zone.zone_id,
                )
            )
            break
        for cell in zone.cells:
            if cell.text and unicodedata.normalize("NFC", cell.text) != cell.text:
                issues.append(
                    QAIssue(
                        "text_not_nfc_normalized",
                        "error",
                        "Cell text is not NFC normalized",
                        zone.zone_id,
                    )
                )
                break


def validate_page_plan(plan: PagePlan) -> QAReport:
    min_w = QA_CONFIG.get("ocr", {}).get("min_width_px", DEFAULT_OCR_MIN_WIDTH)
    min_h = QA_CONFIG.get("ocr", {}).get("min_height_px", DEFAULT_OCR_MIN_HEIGHT)
    min_table_font = QA_CONFIG.get("min_table_font_px", DEFAULT_MIN_TABLE_FONT)
    issues: list[QAIssue] = []

    _check_required_zones(plan, issues)
    _check_required_zones_sparsity(plan, issues)

    language_mix = canonical_language_mix(plan.language_mix)
    mixing_features = plan.metadata.get("mixing_features", [])
    if not isinstance(mixing_features, list):
        mixing_features = []

    visible_chars, date_values, signature_zones = _check_zone_structures(
        plan, min_w, min_h, min_table_font, language_mix, issues
    )

    _check_duplicate_texts(plan, issues)
    _check_zone_overlaps(plan, issues)
    _check_glyph_intersections(plan, issues)
    _check_rendered_text_outside_cell(plan, issues)
    _check_form_field_overlap(plan, issues)
    _check_title_semantic_mismatch(plan, issues)
    _check_empty_page(plan, issues)

    _check_dates_and_signatures(plan, date_values, signature_zones, issues)
    _check_ratios_and_densities(plan, visible_chars, issues)

    for effect in plan.effects:
        for warning in effect.warnings:
            severity = "warning" if warning in SAFE_EFFECT_WARNINGS else "error"
            issues.append(QAIssue("effect_safety", severity, warning))

    _check_language_mixing(plan, language_mix, mixing_features, issues)

    ratio = plan.metadata.get("language_mix_ratio")
    primary_language = plan.metadata.get("primary_language")
    secondary_language = plan.metadata.get("secondary_language")
    expected_keys = {primary_language, "ru", "en"} - {None}
    ratio_valid = (
        isinstance(ratio, dict)
        and expected_keys.issubset(ratio)
        and all(
            isinstance(value, int | float) and 0.0 <= float(value) <= 1.0
            for value in ratio.values()
        )
        and abs(sum(float(value) for value in ratio.values()) - 1.0)
        <= SUM_CHECK_EPSILON
        and (language_mix not in {"ru_kk", "ru_ky"} or secondary_language == "ru")
    )
    if not ratio_valid:
        issues.append(
            QAIssue(
                "invalid_language_mix_ratio",
                "warning",
                "Language mix ratio metadata is missing, incomplete, or does not sum to 1",
            )
        )

    effect_result = plan.metadata.get("effect_result", {})
    _check_geometry_transforms(plan, effect_result, issues)
    _check_pen_artifacts(plan, effect_result, issues)
    _check_latin_confusables(plan, issues)
    _check_character_inventories(plan, issues)
    _check_nfc_normalization(plan, issues)

    return QAReport(ok=not any(i.severity == "error" for i in issues), issues=issues)
