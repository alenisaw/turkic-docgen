from __future__ import annotations

from typing import TYPE_CHECKING

from turkicdocgen.schema import TextStyle, Zone

if TYPE_CHECKING:
    import random
    from collections.abc import Callable

    LayoutBuilder = Callable[..., list[Zone]]

from . import book, official, specialized, structured
from .variants import choose_variant, get_variant_properties

CORE_LAYOUTS = (
    "book_page_single_column",
    "book_page_two_columns",
    "academic_abstract_page",
    "official_statement_page",
    "official_letter_page",
    "simple_form_page",
    "simple_table_page",
    "bulletin_or_newspaper_page",
    "application_form_page",
    "certificate_page",
    "memo_page",
    "meeting_minutes_page",
    "registry_extract_page",
    "exam_sheet_page",
    "worksheet_page",
    "syllabus_page",
    "lecture_notes_page",
    "archival_notice_page",
    "historical_newspaper_page",
    "catalog_entry_page",
    "invoice_like_page",
    "receipt_like_page",
    "schedule_table_page",
    "glossary_page",
    "dictionary_entry_page",
    "index_page",
    "exam_register_page",
    "inventory_sheet_page",
    "attendance_sheet_page",
    "wide_schedule_page",
)

Bounds = tuple[int, int, int, int]
GENERIC_VARIANT_FAMILIES = {"official", "specialized", "reference", "structured"}


LAYOUT_FAMILIES = {
    "book_page_single_column": "book",
    "book_page_two_columns": "book",
    "academic_abstract_page": "book",
    "official_statement_page": "official",
    "official_letter_page": "official",
    "simple_form_page": "form",
    "application_form_page": "form",
    "exam_sheet_page": "form",
    "worksheet_page": "form",
    "receipt_like_page": "form",
    "simple_table_page": "table",
    "registry_extract_page": "table",
    "syllabus_page": "table",
    "catalog_entry_page": "table",
    "invoice_like_page": "table",
    "schedule_table_page": "table",
    "exam_register_page": "table",
    "inventory_sheet_page": "table",
    "attendance_sheet_page": "table",
    "wide_schedule_page": "table",
    "certificate_page": "specialized",
    "memo_page": "specialized",
    "meeting_minutes_page": "specialized",
    "lecture_notes_page": "specialized",
    "archival_notice_page": "specialized",
    "historical_newspaper_page": "specialized",
    "glossary_page": "reference",
    "dictionary_entry_page": "reference",
    "index_page": "reference",
    "bulletin_or_newspaper_page": "structured",
}


def _variant_line(
    *,
    zone_id: str,
    bbox: Bounds,
    order: int,
    orientation: str,
    role: str,
) -> Zone:
    x1, y1, x2, y2 = bbox
    return Zone(
        zone_id=zone_id,
        zone_type="decorative_non_text",
        bbox=bbox,
        polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
        reading_order=order,
        text="",
        language="",
        style=TextStyle("DejaVuSans", 18),
        metadata={
            "role": role,
            "orientation": orientation,
            "stroke_width": 1,
            "color": (112, 112, 112),
            "variant_generated": True,
        },
        lines=[],
        cells=[],
    )


def _apply_generic_variant_structure(
    zones: list[Zone],
    *,
    family: str,
    variant_id: str,
    bounds: Bounds,
    index: int,
) -> list[Zone]:
    if family not in GENERIC_VARIANT_FAMILIES:
        return zones
    props = get_variant_properties(family, variant_id)
    if not props:
        return zones

    filtered = []
    density = str(props.get("density", "standard"))
    density_factor = {"dense": 0.92, "extended": 1.08}.get(density, 1.0)
    for item in zones:
        item.style.line_spacing = max(1.0, item.style.line_spacing * density_factor)
        item.metadata["layout_density"] = density
        filtered.append(item)

    x1, y1, x2, y2 = bounds
    micro_offset = ((index * 17) % 29) - 14
    next_order = max((item.reading_order for item in filtered), default=-1) + 1
    decorations: list[Zone] = []

    def add_line(
        suffix: str,
        bbox: Bounds,
        orientation: str,
        role: str,
    ) -> None:
        nonlocal next_order
        decorations.append(
            _variant_line(
                zone_id=f"variant_{variant_id}_{suffix}",
                bbox=bbox,
                order=next_order,
                orientation=orientation,
                role=role,
            )
        )
        next_order += 1

    registration_width = 18 + ((index * 11) % 37)
    registration_y = y1 + 10 + micro_offset
    add_line(
        "registration_mark",
        (
            x2 - registration_width,
            registration_y,
            x2,
            registration_y + 2,
        ),
        "horizontal",
        "registration_alignment_mark",
    )

    if props.get("has_frame"):
        add_line("frame_top", (x1, y1, x2, y1 + 2), "horizontal", "frame")
        add_line("frame_bottom", (x1, y2 - 2, x2, y2), "horizontal", "frame")
        add_line("frame_left", (x1, y1, x1 + 2, y2), "vertical", "frame")
        add_line("frame_right", (x2 - 2, y1, x2, y2), "vertical", "frame")
    if props.get("has_header") or props.get("has_masthead"):
        add_line(
            "header_rule",
            (x1, y1 + 24 + micro_offset, x2, y1 + 26 + micro_offset),
            "horizontal",
            "header_rule",
        )
    if props.get("has_footer") or props.get("has_footer_note"):
        add_line(
            "footer_rule",
            (x1, y2 - 26 - micro_offset, x2, y2 - 24 - micro_offset),
            "horizontal",
            "footer_rule",
        )
    if props.get("has_separator") or props.get("has_lines"):
        add_line(
            "section_rule",
            (x1, y1 + 52 + micro_offset, x2, y1 + 54 + micro_offset),
            "horizontal",
            "section_rule",
        )
    if props.get("has_sidebar"):
        add_line("sidebar", (x1 + 28, y1, x1 + 30, y2), "vertical", "sidebar_rule")

    columns = max(1, int(props.get("columns", 1)))
    for column in range(1, columns):
        x = x1 + ((x2 - x1) * column // columns)
        add_line(
            f"column_{column}",
            (x - 1, y1 + 70, x + 1, y2 - 40),
            "vertical",
            "column_rule",
        )
    if props.get("has_signature"):
        add_line(
            "signature_anchor",
            (x2 - 260, y2 - 64, x2, y2 - 62),
            "horizontal",
            "signature_rule",
        )
    if props.get("has_stamp"):
        add_line(
            "stamp_anchor",
            (x1, y2 - 90, x1 + 130, y2 - 88),
            "horizontal",
            "stamp_anchor",
        )
    if props.get("has_range"):
        add_line(
            "range_marker",
            (x1 + 160, y1 + 12, x1 + 320, y1 + 14),
            "horizontal",
            "range_marker",
        )
    if props.get("has_page_num"):
        add_line(
            "page_number_anchor",
            (x2 - 80, y2 - 18, x2, y2 - 16),
            "horizontal",
            "page_number_anchor",
        )
    for item in filtered + decorations:
        item.metadata["layout_variant_properties"] = dict(props)
    return filtered + decorations


def build_layout(
    layout_id: str,
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: Bounds,
    variant_id: str | None = None,
) -> list[Zone]:
    builders: dict[str, LayoutBuilder] = {
        "book_page_single_column": book.single_column,
        "book_page_two_columns": book.two_columns,
        "academic_abstract_page": book.academic_abstract,
        "official_statement_page": official.statement,
        "official_letter_page": official.letter,
        "simple_form_page": structured.form,
        "simple_table_page": structured.table,
        "bulletin_or_newspaper_page": structured.bulletin,
        "application_form_page": structured.form,
        "certificate_page": specialized.certificate,
        "memo_page": specialized.memo,
        "meeting_minutes_page": specialized.meeting_minutes,
        "registry_extract_page": structured.table,
        "exam_sheet_page": structured.form,
        "worksheet_page": structured.form,
        "syllabus_page": structured.table,
        "lecture_notes_page": specialized.lecture_notes,
        "archival_notice_page": specialized.archival_notice,
        "historical_newspaper_page": specialized.historical_newspaper,
        "catalog_entry_page": structured.table,
        "invoice_like_page": structured.table,
        "receipt_like_page": structured.form,
        "schedule_table_page": structured.table,
        "glossary_page": specialized.reference_page,
        "dictionary_entry_page": specialized.reference_page,
        "index_page": specialized.reference_page,
        "exam_register_page": structured.table,
        "inventory_sheet_page": structured.table,
        "attendance_sheet_page": structured.table,
        "wide_schedule_page": structured.table,
    }
    try:
        builder = builders[layout_id]
    except KeyError as exc:
        raise ValueError(f"unknown dataset layout: {layout_id}") from exc

    family = LAYOUT_FAMILIES.get(layout_id, "other")
    if variant_id is None:
        # Check orientation from bounds
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        orientation = "landscape" if w > h else "portrait"
        variant_id = choose_variant(family, orientation, index)

    if layout_id in {
        "simple_form_page",
        "application_form_page",
        "exam_sheet_page",
        "worksheet_page",
        "receipt_like_page",
        "simple_table_page",
        "registry_extract_page",
        "syllabus_page",
        "catalog_entry_page",
        "invoice_like_page",
        "schedule_table_page",
        "glossary_page",
        "dictionary_entry_page",
        "index_page",
        "exam_register_page",
        "inventory_sheet_page",
        "attendance_sheet_page",
        "wide_schedule_page",
    }:
        zones = builder(
            index=index,
            language=language,
            rng=rng,
            bounds=bounds,
            layout_id=layout_id,
            variant_id=variant_id,
        )
    else:
        zones = builder(
            index=index,
            language=language,
            rng=rng,
            bounds=bounds,
            variant_id=variant_id,
        )

    zones = _apply_generic_variant_structure(
        zones,
        family=family,
        variant_id=variant_id,
        bounds=bounds,
        index=index,
    )
    for zone in zones:
        zone.metadata["layout_variant_id"] = variant_id

    return zones
