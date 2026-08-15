from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from turkicdocgen.page_planning.content.document_models import (
    bilingual,
    build_document_context,
    choose_density,
    form_schemas,
    table_schemas,
    value_for,
)
from turkicdocgen.page_planning.content.phrase_builder import (
    sample_seed_record,
    seed_record_metadata,
)
from turkicdocgen.schema import TableCell, Zone

from .common import ZoneConfig, style, text, zone

if TYPE_CHECKING:
    import random


@dataclass(frozen=True)
class StructuredTableConfig:
    schema: Any
    context: Any
    language: str
    rng: random.Random
    table_bbox: tuple[int, int, int, int]
    rows: int


TABLE_SCHEMA_BY_LAYOUT = {
    "registry_extract_page": "registry_extract",
    "syllabus_page": "syllabus",
    "catalog_entry_page": "catalog_entry",
    "invoice_like_page": "invoice_like",
    "schedule_table_page": "schedule_table",
    "exam_register_page": "academic_results",
    "inventory_sheet_page": "inventory",
    "attendance_sheet_page": "attendance_sheet",
    "wide_schedule_page": "schedule_table",
}
TABLE_LAYOUT_ROW_RANGES = {
    "simple_table_page": (14, 20),
    "registry_extract_page": (15, 22),
    "syllabus_page": (12, 18),
    "catalog_entry_page": (14, 20),
    "invoice_like_page": (13, 19),
    "schedule_table_page": (14, 20),
    "exam_register_page": (14, 20),
    "inventory_sheet_page": (14, 20),
    "attendance_sheet_page": (14, 20),
    "wide_schedule_page": (14, 20),
}
TABLE_LAYOUT_GEOMETRY = {
    "registry_extract_page": (225, 175, 45),
    "syllabus_page": (285, 220, 80),
    "catalog_entry_page": (205, 155, 30),
    "invoice_like_page": (285, 190, 85),
    "schedule_table_page": (255, 190, 65),
    "exam_register_page": (255, 190, 65),
    "inventory_sheet_page": (255, 190, 65),
    "attendance_sheet_page": (255, 190, 65),
    "wide_schedule_page": (255, 190, 65),
}


def _select_schema(
    schemas: list[Any], layout_id: str | None, rng: random.Random
) -> Any:
    schema_id = TABLE_SCHEMA_BY_LAYOUT.get(layout_id or "")
    if schema_id is None:
        return rng.choice(schemas)
    return next(
        (schema for schema in schemas if schema.schema_id == schema_id), schemas[0]
    )


def _row_ranges(layout_id: str | None) -> dict[str, tuple[int, int]]:
    default = {"standard": (10, 14), "dense": (14, 18), "extended": (18, 22)}
    bounds = TABLE_LAYOUT_ROW_RANGES.get(layout_id or "")
    if bounds is None:
        return default
    (low, high) = bounds
    return {
        "standard": (low, max(low, high - 3)),
        "dense": (low + 2, high),
        "extended": (max(low + 3, high - 2), high + 3),
    }


def _axis_sizes(total: int, weights: list[float]) -> list[int]:
    weight_sum = sum(weights)
    sizes = [int(total * weight / weight_sum) for weight in weights]
    sizes[-1] += total - sum(sizes)
    return sizes


def _max_data_rows_for_table(
    table_bbox: tuple[int, int, int, int],
    column_count: int,
    layout_id: str | None = None,
) -> int:
    available_height = table_bbox[3] - table_bbox[1]
    if layout_id == "simple_table_page":
        # Simple tables use broad, mixed schemas in a portrait page. At release
        # scale bilingual headers and every-fourth-row notes regularly need
        # three lines, so keep the row budget conservative instead of relying
        # on renderer truncation.
        return max(1, min(20, (available_height - 72) // 70))
    min_line_height = 23
    padding = 12
    header_lines = 3 if column_count >= 5 else 2
    header_height = header_lines * min_line_height + padding
    data_rows = 1
    while True:
        note_rows = data_rows // 4
        required = (
            header_height
            + data_rows * (min_line_height + padding)
            + note_rows * min_line_height
        )
        if required > available_height:
            return max(1, data_rows - 1)
        data_rows += 1


def _variant_number(variant_id: str | None) -> int:
    if not variant_id:
        return 0
    try:
        return int(variant_id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _schedule_table_offsets(
    layout_id: str | None,
    variant_id: str | None,
) -> tuple[int, int, int, int]:
    if layout_id not in {"schedule_table_page", "wide_schedule_page"}:
        return (0, 0, 0, 0)
    variant = _variant_number(variant_id)
    top_offsets = (-18, 0, 16, 34, -8, 24)
    footer_offsets = (0, 24, -14, 36, 12, -6)
    inset_offsets = (0, 18, 34, 52, 10, 42)
    metadata_offsets = (0, 12, -6, 18, 6, -10)
    index = variant % len(top_offsets)
    return (
        top_offsets[index],
        footer_offsets[index],
        inset_offsets[index],
        metadata_offsets[index],
    )


def _table_visual_metadata(
    layout_id: str | None,
    variant_id: str | None,
    props: dict[str, Any],
) -> dict[str, Any]:
    if not layout_id or layout_id not in TABLE_SCHEMA_BY_LAYOUT:
        return {}
    variant = _variant_number(variant_id)
    header_fills = ((236, 238, 241), (242, 239, 232), (234, 242, 238), (240, 240, 240))
    band_fills = ((250, 250, 250), (247, 249, 252), (250, 248, 244), None)
    return {
        "visual_variant": f"schedule_grid_{variant % 12:02d}",
        "header_fill": header_fills[variant % len(header_fills)],
        "row_band_fill": band_fills[variant % len(band_fills)],
        "row_band_period": 2 + (variant % 3),
        "outer_border_width": 2 if props.get("has_frame") else 1,
        "grid_line_width": 2 if props.get("has_lines") and variant % 2 == 0 else 1,
        "grid_color": (48, 48, 48) if props.get("has_lines") else (72, 72, 72),
    }


def _table_title(schema_title: str, layout_id: str | None, index: int) -> str:
    if layout_id == "simple_table_page":
        return f"{schema_title} #{index:06d}"
    return schema_title


def _build_table_cells(config: StructuredTableConfig) -> list[TableCell]:
    schema = config.schema
    context = config.context
    language = config.language
    rng = config.rng
    table_bbox = config.table_bbox
    rows = config.rows
    (x1, y1, x2, y2) = table_bbox
    widths = _axis_sizes(x2 - x1, [column.weight for column in schema.columns])
    row_weights = [1.12, *[1.16 if row % 4 == 0 else 1.0 for row in range(1, rows)]]
    row_heights = _axis_sizes(y2 - y1, row_weights)
    row_tops = [y1]
    for height in row_heights:
        row_tops.append(row_tops[-1] + height)
    cells: list[TableCell] = []
    order = 10
    for row in range(rows):
        cursor = x1
        for col, column in enumerate(schema.columns):
            cx2 = x2 if col == len(schema.columns) - 1 else cursor + widths[col]
            value = (
                column.label
                if row == 0
                else value_for(column.value_type, column.key, context, row - 1, rng)
            )
            if row > 0 and column.value_type == "note" and (row % 4 == 0):
                value = f"{value}\n{context.department}"
            bbox = (cursor, row_tops[row], cx2, row_tops[row + 1])
            cells.append(
                TableCell(
                    row,
                    col,
                    bbox,
                    value,
                    language,
                    order,
                    {
                        "column_key": column.key,
                        "value_type": column.value_type,
                        "align": column.align,
                        "header": row == 0,
                    },
                    [
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[1]),
                        (bbox[2], bbox[3]),
                        (bbox[0], bbox[3]),
                    ],
                )
            )
            cursor = cx2
            order += 1
    return cells


def form(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    layout_id: str | None = None,
    variant_id: str | None = None,
) -> list[Zone]:
    (m_left, m_top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    all_schemas = form_schemas(language)
    if layout_id == "application_form_page":
        schema = next(
            (s for s in all_schemas if s.schema_id == "application_form"),
            all_schemas[0],
        )
    elif layout_id == "exam_sheet_page":
        schema = next(
            (s for s in all_schemas if s.schema_id == "exam_sheet"), all_schemas[0]
        )
    elif layout_id == "worksheet_page":
        schema = next(
            (s for s in all_schemas if s.schema_id == "worksheet"), all_schemas[0]
        )
    elif layout_id == "receipt_like_page":
        schema = next(
            (s for s in all_schemas if s.schema_id == "receipt_like"), all_schemas[0]
        )
    else:
        schema = rng.choice(all_schemas)
    from .variants import get_variant_properties

    props = get_variant_properties("form", variant_id) if variant_id else {}
    density = props.get("density", choose_density(rng))
    (fields_top, footer_space, title_height) = {
        "application_form_page": (105, 245, 110),
        "exam_sheet_page": (145, 310, 90),
        "worksheet_page": (95, 205, 75),
        "receipt_like_page": (175, 360, 120),
    }.get(layout_id, (130, 260, 86))
    fields_top += rng.randint(-18, 22)
    footer_space += rng.randint(-20, 26)
    title_height += rng.randint(-8, 12)
    form_inset_left = rng.randint(0, 22)
    form_inset_right = rng.randint(0, 22)
    lines: list[str] = []
    field_types: list[dict[str, str]] = []
    row = 0
    for section in schema.sections:
        lines.append(f"[{section.title}]")
        for field in section.fields:
            if field.key == "date" and schema.schema_id != "document_registration":
                continue
            lines.append(
                f"{field.label}: {value_for(field.value_type, field.key, context, row, rng)}"
            )
            field_types.append(
                {"key": field.key, "type": field.value_type, "section": section.title}
            )
            row += 1
    document_date = datetime.datetime.strptime(context.date, "%d.%m.%Y")
    received_date = (document_date + datetime.timedelta(days=1 + index % 5)).strftime(
        "%d.%m.%Y"
    )
    date_text = (
        received_date if schema.schema_id == "document_registration" else context.date
    )
    date_role = (
        "received_date"
        if schema.schema_id == "document_registration"
        else "document_date"
    )
    footer_baseline_y = bottom - 112 + rng.randint(-18, 18)
    footer_width = right - m_left
    date_right = m_left + int(footer_width * rng.uniform(0.24, 0.31))
    stamp_right = right - rng.randint(12, 30)
    stamp_left = stamp_right - max(170, int(footer_width * rng.uniform(0.16, 0.22)))
    signature_left = date_right + 24
    signature_right = stamp_left - 24
    zones = [
        zone(
            ZoneConfig(
                "title",
                "title",
                (m_left, m_top, right, m_top + title_height),
                schema.title,
                language,
                1,
                style("title", rng, language),
            )
        ),
        zone(
            ZoneConfig(
                "fields",
                "form",
                (
                    m_left + form_inset_left,
                    m_top + fields_top,
                    right - form_inset_right,
                    bottom - footer_space,
                ),
                "\n".join(lines),
                language,
                2,
                style("body", rng, language),
            ),
            role="form_fields",
            content_schema_id=schema.schema_id,
            field_types=field_types,
            layout_density=density,
            date_roles={"date": "document_date"}
            if schema.schema_id == "document_registration"
            else {},
        ),
        zone(
            ZoneConfig(
                "date",
                "metadata",
                (m_left, footer_baseline_y - 48, date_right, footer_baseline_y + 8),
                date_text,
                language,
                3,
                style("metadata", rng, language),
            ),
            role=date_role,
            date_role=date_role,
            footer_baseline_y=footer_baseline_y,
        ),
        zone(
            ZoneConfig(
                "signature",
                "metadata",
                (signature_left, bottom - 190, signature_right, bottom - 70),
                context.person_name,
                language,
                4,
                style("metadata", rng, language),
            ),
            role="signature_zone",
            signature_role="applicant_signature",
            footer_baseline_y=footer_baseline_y,
        ),
        zone(
            ZoneConfig(
                "stamp_safe",
                "stamp",
                (stamp_left, bottom - 285, stamp_right, bottom - 45),
                "",
                language,
                5,
                style("note", rng, language),
            ),
            role="stamp_zone",
            safe_overlay=True,
        ),
    ]
    for item in zones:
        item.metadata.setdefault("layout_density", density)
    return zones


def table(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    layout_id: str | None = None,
    variant_id: str | None = None,
) -> list[Zone]:
    (m_left, m_top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    schema = _select_schema(table_schemas(language), layout_id, rng)
    from .variants import get_variant_properties

    props = get_variant_properties("table", variant_id) if variant_id else {}
    density = props.get("density", choose_density(rng))
    if variant_id and "rows_count" in props:
        data_rows = props["rows_count"]
    else:
        data_rows = rng.randint(*_row_ranges(layout_id)[density])
    cols = len(schema.columns)
    (table_top, table_footer_space, metadata_height) = TABLE_LAYOUT_GEOMETRY.get(
        layout_id or "", (250, 210, 60)
    )
    top_offset, footer_offset, inset_offset, metadata_offset = _schedule_table_offsets(
        layout_id, variant_id
    )
    table_top += top_offset
    table_footer_space = max(150, table_footer_space + footer_offset)
    metadata_height = max(45, metadata_height + metadata_offset)
    table_inset = 12 + inset_offset
    table_bbox = (
        m_left + table_inset,
        m_top + table_top,
        right - table_inset,
        bottom - table_footer_space,
    )
    data_rows = min(
        data_rows, _max_data_rows_for_table(table_bbox, len(schema.columns), layout_id)
    )
    rows = data_rows + 1
    table_zone = zone(
        ZoneConfig(
            "table", "table", table_bbox, "", language, 4, style("table", rng, language)
        ),
        rows=rows,
        cols=cols,
        role="typed_table",
        content_schema_id=schema.schema_id,
        layout_density=density,
        **_table_visual_metadata(layout_id, variant_id, props),
        column_specs=[
            {
                "key": column.key,
                "label": column.label,
                "value_type": column.value_type,
                "weight": column.weight,
                "align": column.align,
            }
            for column in schema.columns
        ],
    )
    table_zone.cells = _build_table_cells(
        StructuredTableConfig(
            schema=schema,
            context=context,
            language=language,
            rng=rng,
            table_bbox=table_bbox,
            rows=rows,
        )
    )
    has_page_number = bool(props.get("has_page_num"))
    note_right = right - 260 if has_page_number else right
    title_style = style("title", rng, language)
    if layout_id == "simple_table_page":
        title_style.font_size_px = min(title_style.font_size_px, 34)
    zones = [
        zone(
            ZoneConfig(
                "organization",
                "metadata",
                (m_left, m_top, right, m_top + 55),
                context.organization,
                language,
                1,
                style("metadata", rng, language),
            ),
            role="agency_header",
            layout_density=density,
        ),
        zone(
            ZoneConfig(
                "title",
                "title",
                (
                    m_left,
                    m_top + 65,
                    right,
                    m_top + (165 if layout_id == "simple_table_page" else 145),
                ),
                _table_title(schema.title, layout_id, index),
                language,
                2,
                title_style,
            )
        ),
        zone(
            ZoneConfig(
                "table_metadata",
                "metadata",
                (
                    m_left,
                    m_top + table_top - metadata_height - 15,
                    right,
                    m_top + table_top - 15,
                ),
                f"{context.document_number}    {bilingual(language, 'Есептік кезең', 'Отчеттук мезгил', 'Отчетный период')}: {context.period}",
                language,
                3,
                style("metadata", rng, language),
            ),
            role="metadata_block",
            date_roles=["period_start", "period_end"],
        ),
        table_zone,
        zone(
            ZoneConfig(
                "note",
                "metadata",
                (m_left, bottom - table_footer_space + 35, note_right, bottom - 35),
                f"{bilingual(language, 'Барлығы', 'Жалпы', 'Итого')}: {data_rows}    {bilingual(language, 'Бекітті', 'Бекитти', 'Утвердил')}: {context.person_name}",
                language,
                5,
                style("note", rng, language),
            ),
            role="table_footer",
        ),
    ]
    if has_page_number:
        zones.append(
            zone(
                ZoneConfig(
                    "page_number",
                    "metadata",
                    (right - 220, bottom - table_footer_space + 35, right, bottom - 35),
                    f"{bilingual(language, 'Бет', 'Бет', 'Стр.')} {index % 97 + 1}",
                    language,
                    6,
                    style("note", rng, language),
                ),
                role="page_number",
                layout_density=density,
            )
        )
    return zones


def bulletin(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    (m_left, m_top, right, bottom) = bounds
    record = sample_seed_record(
        language, rng, layout_id="bulletin_or_newspaper_page", domain="bulletin"
    )
    lead_text = record.text if record else text(language, rng, 900, 1500)
    corpus_meta = seed_record_metadata(record)
    context = build_document_context(language, index, rng)
    from .variants import get_variant_properties

    props = get_variant_properties("structured", variant_id) if variant_id else {}
    density = props.get("density", choose_density(rng))
    column_ranges = {
        "standard": (1450, 1850),
        "dense": (1700, 2150),
        "extended": (1950, 2450),
    }
    gutter = 34
    mid = (m_left + right) // 2
    masthead_style = style("title", rng, language)
    masthead_style.font_size_px = min(masthead_style.font_size_px, 36)
    zones = [
        zone(
            ZoneConfig(
                "masthead",
                "title",
                (m_left, m_top, right, m_top + 90),
                bilingual(
                    language,
                    "АҚПАРАТТЫҚ ХАБАРШЫ",
                    "МААЛЫМАТ БЮЛЛЕТЕНИ",
                    "ИНФОРМАЦИОННЫЙ БЮЛЛЕТЕНЬ",
                ),
                language,
                1,
                masthead_style,
            ),
            role="agency_header",
            **corpus_meta,
        ),
        zone(
            ZoneConfig(
                "issue_metadata",
                "metadata",
                (m_left, m_top + 105, right, m_top + 155),
                f"№ {200 + index}    {context.date}    {context.organization}",
                language,
                2,
                style("metadata", rng, language),
            ),
            role="metadata_block",
            **corpus_meta,
        ),
        zone(
            ZoneConfig(
                "masthead_rule",
                "decorative_non_text",
                (m_left, m_top + 96, right, m_top + 98),
                "",
                language,
                90,
                style("note", rng, language),
            ),
            role="layout_separator",
            orientation="horizontal",
            stroke_width=1,
            color=(72, 72, 72),
        ),
        zone(
            ZoneConfig(
                "issue_rule",
                "decorative_non_text",
                (m_left, m_top + 166, right, m_top + 169),
                "",
                language,
                91,
                style("note", rng, language),
            ),
            role="layout_separator",
            orientation="horizontal",
            stroke_width=2,
            color=(58, 58, 58),
        ),
        zone(
            ZoneConfig(
                "lead_story",
                "body",
                (m_left, m_top + 190, right, m_top + 520),
                f"{context.subject.upper()}\n\n{lead_text}",
                language,
                3,
                style("body", rng, language),
            ),
            role="body",
            min_render_font_px=16,
            **corpus_meta,
        ),
        zone(
            ZoneConfig(
                "column_left",
                "body",
                (m_left, m_top + 560, mid - gutter, bottom - 140),
                text(language, rng, *column_ranges[density]),
                language,
                4,
                style("body", rng, language),
            ),
            role="body",
            min_render_font_px=16,
            **corpus_meta,
        ),
        zone(
            ZoneConfig(
                "column_right",
                "body",
                (mid + gutter, m_top + 560, right, bottom - 140),
                text(language, rng, *column_ranges[density]),
                language,
                5,
                style("body", rng, language),
            ),
            role="body",
            min_render_font_px=16,
            **corpus_meta,
        ),
        zone(
            ZoneConfig(
                "footer_note",
                "metadata",
                (m_left, bottom - 105, right, bottom - 60),
                f"{context.department}    {context.phone}    {context.email}",
                language,
                6,
                style("note", rng, language),
            ),
            role="footer",
            **corpus_meta,
        ),
    ]
    for item in zones:
        item.metadata.setdefault("layout_density", density)
    return zones
