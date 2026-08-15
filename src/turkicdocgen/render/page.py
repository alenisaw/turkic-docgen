from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from turkicdocgen.schema import LineBox, PagePlan, Zone

if TYPE_CHECKING:
    from pathlib import Path

    from turkicdocgen.schema import TableCell


@dataclass(frozen=True, slots=True)
class AlignmentBounds:
    left: int
    right: int
    align: str
    padding: int = 0


PAPER_BASES = (
    ("bright_white", 45, (255, 255, 254)),
    ("neutral_white", 30, (250, 250, 249)),
    ("cool_white", 12, (247, 249, 251)),
    ("light_ivory", 10, (251, 249, 241)),
    ("recycled_gray", 3, (244, 244, 241)),
)
TABLE_MIN_FONT_SIZE_PX = 18
TABLE_MAX_FONT_SIZE_PX = 24
TABLE_HORIZONTAL_PADDING_PX = 7
TABLE_VERTICAL_PADDING_PX = 4
FORM_MIN_FONT_SIZE_PX = 18
FORM_HORIZONTAL_PADDING_PX = 10
FORM_VERTICAL_PADDING_PX = 5


def _paper_base(page_id: str) -> tuple[str, tuple[int, int, int]]:
    bucket = int(hashlib.sha256(page_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    cumulative = 0
    for name, weight, color in PAPER_BASES:
        cumulative += weight
        if bucket < cumulative:
            return name, color
    return PAPER_BASES[-1][0], PAPER_BASES[-1][2]


_glyph_measurement_time = 0.0


def get_glyph_measurement_time() -> float:
    return _glyph_measurement_time


def reset_glyph_measurement_time() -> float:
    global _glyph_measurement_time
    t = _glyph_measurement_time
    _glyph_measurement_time = 0.0
    return t


@lru_cache(maxsize=16384)
def _cached_textlength(
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str
) -> float:
    t0 = time.perf_counter()
    if hasattr(font, "getlength"):
        res = font.getlength(text)
    else:
        res = len(text) * 12.0
    global _glyph_measurement_time
    _glyph_measurement_time += time.perf_counter() - t0
    return res


@lru_cache(maxsize=16384)
def _cached_getbbox(
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str
) -> tuple[int, int, int, int]:
    t0 = time.perf_counter()
    if hasattr(font, "getbbox"):
        res = font.getbbox(text, anchor="lt")
    else:
        res = (0, 0, len(text) * 12, 16)
    global _glyph_measurement_time
    _glyph_measurement_time += time.perf_counter() - t0
    return res


def _measured_text_width(font: ImageFont.ImageFont, text: str) -> int:
    left, _, right, _ = _cached_getbbox(font, text)
    return max(int(_cached_textlength(font, text)), right - left)


@lru_cache(maxsize=64)
def _font(
    size: int, bold: bool = False, font_path: str | None = None
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current: list[str] = []
        for word in words:
            if _measured_text_width(font, word) > width:
                if current:
                    lines.append(" ".join(current))
                    current = []
                fragment = ""
                for character in word:
                    candidate = f"{fragment}{character}"
                    if fragment and _measured_text_width(font, candidate) > width:
                        lines.append(fragment)
                        fragment = character
                    else:
                        fragment = candidate
                if fragment:
                    current = [fragment]
                continue
            candidate = " ".join([*current, word])
            if _measured_text_width(font, candidate) <= width or not current:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
    return lines


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = _cached_getbbox(font, text)
    x, y = position
    return int(x + left), int(y + top), int(x + right), int(y + bottom)


def _union_bbox(
    boxes: list[tuple[int, int, int, int]],
    fallback: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if not boxes:
        return fallback
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _contain_text_position(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    bounds: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    x, y = position
    left, top, right, bottom = _text_bbox(draw, position, text, font)
    x += max(0, bounds[0] - left)
    x -= max(0, right - bounds[2])
    y += max(0, bounds[1] - top)
    y -= max(0, bottom - bounds[3])
    adjusted = (x, y)
    return adjusted, _text_bbox(draw, adjusted, text, font)


def _aligned_text_x(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    bounds: AlignmentBounds,
) -> tuple[int, int]:
    left = bounds.left
    right = bounds.right
    align = bounds.align
    padding = bounds.padding
    text_width = int(_cached_textlength(font, text))
    if align == "right":
        x = right - padding - text_width
    elif align == "center":
        x = left + ((right - left) - text_width) // 2
    else:
        x = left + padding
    return max(left + padding, x), text_width


def _cell_lines(
    draw: ImageDraw.ImageDraw,
    cell: TableCell,
    font: ImageFont.ImageFont,
) -> list[str]:
    width = cell.bbox[2] - cell.bbox[0]
    available_width = max(8, width - 2 * TABLE_HORIZONTAL_PADDING_PX)
    return _wrap_text(draw, cell.text, font, available_width)


def _line_box(
    zone: Zone,
    index: int,
    bbox: tuple[int, int, int, int],
    text: str,
) -> LineBox:
    x1, y1, x2, y2 = bbox
    return LineBox(
        f"{zone.zone_id}_line_{index:03d}",
        bbox,
        text,
        zone.reading_order * 1000 + index,
        [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
    )


def _draw_decorative_zone(draw: ImageDraw.ImageDraw, zone: Zone) -> None:
    x1, y1, x2, y2 = zone.bbox
    orientation = zone.metadata.get("orientation", "horizontal")
    color = tuple(zone.metadata.get("color", (96, 96, 96)))
    width = int(zone.metadata.get("stroke_width", 1))
    if orientation == "vertical":
        x = (x1 + x2) // 2
        draw.line((x, y1, x, y2), fill=color, width=width)
    else:
        y = (y1 + y2) // 2
        draw.line((x1, y, x2, y), fill=color, width=width)


def _axis_positions(start: int, sizes: list[int]) -> list[int]:
    positions = [start]
    for size in sizes:
        positions.append(positions[-1] + size)
    return positions


def _weighted_axis_sizes(
    total: int, weights: list[float], minimums: list[int]
) -> list[int]:
    minimum_total = sum(minimums)
    if minimum_total >= total:
        scaled = [max(1, int(total * value / minimum_total)) for value in minimums]
        scaled[-1] += total - sum(scaled)
        return scaled
    remaining = total - minimum_total
    weight_total = sum(weights) or float(len(weights))
    sizes = [
        minimum + int(remaining * weight / weight_total)
        for minimum, weight in zip(minimums, weights, strict=True)
    ]
    sizes[-1] += total - sum(sizes)
    return sizes


def _table_column_sizes(
    draw: ImageDraw.ImageDraw,
    zone: Zone,
    font: ImageFont.ImageFont,
) -> list[int]:
    x1, _, x2, _ = zone.bbox
    cols = max((cell.col for cell in zone.cells), default=0) + 1
    specs = zone.metadata.get("column_specs") or []
    type_minimums = {
        "sequence": 64,
        "date": 138,
        "doc": 150,
        "amount": 128,
        "score": 104,
        "status": 150,
    }
    weights: list[float] = []
    minimums: list[int] = []
    for col in range(cols):
        column_cells = [cell for cell in zone.cells if cell.col == col]
        spec = specs[col] if col < len(specs) and isinstance(specs[col], dict) else {}
        value_type = str(spec.get("value_type", "text"))
        longest_token = max(
            (
                _cached_textlength(font, token)
                for cell in column_cells
                for token in cell.text.split()
            ),
            default=40.0,
        )
        content_demand = min(330, int(longest_token) + 2 * TABLE_HORIZONTAL_PADDING_PX)
        minimum = max(type_minimums.get(value_type, 145), content_demand)
        minimums.append(minimum)
        source_width = max(
            (cell.bbox[2] - cell.bbox[0] for cell in column_cells),
            default=minimum,
        )
        weights.append(max(float(source_width), float(minimum)))
    return _weighted_axis_sizes(x2 - x1, weights, minimums)


def _table_layout(
    draw: ImageDraw.ImageDraw,
    zone: Zone,
) -> tuple[ImageFont.ImageFont, int, dict[int, list[str]], list[int], list[int]]:
    x1, y1, _, y2 = zone.bbox
    rows = max((cell.row for cell in zone.cells), default=0) + 1
    chosen: (
        tuple[
            ImageFont.ImageFont,
            int,
            dict[int, list[str]],
            list[int],
            list[int],
        ]
        | None
    ) = None
    for size in range(TABLE_MAX_FONT_SIZE_PX, TABLE_MIN_FONT_SIZE_PX - 1, -1):
        font = _font(size, font_path=zone.style.font_path)
        line_height = max(size + 4, int(size * zone.style.line_spacing))
        column_sizes = _table_column_sizes(draw, zone, font)
        column_positions = _axis_positions(x1, column_sizes)
        for cell in zone.cells:
            cell.bbox = (
                column_positions[cell.col],
                y1,
                column_positions[cell.col + 1],
                y1 + 1,
            )
        lines_by_order = {
            cell.reading_order: _cell_lines(draw, cell, font) for cell in zone.cells
        }
        row_sizes = []
        for row in range(rows):
            line_count = max(
                (
                    len(lines_by_order[cell.reading_order])
                    for cell in zone.cells
                    if cell.row == row
                ),
                default=1,
            )
            row_sizes.append(line_count * line_height + 2 * TABLE_VERTICAL_PADDING_PX)
        chosen = (font, line_height, lines_by_order, column_sizes, row_sizes)
        if sum(row_sizes) <= y2 - y1:
            break
    assert chosen is not None
    font, line_height, lines_by_order, column_sizes, row_sizes = chosen
    available_height = y2 - y1
    if sum(row_sizes) < available_height:
        extra_sizes = _weighted_axis_sizes(
            available_height - sum(row_sizes),
            [1.0] * rows,
            [0] * rows,
        )
        row_sizes = [
            size + extra for size, extra in zip(row_sizes, extra_sizes, strict=True)
        ]
    elif sum(row_sizes) > available_height:
        row_sizes = _weighted_axis_sizes(
            available_height,
            [float(size) for size in row_sizes],
            [max(1, min(size, line_height)) for size in row_sizes],
        )
    return font, line_height, lines_by_order, column_sizes, row_sizes


def _draw_table_zone(draw: ImageDraw.ImageDraw, zone: Zone) -> None:
    x1, y1, x2, y2 = zone.bbox
    font, line_height, lines_by_order, column_sizes, row_sizes = _table_layout(
        draw, zone
    )
    column_positions = _axis_positions(x1, column_sizes)
    row_positions = _axis_positions(y1, row_sizes)
    header_fill = zone.metadata.get("header_fill")
    if header_fill and len(row_positions) > 1:
        draw.rectangle(
            (x1, row_positions[0], x2, row_positions[1]), fill=tuple(header_fill)
        )
    band_fill = zone.metadata.get("row_band_fill")
    band_period = int(zone.metadata.get("row_band_period", 0) or 0)
    if band_fill and band_period > 0:
        for row in range(1, len(row_positions) - 1):
            if row % band_period == 0:
                draw.rectangle(
                    (x1, row_positions[row], x2, row_positions[row + 1]),
                    fill=tuple(band_fill),
                )
    grid_color = tuple(zone.metadata.get("grid_color", (40, 40, 40)))
    grid_width = int(zone.metadata.get("grid_line_width", 1))
    outer_width = int(zone.metadata.get("outer_border_width", grid_width))
    for index, y in enumerate(row_positions):
        width = outer_width if index in {0, len(row_positions) - 1} else grid_width
        draw.line((x1, y, x2, y), fill=grid_color, width=width)
    for index, x in enumerate(column_positions):
        width = outer_width if index in {0, len(column_positions) - 1} else grid_width
        draw.line((x, y1, x, y2), fill=grid_color, width=width)

    wrapped = 0
    truncated = 0
    for cell in zone.cells:
        cell.bbox = (
            column_positions[cell.col],
            row_positions[cell.row],
            column_positions[cell.col + 1],
            row_positions[cell.row + 1],
        )
        cx1, cy1, cx2, cy2 = cell.bbox
        cell.polygon = [(cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)]
        lines = lines_by_order[cell.reading_order]
        if len(lines) > max(1, len(cell.text.splitlines())):
            wrapped += 1
        text_height = len(lines) * line_height
        available_height = cy2 - cy1 - 2 * TABLE_VERTICAL_PADDING_PX
        rendered_complete = text_height <= available_height and "".join(
            cell.text.split()
        ) == "".join("".join(lines).split())
        if not rendered_complete:
            truncated += 1
        text_y = cy1 + max(
            TABLE_VERTICAL_PADDING_PX,
            ((cy2 - cy1) - text_height) // 2,
        )
        line_boxes: list[tuple[int, int, int, int]] = []
        rendered_lines_actual = []
        line_boxes_actual = []
        for index, line in enumerate(lines):
            line_y = text_y + index * line_height
            text_x, _ = _aligned_text_x(
                draw,
                line,
                font,
                AlignmentBounds(
                    cx1,
                    cx2,
                    cell.metadata.get("align", "left"),
                    TABLE_HORIZONTAL_PADDING_PX,
                ),
            )
            position, bbox = _contain_text_position(
                draw,
                (text_x, line_y),
                line,
                font,
                (
                    cx1 + TABLE_HORIZONTAL_PADDING_PX,
                    cy1 + TABLE_VERTICAL_PADDING_PX,
                    cx2 - TABLE_HORIZONTAL_PADDING_PX,
                    cy2 - TABLE_VERTICAL_PADDING_PX,
                ),
            )
            line_boxes.append(bbox)
            if position[1] + line_height <= cy2 - TABLE_VERTICAL_PADDING_PX + 1:
                draw.text(
                    position,
                    line,
                    fill=(20, 20, 20),
                    font=font,
                    anchor="lt",
                )
                rendered_lines_actual.append(line)
                line_boxes_actual.append(bbox)
        rendered_bbox = _union_bbox(line_boxes, (cx1, cy1, cx1, cy1))
        rendered_bbox_actual = _union_bbox(line_boxes_actual, (cx1, cy1, cx1, cy1))
        inside_cell = (
            cx1 <= rendered_bbox[0] <= rendered_bbox[2] <= cx2
            and cy1 <= rendered_bbox[1] <= rendered_bbox[3] <= cy2
        )
        cell.metadata.update(
            {
                "rendered_lines": list(lines),
                "rendered_font_size": int(
                    getattr(font, "size", zone.style.font_size_px)
                ),
                "rendered_line_height": line_height,
                "rendered_text_y": text_y,
                "rendered_bbox": list(rendered_bbox_actual),
                "rendered_complete": rendered_complete,
                "rendered_inside_cell": inside_cell,
                "wrapped": len(lines) > max(1, len(cell.text.splitlines())),
                "planned_text": cell.text,
                "rendered_text": "\n".join(rendered_lines_actual),
                "font_family": zone.style.font_family,
                "font_size": int(getattr(font, "size", zone.style.font_size_px)),
                "line_count": len(rendered_lines_actual),
                "wrap_state": len(lines) > max(1, len(cell.text.splitlines())),
                "completion_state": rendered_complete,
            }
        )
    zone.style.font_size_px = int(getattr(font, "size", TABLE_MIN_FONT_SIZE_PX))
    zone.metadata.update(
        {
            "rendered_cell_count": len(zone.cells),
            "fitted_cell_count": wrapped,
            "wrapped_cell_count": wrapped,
            "truncated_cell_count": truncated,
            "rendered_font_size": zone.style.font_size_px,
            "text_truncated": truncated > 0,
            "planned_text": zone.text,
            "rendered_text": " | ".join(cell.text for cell in zone.cells),
            "rendered_bbox": list(
                _union_bbox(
                    [
                        c.metadata["rendered_bbox"]
                        for c in zone.cells
                        if c.metadata.get("rendered_bbox")
                    ],
                    zone.bbox,
                )
            ),
            "font_family": zone.style.font_family,
            "font_size": zone.style.font_size_px,
            "line_count": sum(c.metadata.get("line_count", 0) for c in zone.cells),
            "wrap_state": any(c.metadata.get("wrap_state") for c in zone.cells),
            "completion_state": truncated == 0,
        }
    )


def _form_layout(
    draw: ImageDraw.ImageDraw,
    zone: Zone,
) -> tuple[
    ImageFont.ImageFont,
    int,
    int,
    list[tuple[str, list[str], list[str], int]],
]:
    x1, y1, x2, y2 = zone.bbox
    source_lines = zone.text.splitlines()
    preferred_size = max(FORM_MIN_FONT_SIZE_PX, zone.style.font_size_px)
    chosen: (
        tuple[
            ImageFont.ImageFont,
            int,
            int,
            list[tuple[str, list[str], list[str], int]],
        ]
        | None
    ) = None
    for size in range(preferred_size, FORM_MIN_FONT_SIZE_PX - 1, -1):
        font = _font(size, font_path=zone.style.font_path)
        line_height = max(size + 4, int(size * zone.style.line_spacing))
        labels = [
            line.partition(":")[0]
            for line in source_lines
            if not (line.startswith("[") and line.endswith("]"))
        ]
        preferred_label_width = max(
            (
                _measured_text_width(font, label) + 3 * FORM_HORIZONTAL_PADDING_PX
                for label in labels
            ),
            default=int((x2 - x1) * 0.38),
        )
        label_width = min(
            int((x2 - x1) * 0.62),
            max(int((x2 - x1) * 0.38), preferred_label_width),
        )
        entries: list[tuple[str, list[str], list[str], int]] = []
        for line in source_lines:
            if line.startswith("[") and line.endswith("]"):
                entries.append(
                    (
                        "section",
                        _wrap_text(
                            draw,
                            line[1:-1],
                            font,
                            x2 - x1 - 2 * FORM_HORIZONTAL_PADDING_PX,
                        ),
                        [],
                        line_height + 2 * FORM_VERTICAL_PADDING_PX,
                    )
                )
                continue
            label, _, value = line.partition(":")
            label_lines = _wrap_text(
                draw,
                label,
                font,
                label_width - 2 * FORM_HORIZONTAL_PADDING_PX,
            )
            value_lines = _wrap_text(
                draw,
                value.strip(),
                font,
                x2 - x1 - label_width - 2 * FORM_HORIZONTAL_PADDING_PX,
            )
            row_height = (
                max(len(label_lines), len(value_lines), 1) * line_height
                + 2 * FORM_VERTICAL_PADDING_PX
            )
            entries.append(("field", label_lines, value_lines, row_height))
        chosen = (font, line_height, label_width, entries)
        if sum(entry[3] for entry in entries) <= y2 - y1:
            break
    assert chosen is not None
    return chosen


def _draw_form_zone(draw: ImageDraw.ImageDraw, zone: Zone) -> None:
    x1, y1, x2, y2 = zone.bbox
    font, line_height, label_width, entries = _form_layout(draw, zone)
    required_height = sum(entry[3] for entry in entries)
    if required_height < y2 - y1:
        extras = _weighted_axis_sizes(
            y2 - y1 - required_height,
            [0.35 if entry[0] == "section" else 1.0 for entry in entries],
            [0] * len(entries),
        )
    else:
        extras = [0] * len(entries)
    source_lines = zone.text.splitlines()
    rendered_fields: list[dict[str, object]] = []
    field_types = zone.metadata.get("field_types", [])
    field_idx = 0
    font_size = getattr(font, "size", zone.style.font_size_px)
    y = y1
    rendered_count = 0
    for index, (entry_type, label_lines, value_lines, base_height) in enumerate(
        entries
    ):
        row_height = base_height + extras[index]
        row_bottom = min(y2, y + row_height)
        source_line = source_lines[index]
        if entry_type == "section":
            draw.rectangle(
                (x1, y, x2, row_bottom),
                fill=(232, 234, 236),
                outline=(80, 80, 80),
            )
            text_boxes = []
            for line_index, line in enumerate(label_lines):
                line_y = y + FORM_VERTICAL_PADDING_PX + line_index * line_height
                if line_y + line_height > row_bottom:
                    break
                draw.text(
                    (x1 + FORM_HORIZONTAL_PADDING_PX, line_y),
                    line,
                    fill=(20, 20, 20),
                    font=font,
                    anchor="lt",
                )
                text_boxes.append(
                    _text_bbox(
                        draw,
                        (x1 + FORM_HORIZONTAL_PADDING_PX, line_y),
                        line,
                        font,
                    )
                )
            current_section = source_line[1:-1]
            label_bbox = _union_bbox(text_boxes, (x1, y, x1, y))
            rendered_fields.append(
                {
                    "kind": "section",
                    "field_key": "",
                    "label_text": current_section,
                    "value_text": "",
                    "label_bbox": list(label_bbox),
                    "value_bbox": None,
                    "row_bbox": [x1, y, x2, row_bottom],
                    "section_id": current_section,
                    "rendered_complete": len(text_boxes) == len(label_lines),
                    "font_size": font_size,
                    "wrap_count": 0,
                    "planned_text": source_line,
                    "rendered_text": "\n".join(label_lines[: len(text_boxes)]),
                    "rendered_bbox": list(label_bbox),
                    "font_family": zone.style.font_family,
                    "line_count": len(text_boxes),
                    "wrap_state": len(label_lines) > 1,
                    "completion_state": len(text_boxes) == len(label_lines),
                }
            )
            rendered_count += int(len(text_boxes) == len(label_lines))
            y = row_bottom
            continue
        draw.rectangle((x1, y, x2, row_bottom), outline=(80, 80, 80))
        draw.line(
            (x1 + label_width, y, x1 + label_width, row_bottom),
            fill=(80, 80, 80),
        )
        label, _, value = source_line.partition(":")
        label_text = label
        value_text = value.strip()
        field_key = ""
        section_id = ""
        if field_idx < len(field_types):
            field_key = field_types[field_idx].get("key", "")
            section_id = field_types[field_idx].get("section", "")
            field_idx += 1

        label_boxes = []
        value_boxes = []
        for line_index, line in enumerate(label_lines):
            line_y = y + FORM_VERTICAL_PADDING_PX + line_index * line_height
            if line_y + line_height > row_bottom:
                break
            position = (x1 + FORM_HORIZONTAL_PADDING_PX, line_y)
            draw.text(position, line, fill=(20, 20, 20), font=font, anchor="lt")
            label_boxes.append(_text_bbox(draw, position, line, font))
        for line_index, line in enumerate(value_lines):
            line_y = y + FORM_VERTICAL_PADDING_PX + line_index * line_height
            if line_y + line_height > row_bottom:
                break
            position = (x1 + label_width + FORM_HORIZONTAL_PADDING_PX, line_y)
            draw.text(position, line, fill=(20, 20, 20), font=font, anchor="lt")
            value_boxes.append(_text_bbox(draw, position, line, font))
        label_bbox = _union_bbox(label_boxes, (x1, y, x1, y))
        value_bbox = _union_bbox(
            value_boxes,
            (x1 + label_width, y, x1 + label_width, y),
        )
        complete = (
            len(label_boxes) == len(label_lines)
            and len(value_boxes) == len(value_lines)
            and label_bbox[2] <= x1 + label_width
            and value_bbox[0] >= x1 + label_width
            and value_bbox[2] <= x2
        )
        rendered_fields.append(
            {
                "kind": "field",
                "field_key": field_key,
                "label_text": label_text,
                "value_text": value_text,
                "label_bbox": list(label_bbox),
                "value_bbox": list(value_bbox),
                "row_bbox": [x1, y, x2, row_bottom],
                "section_id": section_id,
                "rendered_complete": complete,
                "font_size": font_size,
                "wrap_count": len(value_lines),
                "planned_text": source_line,
                "rendered_text": "\n".join(label_lines[: len(label_boxes)])
                + ":"
                + "\n".join(value_lines[: len(value_boxes)]),
                "rendered_bbox": list(
                    _union_bbox(label_boxes + value_boxes, (x1, y, x1, y))
                ),
                "font_family": zone.style.font_family,
                "line_count": len(label_boxes) + len(value_boxes),
                "wrap_state": len(label_lines) > 1 or len(value_lines) > 1,
                "completion_state": complete,
            }
        )
        rendered_count += int(complete)
        zone.lines.append(_line_box(zone, index, (x1, y, x2, row_bottom), source_line))
        y = row_bottom
    zone.style.font_size_px = int(font_size)
    zone.metadata.update(
        {
            "expected_line_count": len(entries),
            "rendered_line_count": len(zone.lines),
            "rendered_entry_count": rendered_count,
            "rendered_fields": rendered_fields,
            "form_label_width": label_width,
            "text_truncated": rendered_count < len(entries),
            "planned_text": zone.text,
            "rendered_text": "\n".join(f["rendered_text"] for f in rendered_fields),
            "rendered_bbox": list(
                _union_bbox(
                    [
                        f["rendered_bbox"]
                        for f in rendered_fields
                        if f.get("rendered_bbox")
                    ],
                    zone.bbox,
                )
            ),
            "font_family": zone.style.font_family,
            "font_size": zone.style.font_size_px,
            "line_count": sum(f.get("line_count", 0) for f in rendered_fields),
            "wrap_state": any(f.get("wrap_state") for f in rendered_fields),
            "completion_state": rendered_count == len(entries),
        }
    )


def _draw_signature_zone(
    draw: ImageDraw.ImageDraw,
    zone: Zone,
    font: ImageFont.ImageFont,
    line_height: int,
) -> None:
    x1, y1, x2, y2 = zone.bbox
    line_x1 = x1 + 8
    line_x2 = x1 + max(120, int((x2 - x1) * 0.52))
    line_y = min(
        y2 - 14,
        int(zone.metadata.get("footer_baseline_y", y1 + (y2 - y1) * 0.70)),
    )
    draw.line((line_x1, line_y, line_x2, line_y), fill=(60, 60, 60))
    mark_top = y1 + 3
    zone.metadata["signature_mark_bbox"] = [
        line_x1,
        mark_top,
        line_x2,
        max(mark_top + 28, line_y - 3),
    ]
    zone.metadata["signature_line"] = [line_x1, line_y, line_x2, line_y]

    has_planned_signature_rule = "_" in zone.text
    text_x = x1 if has_planned_signature_rule else min(x2 - 1, line_x2 + 24)
    lines = _wrap_text(draw, zone.text, font, max(1, x2 - text_x))
    blank_spacing = max(4, line_height // 4)
    total_text_height = sum(
        blank_spacing if not line.strip() else line_height for line in lines
    )
    y = max(y1, min(line_y - line_height + 2, y2 - total_text_height))
    rendered_slots = 0
    for index, line in enumerate(lines):
        slot_height = blank_spacing if not line.strip() else line_height
        if y + slot_height > y2:
            break
        rendered_slots += 1
        if not line.strip():
            y += blank_spacing
            continue
        draw.text((text_x, y), line, fill=(18, 18, 18), font=font)
        right = min(x2, text_x + max(int(draw.textlength(line, font=font)), 1))
        zone.lines.append(
            _line_box(zone, index, (text_x, y, right, y + line_height), line)
        )
        y += line_height
    zone.metadata.update(
        {
            "expected_line_count": len(lines),
            "rendered_line_count": rendered_slots,
            "text_truncated": rendered_slots < len(lines),
            "planned_text": zone.text,
            "rendered_text": "\n".join(lines[:rendered_slots]),
            "rendered_bbox": list(
                _union_bbox([line.bbox for line in zone.lines], (x1, y1, x1, y1))
            ),
            "font_family": zone.style.font_family,
            "font_size": zone.style.font_size_px,
            "line_count": len(zone.lines),
            "wrap_state": len(lines) > max(1, len(zone.text.splitlines())),
            "completion_state": rendered_slots >= len(lines),
        }
    )


def _draw_footer_zone(
    draw: ImageDraw.ImageDraw,
    zone: Zone,
    font: ImageFont.ImageFont,
    line_height: int,
) -> None:
    x1, y1, x2, _ = zone.bbox
    lines = _wrap_text(draw, zone.text, font, x2 - x1)
    y = max(y1, int(zone.metadata["footer_baseline_y"]) - line_height + 2)
    if lines:
        line = lines[0]
        if line.strip():
            draw.text((x1, y), line, fill=(18, 18, 18), font=font)
            right = min(x2, x1 + max(int(draw.textlength(line, font=font)), 1))
            zone.lines.append(_line_box(zone, 0, (x1, y, right, y + line_height), line))
    zone.metadata.update(
        {
            "expected_line_count": len(lines),
            "rendered_line_count": len(zone.lines),
            "text_truncated": len(lines) > 1,
            "planned_text": zone.text,
            "rendered_text": "\n".join(lines[: len(zone.lines)]),
            "rendered_bbox": list(
                _union_bbox([line.bbox for line in zone.lines], (x1, y1, x1, y1))
            ),
            "font_family": zone.style.font_family,
            "font_size": zone.style.font_size_px,
            "line_count": len(zone.lines),
            "wrap_state": len(lines) > max(1, len(zone.text.splitlines())),
            "completion_state": len(lines) <= 1,
        }
    )


def _fit_text_zone_font(
    draw: ImageDraw.ImageDraw, zone: Zone
) -> tuple[ImageFont.ImageFont, int]:
    x1, y1, x2, y2 = zone.bbox
    minimum_size = {
        "title": 32,
        "subtitle": 18,
        "metadata": 14,
        "body": 16,
        "paragraph": 16,
    }.get(zone.zone_type, 14)
    minimum_size = max(
        minimum_size, int(zone.metadata.get("min_render_font_px", 0) or 0)
    )
    font = _font(zone.style.font_size_px, zone.style.bold, zone.style.font_path)
    line_height = max(
        zone.style.font_size_px + 4,
        int(zone.style.font_size_px * zone.style.line_spacing),
    )
    if zone.text.strip():
        for candidate_size in range(zone.style.font_size_px, minimum_size - 1, -1):
            candidate_font = _font(
                candidate_size, zone.style.bold, zone.style.font_path
            )
            candidate_line_height = max(
                candidate_size + 4,
                int(candidate_size * zone.style.line_spacing),
            )
            candidate_lines = _wrap_text(draw, zone.text, candidate_font, x2 - x1)
            if len(candidate_lines) * candidate_line_height <= y2 - y1:
                font = candidate_font
                line_height = candidate_line_height
                zone.style.font_size_px = candidate_size
                break
    return font, line_height


def _draw_text_zone(draw: ImageDraw.ImageDraw, zone: Zone) -> None:
    x1, y1, x2, y2 = zone.bbox
    if not zone.text.strip():
        zone.metadata.update(
            {
                "expected_line_count": 0,
                "rendered_line_count": 0,
                "rendered_line_slots": 0,
                "text_truncated": False,
                "planned_text": zone.text,
                "rendered_text": "",
                "rendered_bbox": None,
                "font_family": zone.style.font_family,
                "font_size": zone.style.font_size_px,
                "line_count": 0,
                "wrap_state": False,
                "completion_state": True,
            }
        )
        if zone.zone_type in {"body", "paragraph"}:
            zone.metadata["rendered_fill_ratio"] = 0.0
        return

    font, line_height = _fit_text_zone_font(draw, zone)
    y = y1
    lines = _wrap_text(draw, zone.text, font, x2 - x1)
    rendered_slots = 0
    for index, line in enumerate(lines):
        if y + line_height > y2:
            break
        rendered_slots += 1
        if not line.strip():
            y += line_height
            continue
        text_x, text_w = _aligned_text_x(
            draw, line, font, AlignmentBounds(x1, x2, zone.style.align)
        )
        draw.text((text_x, y), line, fill=(18, 18, 18), font=font)
        line_right = min(x2, text_x + max(text_w, 1))
        zone.lines.append(
            _line_box(zone, index, (text_x, y, line_right, y + line_height), line)
        )
        y += line_height
    zone.metadata["expected_line_count"] = len(lines)
    zone.metadata["rendered_line_count"] = len(zone.lines)
    zone.metadata["rendered_line_slots"] = rendered_slots
    zone.metadata["text_truncated"] = rendered_slots < len(lines)
    zone.metadata.update(
        {
            "planned_text": zone.text,
            "rendered_text": "\n".join(lines[:rendered_slots]),
            "rendered_bbox": list(
                _union_bbox([line.bbox for line in zone.lines], (x1, y1, x1, y1))
            ),
            "font_family": zone.style.font_family,
            "font_size": zone.style.font_size_px,
            "line_count": len(zone.lines),
            "wrap_state": len(lines) > max(1, len(zone.text.splitlines())),
            "completion_state": rendered_slots >= len(lines),
        }
    )
    if zone.zone_type in {"body", "paragraph"}:
        used_height = len(zone.lines) * line_height
        zone.metadata["rendered_fill_ratio"] = round(
            min(1.0, used_height / max(1, y2 - y1)),
            4,
        )


def _draw_zone(draw: ImageDraw.ImageDraw, zone: Zone) -> None:
    if zone.zone_type == "title":
        zone.style.font_size_px = max(32, zone.style.font_size_px)
    zone.lines.clear()
    if zone.zone_type == "stamp":
        return
    if zone.zone_type == "decorative_non_text":
        _draw_decorative_zone(draw, zone)
        return
    if zone.zone_type == "table":
        _draw_table_zone(draw, zone)
        return
    line_height = max(
        zone.style.font_size_px + 4,
        int(zone.style.font_size_px * zone.style.line_spacing),
    )
    if zone.zone_type == "form":
        _draw_form_zone(draw, zone)
        return
    font = _font(zone.style.font_size_px, zone.style.bold, zone.style.font_path)
    if zone.metadata.get("role") == "signature_zone":
        _draw_signature_zone(draw, zone, font, line_height)
        return
    if zone.metadata.get("footer_baseline_y") is not None:
        _draw_footer_zone(draw, zone, font, line_height)
        return
    _draw_text_zone(draw, zone)


def render_plan(plan: PagePlan, image_path: Path) -> Path:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    paper_name, paper_color = _paper_base(plan.page_id)
    image = Image.new("RGB", (plan.width, plan.height), paper_color)
    plan.metadata["paper_base"] = {
        "name": paper_name,
        "rgb": list(paper_color),
    }
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (32, 32, plan.width - 32, plan.height - 32), outline=(228, 226, 218), width=2
    )
    for zone in sorted(plan.zones, key=lambda item: item.reading_order):
        _draw_zone(draw, zone)
    occupied = [
        zone.bbox
        for zone in plan.zones
        if zone.zone_type != "stamp" and zone.text.strip()
    ]
    if occupied:
        content_top = min(box[1] for box in occupied)
        content_bottom = max(box[3] for box in occupied)
        plan.metadata["content_height_ratio"] = round(
            (content_bottom - content_top) / plan.height, 4
        )
    else:
        plan.metadata["content_height_ratio"] = 0.0
    plan.metadata["render_truncated_zones"] = [
        zone.zone_id for zone in plan.zones if zone.metadata.get("text_truncated")
    ]
    if str(image_path).lower().endswith(".png"):
        image.save(image_path, optimize=True)
    else:
        image.convert("RGB").save(image_path, quality=92, optimize=True)
    return image_path
