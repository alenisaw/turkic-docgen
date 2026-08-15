from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from turkicdocgen.page_planning.planner import build_page_plan
from turkicdocgen.render.fonts import discover_font_paths
from turkicdocgen.render.page import render_plan


def _intersects(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    return max(first[0], second[0]) < min(first[2], second[2]) and max(
        first[1], second[1]
    ) < min(first[3], second[3])


def test_table_cells_remain_stable_and_rendered_inside_bounds(tmp_path: Path) -> None:
    plan = build_page_plan(
        9,
        "visual_300",
        20260612,
        layout_override="simple_table_page",
        effect_override="clean",
    )
    original_text = {
        (zone.zone_id, cell.row, cell.col): cell.text
        for zone in plan.zones
        for cell in zone.cells
    }

    render_plan(plan, tmp_path / "table.png")

    table = next(zone for zone in plan.zones if zone.zone_type == "table")
    assert table.cells
    for cell in table.cells:
        key = (table.zone_id, cell.row, cell.col)
        assert cell.text == original_text[key]
        assert cell.metadata.get("rendered_complete") is True
        rendered_bbox = tuple(cell.metadata["rendered_bbox"])
        assert rendered_bbox[0] >= cell.bbox[0]
        assert rendered_bbox[1] >= cell.bbox[1]
        assert rendered_bbox[2] <= cell.bbox[2]
        assert rendered_bbox[3] <= cell.bbox[3]
        assert int(cell.metadata["rendered_font_size"]) >= 18


def test_form_label_and_value_regions_stay_separate_and_inside_rows(
    tmp_path: Path,
) -> None:
    plan = build_page_plan(
        4,
        "visual_300",
        20260612,
        layout_override="simple_form_page",
        effect_override="clean",
    )

    form = next(zone for zone in plan.zones if zone.zone_id == "fields")
    render_plan(plan, tmp_path / "form.png")

    rendered_fields = form.metadata["rendered_fields"]
    assert rendered_fields
    for field in rendered_fields:
        row_bbox = tuple(field["row_bbox"])
        label_bbox = tuple(field["label_bbox"])
        assert field["rendered_complete"] is True
        assert form.bbox[0] <= row_bbox[0] < row_bbox[2] <= form.bbox[2]
        assert form.bbox[1] <= row_bbox[1] < row_bbox[3] <= form.bbox[3]
        assert row_bbox[0] <= label_bbox[0] <= label_bbox[2] <= row_bbox[2]
        assert row_bbox[1] <= label_bbox[1] <= label_bbox[3] <= row_bbox[3]
        value = field["value_bbox"]
        if value is None:
            continue
        value_bbox = tuple(value)
        assert not _intersects(label_bbox, value_bbox)
        assert row_bbox[0] <= value_bbox[0] <= value_bbox[2] <= row_bbox[2]
        assert row_bbox[1] <= value_bbox[1] <= value_bbox[3] <= row_bbox[3]


def test_cyrillic_italic_and_multiline_text_measure_correctly() -> None:
    image = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(image)
    font_path = next(
        (
            path
            for path in discover_font_paths()
            if "italic" in path.name.lower() or "oblique" in path.name.lower()
        ),
        discover_font_paths()[0],
    )
    font = ImageFont.truetype(str(font_path), 24)
    text = "Қазақ мәтіні\nекінші жол"

    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
    first_line_bbox = draw.textbbox((0, 0), "Қазақ мәтіні", font=font)
    second_line_bbox = draw.textbbox((0, 0), "екінші жол", font=font)

    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]
    assert (bbox[3] - bbox[1]) > (first_line_bbox[3] - first_line_bbox[1])
    assert (bbox[2] - bbox[0]) >= max(
        first_line_bbox[2] - first_line_bbox[0],
        second_line_bbox[2] - second_line_bbox[0],
    )
