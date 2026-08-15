from __future__ import annotations

import json
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from turkicdocgen.export import export_page
from turkicdocgen.page_planning.content.document_models import (
    build_document_context,
    value_for,
)
from turkicdocgen.page_planning.language_mixing import (
    attach_mixing_metadata,
    estimate_language_mix_ratio,
    resolve_primary_secondary,
)
from turkicdocgen.page_planning.planner import build_page_plan
from turkicdocgen.qa import validate_page_plan
from turkicdocgen.schema import PagePlan, QAReport, TextStyle, Zone

RATIO_TOLERANCE = 0.01


def _zone(
    zone_id: str,
    zone_type: str,
    text: str,
    *,
    role: str | None = None,
) -> Zone:
    metadata = {"role": role} if role else {}
    return Zone(
        zone_id=zone_id,
        zone_type=zone_type,
        bbox=(10, 10, 300, 120),
        polygon=[(10, 10), (300, 10), (300, 120), (10, 120)],
        text=text,
        language="kk",
        reading_order=1,
        style=TextStyle("DejaVu Sans", 20),
        metadata=metadata,
    )


def test_mixed_languages_resolve_primary_and_secondary() -> None:
    assert resolve_primary_secondary("ru_kk") == ("kk", "ru")
    assert resolve_primary_secondary("ru_ky") == ("ky", "ru")


def test_monolingual_plans_have_no_mixing_features() -> None:
    for language in ("kk", "ky"):
        plan = build_page_plan(
            0,
            "visual_300",
            42,
            language_override=language,
            layout_override="simple_form_page",
            effect_override="clean",
        )
        assert plan.metadata["primary_language"] == language
        assert plan.metadata["secondary_language"] is None
        assert plan.metadata["mixing_features"] == []


def test_clean_mixed_plans_do_not_declare_invisible_stamp_mixing() -> None:
    for index in range(40):
        plan = build_page_plan(
            index,
            "visual_300",
            42,
            language_override="ru_kk",
            layout_override="simple_form_page",
            effect_override="clean",
        )
        assert "stamp_level" not in plan.metadata["mixing_features"]


def test_mixed_form_and_table_features_mark_zones() -> None:
    form_plan = build_page_plan(
        0,
        "visual_300",
        42,
        language_override="ru_kk",
        layout_override="simple_form_page",
        effect_override="clean",
    )
    assert attach_mixing_metadata(form_plan.zones, "ru_kk", ["field_level"])
    form_zone = next(zone for zone in form_plan.zones if zone.zone_type == "form")
    assert form_zone.language == "bilingual_kk_ru"
    assert form_zone.metadata["mixing_feature"] == "field_level"

    table_plan = build_page_plan(
        0,
        "visual_300",
        42,
        language_override="ru_ky",
        layout_override="simple_table_page",
        effect_override="clean",
    )
    assert attach_mixing_metadata(table_plan.zones, "ru_ky", ["table_level"])
    table_zone = next(zone for zone in table_plan.zones if zone.zone_type == "table")
    headers = [cell for cell in table_zone.cells if cell.row == 0]
    assert headers
    assert all(cell.language == "bilingual_ky_ru" for cell in headers)
    assert all(cell.metadata["mixing_feature"] == "table_level" for cell in headers)


def test_header_footer_and_parallel_lines_do_not_overwrite_title_fallback() -> None:
    title = _zone("title", "title", "Original title", role="title")

    assert attach_mixing_metadata([title], "ru_kk", ["header_footer"]) == []
    assert title.text == "Original title"
    assert attach_mixing_metadata([title], "ru_kk", ["parallel_lines"]) == []
    assert title.text == "Original title"


def test_parallel_lines_preserve_source_date() -> None:
    metadata = _zone("meta", "metadata", "Issued: 23.09.2024")

    assert attach_mixing_metadata([metadata], "ru_kk", ["parallel_lines"]) == [
        "parallel_lines"
    ]
    assert metadata.text.count("23.09.2024") == 2
    assert "12.04.2026" not in metadata.text


def test_section_level_changes_form_section_not_document_title() -> None:
    title = _zone("title", "title", "Application", role="title")
    form = _zone("form", "form", "[Applicant]\nName: Ada")

    assert attach_mixing_metadata([title, form], "ru_ky", ["section_level"]) == [
        "section_level"
    ]
    assert title.text == "Application"
    assert form.text.splitlines()[0] != "[Applicant]"
    assert form.metadata["section_heading_mixed"] is True


def test_row_dates_and_document_numbers_vary_from_context_date() -> None:
    rng = random.Random(17)
    context = build_document_context("kk", 8, rng)

    dates = [value_for("date", "date", context, row, rng) for row in range(3)]
    numbers = [value_for("doc", "doc", context, row, rng) for row in range(3)]

    assert len(set(dates)) == 3
    assert all(number.startswith(dates[row][-4:]) for row, number in enumerate(numbers))


def test_table_items_names_and_notes_have_enough_row_variation() -> None:
    rng = random.Random(19)
    context = build_document_context("ru_kk", 12, rng)

    for value_type in ("name", "item", "note"):
        values = [
            value_for(value_type, value_type, context, row, rng) for row in range(10)
        ]
        assert len(set(values)) >= 8


def test_every_mixed_page_gets_a_compatible_secondary_language_feature() -> None:
    for index in range(300):
        plan = build_page_plan(index, "visual_300", 20260611)
        if plan.language_mix not in {"ru_kk", "ru_ky"}:
            continue
        assert plan.metadata["mixing_features"], (index, plan.layout_id)
        assert plan.metadata["language_mix_ratio"]["ru"] > 0.0, (
            index,
            plan.layout_id,
        )


def test_export_preserves_page_and_zone_mixing_metadata(tmp_path: Path) -> None:
    plan = build_page_plan(
        0,
        "visual_300",
        42,
        language_override="ru_kk",
        layout_override="simple_table_page",
        effect_override="clean",
    )
    applied = attach_mixing_metadata(plan.zones, "ru_kk", ["table_level"])
    plan.metadata["mixing_features"] = applied
    plan.metadata["language_mix_ratio"] = estimate_language_mix_ratio(
        plan.zones, plan.language_mix
    )
    export_page(tmp_path, plan, QAReport(ok=True), "images/sample.png")

    manifest = json.loads(
        (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert manifest["primary_language"] == "kk"
    assert manifest["secondary_language"] == "ru"
    assert manifest["mixing_features"] == ["table_level"]
    assert abs(sum(manifest["language_mix_ratio"].values()) - 1.0) < RATIO_TOLERANCE

    zone_gt = json.loads(
        (tmp_path / "zone_gt.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    table = next(zone for zone in zone_gt["zones"] if zone["zone_type"] == "table")
    assert table["language"] == "bilingual_kk_ru"
    assert table["metadata"]["mixing_feature"] == "table_level"
    assert any(
        cell["metadata"]["mixing_feature"] == "table_level"
        for cell in table["cells"]
        if cell["row"] == 0
    )


def test_qa_warns_when_mixed_sample_has_no_secondary_content() -> None:
    zone = Zone(
        zone_id="body",
        zone_type="body",
        bbox=(10, 10, 190, 190),
        polygon=[(10, 10), (190, 10), (190, 190), (10, 190)],
        text="Қазақша мәтін",
        language="kk",
        reading_order=1,
        style=TextStyle("DejaVu Sans", 20),
    )
    plan = PagePlan(
        page_id="missing-secondary",
        width=200,
        height=200,
        layout_id="book_page_single_column",
        language_mix="ru_kk",
        quality_profile="clean",
        zones=[zone],
        metadata={
            "primary_language": "kk",
            "secondary_language": "ru",
            "mixing_features": [],
            "language_mix_ratio": {"kk": 1.0, "ru": 0.0, "en": 0.0},
        },
    )
    report = validate_page_plan(plan)
    assert any(
        issue.code == "mixed_language_has_no_secondary_content"
        for issue in report.issues
    )
