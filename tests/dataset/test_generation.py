from __future__ import annotations

import hashlib
import importlib.resources
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from turkicdocgen.dataset import (
    OLD_PAPER_VARIANT_CYCLE,
    _compact_manifest_row_for_audit,
    generate_dataset,
    validate_output,
)
from turkicdocgen.dedup import compute_full_page_dhash
from turkicdocgen.languages import (
    FORBIDDEN_LATIN_FALLBACK_TOKENS,
    canonical_language_mix,
    required_chars,
)
from turkicdocgen.page_planning.content.phrase_builder import (
    is_layout_artifact_line,
    pool,
    read_lines,
    sanitize_corpus_text,
    seed_records,
)
from turkicdocgen.page_planning.layouts.registry import build_layout
from turkicdocgen.page_planning.planner import CORE_LAYOUTS, build_page_plan
from turkicdocgen.qa import validate_page_plan
from turkicdocgen.render.effects import (
    _draw_pen_artifacts,
    _draw_signature_marks,
    _phone_geometry_tier,
    _render_stamp_layer,
    apply_effect_pipeline,
)
from turkicdocgen.render.fonts import _supports, choose_font, valid_fonts
from turkicdocgen.render.page import _paper_base, render_plan
from turkicdocgen.schema import (
    EffectSpec,
    LineBox,
    PagePlan,
    TableCell,
    TextStyle,
    Zone,
)

MIN_BODY_FONT_SIZE_PX = 20
MIN_LANGUAGE_SAMPLE_COUNT = 15
MIN_SIGNATURE_POLYGON_POINTS = 5
MIN_VARIANT_COUNT = 3
FOOTER_ALIGNMENT_TOLERANCE_PX = 8
MIN_RENDERED_FILL_RATIO = 0.55
TEST_PAPER_SAMPLE_SIZE = 2000
PAPER_BRIGHT_WHITE_MIN_RATIO = 0.40
PAPER_BRIGHT_WHITE_MAX_RATIO = 0.50
PAPER_NEUTRAL_WHITE_MIN_RATIO = 0.25
PAPER_NEUTRAL_WHITE_MAX_RATIO = 0.35
PAPER_COOL_WHITE_MIN_RATIO = 0.07
PAPER_COOL_WHITE_MAX_RATIO = 0.17
PAPER_LIGHT_IVORY_MIN_RATIO = 0.06
PAPER_LIGHT_IVORY_MAX_RATIO = 0.14
PAPER_RECYCLED_GRAY_MAX_RATIO = 0.06
GEOMETRY_SAMPLE_SIZE = 4000
GEOMETRY_MILD_MIN_RATIO = 0.52
GEOMETRY_MILD_MAX_RATIO = 0.58
GEOMETRY_MODERATE_MIN_RATIO = 0.22
GEOMETRY_MODERATE_MAX_RATIO = 0.28
GEOMETRY_EXTREME_MIN_RATIO = 0.17
GEOMETRY_EXTREME_MAX_RATIO = 0.23
LIGHT_PERSPECTIVE_MIN_ANGLE_DEGREES = 9.0
MIXED_LANGUAGE_REPETITION_COUNT = 3
NEWSPAPER_HORIZONTAL_RULES_MIN = 2
TITLE_MIN_FONT_SIZE_PX = 32
TABLE_LEFT_MARGIN_PX = 132
TABLE_RIGHT_MARGIN_PX = 132
TABLE_MIN_FONT_SIZE_PX = 18
TABLE_MAX_FONT_SIZE_PX = 24
TABLE_MIN_TEXT_PADDING_PX = 4
SUMMARY_EXPECTED_SAMPLE_COUNT = 8


def test_all_core_layouts_are_seed_reachable() -> None:
    seen = {build_page_plan(index, "visual_300", 42).layout_id for index in range(1000)}
    assert set(CORE_LAYOUTS).issubset(seen)


def test_plans_have_no_body_justify_or_tiny_body_fonts() -> None:
    for index in range(30):
        plan = build_page_plan(index, "visual_300", 99)
        for zone in plan.zones:
            assert zone.style.align != "justify"
            if zone.zone_type == "body":
                assert zone.style.font_size_px >= MIN_BODY_FONT_SIZE_PX


def test_font_choice_is_deterministic_and_glyph_filtered() -> None:
    first = choose_font("kk", 123)
    second = choose_font("kk", 123)
    assert first == second
    assert valid_fonts("kk")


def test_language_aliases_and_official_layouts_are_canonical_and_complete() -> None:
    assert canonical_language_mix("ru-kk") == "ru_kk"
    assert canonical_language_mix("ru-kg") == "ru_ky"
    bounds = (120, 180, 1534, 2179)
    required = {
        "recipient",
        "applicant",
        "doc_number",
        "date",
        "title",
        "body",
        "attachment_note",
        "signature",
        "stamp_safe",
    }
    for layout_id in ("official_statement_page", "official_letter_page"):
        zones = build_layout(
            layout_id,
            index=3,
            language="ru-kk",
            rng=random.Random(17),
            bounds=bounds,
        )
        assert required.issubset({zone.zone_id for zone in zones})
        assert [zone.reading_order for zone in zones] == sorted(
            zone.reading_order for zone in zones
        )
        text = "\n".join(zone.text for zone in zones)
        assert not any(token in text for token in FORBIDDEN_LATIN_FALLBACK_TOKENS)


def test_stamp_phrases_are_cyrillic_and_canonical() -> None:
    rows = [
        json.loads(line)
        for line in Path(
            str(
                importlib.resources.files("turkicdocgen")
                / "data"
                / "corpus"
                / "stamp_phrases.jsonl"
            )
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    counts: dict[str, int] = {}
    for row in rows:
        language = canonical_language_mix(row["language_mix"])
        counts[language] = counts.get(language, 0) + 1
        assert row["language_mix"] == language
        assert not any(
            token in row["text"] for token in FORBIDDEN_LATIN_FALLBACK_TOKENS
        )
    assert all(
        counts.get(language, 0) >= MIN_LANGUAGE_SAMPLE_COUNT
        for language in ("kk", "ky", "ru_kk", "ru_ky")
    )


def test_compact_audit_row_drops_glyph_traces_but_keeps_gate_inputs() -> None:
    row = {
        "page_id": "page-1",
        "layout_id": "form",
        "effect_metadata": {
            "full_page_dhash_32": "abc",
            "effect_chain": ["blur"],
            "transformed_annotations": [{"glyphs": list(range(100))}],
        },
        "zones": [
            {
                "zone_type": "form",
                "bbox": [1, 2, 3, 4],
                "text": "Value",
                "style": {"font_family": "Noto", "font_size_px": 22, "color": "black"},
                "lines": [
                    {"text": "Value", "bbox": [1, 2, 3, 4], "glyphs": list(range(50))}
                ],
                "metadata": {
                    "role": "body",
                    "glyph_boxes": list(range(100)),
                    "rendered_fields": [
                        {
                            "label_text": "Label",
                            "value_text": "Value",
                            "label_bbox": [1, 2, 2, 3],
                            "value_bbox": [2, 2, 3, 3],
                            "glyph_boxes": list(range(100)),
                        }
                    ],
                },
            }
        ],
    }

    compact = _compact_manifest_row_for_audit(row)

    assert compact["effect_metadata"] == {
        "effect_chain": ["blur"],
        "full_page_dhash_32": "abc",
    }
    assert "transformed_annotations" not in compact["effect_metadata"]
    assert "glyph_boxes" not in compact["zones"][0]["metadata"]
    assert (
        compact["zones"][0]["metadata"]["rendered_fields"][0]["value_text"] == "Value"
    )
    assert compact["zones"][0]["lines"][0] == {
        "bbox": [1, 2, 3, 4],
        "text": "Value",
    }


def test_official_stamped_effect_records_stamp_and_printer_metadata(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (900, 1200), "white").save(image_path)
    zones = build_layout(
        "official_statement_page",
        index=5,
        language="ru_ky",
        rng=random.Random(21),
        bounds=(80, 100, 820, 1120),
    )
    planned_stamp = "КАБЫЛ АЛЫНДЫ / ПРИНЯТО"
    next(zone for zone in zones if zone.zone_type == "stamp").metadata["stamp_text"] = (
        planned_stamp
    )
    plan = PagePlan(
        page_id="sample",
        width=900,
        height=1200,
        layout_id="official_statement_page",
        language_mix="ru_ky",
        quality_profile="official_stamped",
        zones=zones,
        effects=[EffectSpec("official_stamped", "official_stamped", {})],
    )
    result = apply_effect_pipeline(image_path, "official_stamped", plan, seed=123)
    metadata = result.metadata
    assert int(metadata["full_page_dhash_32"], 16) == compute_full_page_dhash(
        image_path, hash_size=32
    )
    stamp = metadata["stamp_metadata"]
    assert stamp["stamp_text"] == planned_stamp
    assert stamp["stamp_language_mix"] == "ru_ky"
    assert stamp["stamp_rotation_degrees"] is not None
    effects = {item["effect"] for item in metadata["effect_chain"]}
    assert {"language_stamp", "signature_marks"}.issubset(effects)
    assert effects.isdisjoint({"sparse_pen_marks", "underlines_checks"})
    assert effects.intersection(
        {"toner_speckles", "roller_streaks", "copy_border_shadow"}
    )
    artifacts = metadata["exact_parameters"]["signature_marks"]["artifacts"]
    assert artifacts
    assert {
        "type",
        "variant",
        "color_rgb",
        "opacity",
        "stroke_width",
        "target_zone",
        "polygon",
        "strokes",
        "bbox",
        "seed",
    }.issubset(artifacts[0])
    assert artifacts[0]["type"] == "handwritten_signature"


def test_manifest_contains_zone_effect_and_font_metadata(tmp_path: Path) -> None:
    generate_dataset("visual_300", tmp_path, seed=42, count=6, force=True)
    assert validate_output(tmp_path) == []
    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows
    assert all(row["effect_metadata"]["effects"] for row in rows)
    assert all(
        set(row["metadata_groups"]).issuperset(
            {
                "identity",
                "generation",
                "language",
                "layout",
                "corpus",
                "render",
                "effects",
                "fonts",
                "qa",
                "review",
                "release",
            }
        )
        for row in rows
    )
    assert all(
        {"family", "path", "source", "category", "coverage_language", "coverage_ok"}
        <= set(font)
        for row in rows
        for font in row["selected_fonts"]
    )
    assert any(
        zone["style"].get("font_path") or zone["style"].get("font_family")
        for row in rows
        for zone in row["zones"]
    )
    assert (tmp_path / "zone_gt.jsonl").exists()
    assert (tmp_path / "ocr_det.jsonl").exists()


def test_manifest_effect_metadata_has_exact_profile_chain_params(
    tmp_path: Path,
) -> None:
    generate_dataset("visual_300", tmp_path, seed=42, count=8, force=True)
    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows
    for row in rows:
        meta = row["effect_metadata"]
        assert row["effect_profile"] == meta["effect_profile"]
        assert meta["effect_chain"]
        assert isinstance(meta["exact_parameters"], dict)
        assert meta["seed"]
        assert row["selected_fonts"]
        assert "qa_flags" in row


def test_typed_table_cells_match_column_schema() -> None:
    zones = build_layout(
        "simple_table_page",
        index=9,
        language="ru_kk",
        rng=random.Random(902),
        bounds=(120, 180, 1534, 2179),
    )
    table = next(zone for zone in zones if zone.zone_type == "table")
    assert table.metadata["content_schema_id"] in {
        "appeal_registry",
        "employee_list",
        "document_log",
        "academic_results",
        "inventory",
        "expense_register",
    }
    headers = [cell for cell in table.cells if cell.metadata["header"]]
    assert len(headers) == table.metadata["cols"]
    assert all(cell.metadata["value_type"] for cell in table.cells)
    for cell in table.cells:
        if cell.metadata["value_type"] == "sequence" and not cell.metadata["header"]:
            assert cell.text.isdigit()
        if cell.metadata["value_type"] == "amount" and not cell.metadata["header"]:
            assert cell.metadata["align"] == "right"


def test_phone_photo_records_invertible_homography(tmp_path: Path) -> None:
    image_path = tmp_path / "phone.jpg"
    Image.new("RGB", (900, 1200), "white").save(image_path)
    zones = build_layout(
        "official_letter_page",
        index=4,
        language="kk",
        rng=random.Random(8),
        bounds=(80, 100, 820, 1120),
    )
    plan = PagePlan(
        page_id="phone",
        width=900,
        height=1200,
        layout_id="official_letter_page",
        language_mix="kk",
        quality_profile="phone_photo",
        zones=zones,
    )
    result = apply_effect_pipeline(image_path, "phone_photo", plan, seed=987)
    transform = result.metadata["transform"]
    assert transform["kind"] == "cumulative_homography"
    assert transform["components"] == [
        "rotation",
        "perspective",
        "fit_scale",
        "translation",
    ]
    forward = np.asarray(transform["forward"])
    inverse = np.asarray(transform["inverse"])
    assert np.allclose(forward @ inverse, np.eye(3), atol=1e-6)
    assert all(zone.polygon for zone in plan.zones)


def test_transforms_update_line_and_cell_polygons_without_empty_boxes(
    tmp_path: Path,
) -> None:
    for layout_id in ("book_page_single_column", "simple_table_page"):
        plan = build_page_plan(
            17,
            "visual_300",
            991,
            layout_override=layout_id,
            effect_override="phone_photo",
        )
        image_path = tmp_path / f"{layout_id}.png"
        render_plan(plan, image_path)
        original_lines = {
            line.line_id: tuple(line.polygon)
            for zone in plan.zones
            for line in zone.lines
            if line.text.strip()
        }
        original_cells = {
            (zone.zone_id, cell.row, cell.col): tuple(cell.polygon)
            for zone in plan.zones
            for cell in zone.cells
            if cell.text.strip()
        }
        apply_effect_pipeline(image_path, "phone_photo", plan, seed=12345)

        assert all(
            line.text.strip() and line.polygon and line.bbox[0] < line.bbox[2]
            for zone in plan.zones
            for line in zone.lines
        )
        assert all(
            cell.text.strip() and cell.polygon and cell.bbox[0] < cell.bbox[2]
            for zone in plan.zones
            for cell in zone.cells
        )
        if original_lines:
            assert any(
                tuple(line.polygon) != original_lines[line.line_id]
                for zone in plan.zones
                for line in zone.lines
                if line.line_id in original_lines
            )
        if original_cells:
            assert any(
                tuple(cell.polygon)
                != original_cells[(zone.zone_id, cell.row, cell.col)]
                for zone in plan.zones
                for cell in zone.cells
                if (zone.zone_id, cell.row, cell.col) in original_cells
            )


def test_stamp_renderer_supports_five_actual_shapes() -> None:
    shapes = set()
    for shape_index in range(5):
        layer, _, metadata = _render_stamp_layer(
            None,
            {
                "id": "test",
                "text": "ҚАБЫЛДАНДЫ",
                "style": "rectangular_stamp",
                "language_mix": "kk",
            },
            (0, 0, 240, 160),
            random.Random(shape_index),
            alpha=110,
            angle=float(shape_index - 2),
            color_pick=shape_index / 5,
            shape_index=shape_index,
        )
        assert layer.getbbox()
        shapes.add(metadata["stamp_shape"])
    assert shapes == {
        "round_seal",
        "oval_seal",
        "rectangular_approval",
        "received_date",
        "archive_copy",
    }


def test_manifest_has_content_schema_fields(tmp_path: Path) -> None:
    generate_dataset(
        "visual_300",
        tmp_path,
        seed=12,
        count=2,
        force=True,
        layout="simple_table_page",
        language="ky",
        effect="clean",
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["content_schema_id"] for row in rows)
    assert all(row["layout_variant"] for row in rows)
    assert all(row["generator_schema_version"] == "2.1" for row in rows)


def test_forms_have_one_low_signature_and_distinct_structural_dates() -> None:
    registration = None
    for seed in range(100):
        zones = build_layout(
            "simple_form_page",
            index=seed,
            language="kk",
            rng=random.Random(seed),
            bounds=(120, 180, 1534, 2179),
        )
        fields = next(zone for zone in zones if zone.zone_id == "fields")
        if fields.metadata["content_schema_id"] == "document_registration":
            registration = zones
            break
    assert registration is not None
    signatures = [zone for zone in registration if zone.metadata.get("signature_role")]
    assert len(signatures) == 1
    assert signatures[0].bbox[1] >= 2339 * 0.70
    fields = next(zone for zone in registration if zone.zone_id == "fields")
    assert all(item["type"] != "signature" for item in fields.metadata["field_types"])
    internal_dates = re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b", fields.text)
    external_date = next(zone.text for zone in registration if zone.zone_id == "date")
    assert internal_dates
    assert external_date not in internal_dates


def test_seed_corpus_excludes_layout_wrappers_and_signature_placeholders() -> None:
    raw = (
        "Алматы қаласының оқу бөлімі басшысына\n"
        "Айдана атынан\n\n"
        "ӨТІНІШ\n"
        "Негізгі мазмұн сақталуы керек.\n"
        "Күні: 09.04.2026 ж.    Қолы: ____________"
    )
    assert sanitize_corpus_text(raw) == "Негізгі мазмұн сақталуы керек."
    assert all(
        not is_layout_artifact_line(line)
        for record in seed_records()
        for line in record.text.splitlines()
    )
    assert all("____" not in record.text for record in seed_records())


def test_official_body_contains_content_not_nested_document_layout() -> None:
    forbidden = ("____", "\nӨТІНІШ\n", "\nАРЫЗ\n", "Қолы:", "Колу:")
    for language in ("kk", "ky", "ru_kk", "ru_ky"):
        for layout in ("official_statement_page", "official_letter_page"):
            for seed in range(12):
                zones = build_layout(
                    layout,
                    index=seed,
                    language=language,
                    rng=random.Random(seed),
                    bounds=(120, 180, 1534, 2179),
                )
                body = next(zone.text for zone in zones if zone.zone_id == "body")
                assert not any(token in body for token in forbidden)


def test_handwritten_signatures_have_multiple_variants(tmp_path: Path) -> None:
    zones = build_layout(
        "official_statement_page",
        index=7,
        language="kk",
        rng=random.Random(7),
        bounds=(120, 180, 1534, 2179),
    )
    plan = PagePlan(
        page_id="signature-variants",
        width=1654,
        height=2339,
        layout_id="official_statement_page",
        language_mix="kk",
        quality_profile="office_scan",
        zones=zones,
    )
    render_plan(plan, tmp_path / "signature-layout.jpg")
    signature_zone = next(
        zone for zone in plan.zones if zone.metadata.get("role") == "signature_zone"
    )
    mark_x1, mark_y1, mark_x2, mark_y2 = signature_zone.metadata["signature_mark_bbox"]
    line_y = signature_zone.metadata["signature_line"][1]
    variants = set()
    for seed in range(24):
        image = Image.new("RGBA", (1654, 2339), (0, 0, 0, 0))
        artifacts = _draw_signature_marks(
            ImageDraw.Draw(image, "RGBA"),
            plan,
            random.Random(seed),
            seed=str(seed),
        )
        assert artifacts
        artifact = artifacts[0]
        variants.add(artifact["variant"])
        assert len(artifact["polygon"]) >= MIN_SIGNATURE_POLYGON_POINTS
        assert artifact["strokes"]
        assert artifact["target_zone"] == signature_zone.zone_id
        assert all(
            mark_x1 <= x < mark_x2 and mark_y1 <= y < mark_y2 and y < line_y
            for stroke in artifact["strokes"]
            for x, y in stroke
        )
        assert image.getbbox()
    assert len(variants) >= MIN_VARIANT_COUNT

    pen_targets = set()
    for seed in range(40):
        image = Image.new("RGBA", (1654, 2339), (0, 0, 0, 0))
        pen_targets.update(
            artifact["target_zone"]
            for artifact in _draw_pen_artifacts(
                ImageDraw.Draw(image, "RGBA"),
                plan,
                random.Random(seed),
                count=3,
                seed=str(seed),
            )
        )
    assert "date" not in pen_targets
    assert "signature" not in pen_targets


def test_footer_date_signature_and_stamp_have_separate_aligned_zones(
    tmp_path: Path,
) -> None:
    def intersects(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> bool:
        return max(first[0], second[0]) < min(first[2], second[2]) and max(
            first[1], second[1]
        ) < min(first[3], second[3])

    for index, layout in enumerate(("official_statement_page", "simple_form_page")):
        plan = build_page_plan(
            index,
            "visual_300",
            991,
            layout_override=layout,
            effect_override="official_stamped",
        )
        render_plan(plan, tmp_path / f"{layout}.jpg")
        date_zone = next(zone for zone in plan.zones if zone.zone_id == "date")
        signature_zone = next(
            zone for zone in plan.zones if zone.metadata.get("role") == "signature_zone"
        )
        stamp_zone = next(zone for zone in plan.zones if zone.zone_type == "stamp")
        baseline = int(signature_zone.metadata["signature_line"][1])

        assert int(date_zone.metadata["footer_baseline_y"]) == baseline
        assert (
            abs(date_zone.lines[0].bbox[3] - baseline) <= FOOTER_ALIGNMENT_TOLERANCE_PX
        )
        assert (
            abs(signature_zone.lines[0].bbox[3] - baseline)
            <= FOOTER_ALIGNMENT_TOLERANCE_PX
        )
        assert not intersects(date_zone.bbox, signature_zone.bbox)
        assert not intersects(stamp_zone.bbox, date_zone.bbox)
        assert not intersects(stamp_zone.bbox, signature_zone.bbox)


def test_density_variants_and_rendered_content_are_complete(tmp_path: Path) -> None:
    densities = {
        build_page_plan(index, "visual_300", 606).metadata["layout_density"]
        for index in range(120)
    }
    assert densities == {"standard", "dense", "extended"}
    minimums = {
        "simple_form_page": 0.70,
        "simple_table_page": 0.65,
        "official_statement_page": 0.62,
        "official_letter_page": 0.62,
    }
    for index, layout in enumerate(CORE_LAYOUTS):
        plan = build_page_plan(
            index,
            "visual_300",
            709,
            layout_override=layout,
            effect_override="clean",
        )
        render_plan(plan, tmp_path / f"{layout}.jpg")
        assert not plan.metadata["render_truncated_zones"]
        if layout in minimums:
            assert plan.metadata["content_height_ratio"] >= minimums[layout]
        if layout in {
            "book_page_single_column",
            "book_page_two_columns",
            "academic_abstract_page",
        }:
            body_zones = [zone for zone in plan.zones if zone.zone_type == "body"]
            assert body_zones
            assert (
                min(float(zone.metadata["rendered_fill_ratio"]) for zone in body_zones)
                >= MIN_RENDERED_FILL_RATIO
            )
        assert validate_page_plan(plan).ok


def test_paper_base_and_old_paper_variant_distributions() -> None:
    paper_counts = Counter(_paper_base(f"page_{index}")[0] for index in range(2000))
    assert (
        PAPER_BRIGHT_WHITE_MIN_RATIO
        <= paper_counts["bright_white"] / TEST_PAPER_SAMPLE_SIZE
        <= PAPER_BRIGHT_WHITE_MAX_RATIO
    )
    assert (
        PAPER_NEUTRAL_WHITE_MIN_RATIO
        <= paper_counts["neutral_white"] / TEST_PAPER_SAMPLE_SIZE
        <= PAPER_NEUTRAL_WHITE_MAX_RATIO
    )
    assert (
        PAPER_COOL_WHITE_MIN_RATIO
        <= paper_counts["cool_white"] / TEST_PAPER_SAMPLE_SIZE
        <= PAPER_COOL_WHITE_MAX_RATIO
    )
    assert (
        PAPER_LIGHT_IVORY_MIN_RATIO
        <= paper_counts["light_ivory"] / TEST_PAPER_SAMPLE_SIZE
        <= PAPER_LIGHT_IVORY_MAX_RATIO
    )
    assert (
        paper_counts["recycled_gray"] / TEST_PAPER_SAMPLE_SIZE
        <= PAPER_RECYCLED_GRAY_MAX_RATIO
    )

    variants = Counter(OLD_PAPER_VARIANT_CYCLE)
    assert variants == {
        "neutral_faded": 7,
        "light_warm": 7,
        "archive_gray": 4,
        "strong_yellow": 2,
    }


def test_geometry_tiers_and_pen_artifacts_have_exact_metadata(tmp_path: Path) -> None:
    tier_counts = Counter(
        _phone_geometry_tier(hashlib.sha256(str(index).encode()).hexdigest())
        for index in range(GEOMETRY_SAMPLE_SIZE)
    )
    assert (
        GEOMETRY_MILD_MIN_RATIO
        <= tier_counts["mild"] / GEOMETRY_SAMPLE_SIZE
        <= GEOMETRY_MILD_MAX_RATIO
    )
    assert (
        GEOMETRY_MODERATE_MIN_RATIO
        <= tier_counts["moderate"] / GEOMETRY_SAMPLE_SIZE
        <= GEOMETRY_MODERATE_MAX_RATIO
    )
    assert (
        GEOMETRY_EXTREME_MIN_RATIO
        <= tier_counts["extreme"] / GEOMETRY_SAMPLE_SIZE
        <= GEOMETRY_EXTREME_MAX_RATIO
    )

    extreme_seed = next(
        seed
        for seed in range(1000)
        if _phone_geometry_tier(hashlib.sha256(str(seed).encode()).hexdigest())
        == "extreme"
    )
    image_path = tmp_path / "extreme.jpg"
    plan = build_page_plan(
        11,
        "visual_300",
        808,
        layout_override="simple_form_page",
        effect_override="phone_photo",
    )
    render_plan(plan, image_path)
    result = apply_effect_pipeline(image_path, "phone_photo", plan, seed=extreme_seed)
    assert result.metadata["geometry_tier"] == "extreme"
    assert (
        abs(result.metadata["exact_parameters"]["light_perspective"]["angle_degrees"])
        >= LIGHT_PERSPECTIVE_MIN_ANGLE_DEGREES
    )
    assert result.metadata["degradation_tier"] in {"light", "medium", "heavy"}
    assert result.metadata["transform"]["kind"] == "cumulative_homography"
    assert all(
        0 <= x < plan.width and 0 <= y < plan.height
        for zone in plan.zones
        for x, y in zone.polygon
    )


def test_layout_realism_profile_is_exactly_stratified() -> None:
    plans = [
        build_page_plan(index, "layout_realism_160", 20260606) for index in range(160)
    ]
    per_layout = 160 // len(CORE_LAYOUTS)
    expected = Counter(
        CORE_LAYOUTS[(index // per_layout) % len(CORE_LAYOUTS)] for index in range(160)
    )
    assert Counter(plan.layout_id for plan in plans) == expected
    assert set(plan.language_mix for plan in plans) == {"kk", "ky", "ru_kk", "ru_ky"}
    assert {
        "clean",
        "office_scan",
        "low_dpi_scan",
        "photocopy",
        "phone_photo",
        "old_paper",
        "official_stamped",
    }.issubset(plan.quality_profile for plan in plans)


def test_expanded_layouts_have_distinct_structures() -> None:
    layout_ids = (
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
    )
    fingerprints: dict[tuple, str] = {}
    for index, layout_id in enumerate(layout_ids):
        zones = build_layout(
            layout_id,
            index=index,
            language="kk",
            rng=random.Random(1000 + index),
            bounds=(120, 120, 1534, 2180),
        )
        fingerprint = tuple((item.zone_id, item.zone_type, item.bbox) for item in zones)
        assert fingerprint not in fingerprints, (
            f"{layout_id} duplicates {fingerprints.get(fingerprint)}"
        )
        fingerprints[fingerprint] = layout_id


def test_all_selected_fonts_render_required_language_glyphs() -> None:
    for language in ("kk", "ky", "ru_kk", "ru_ky"):
        fonts = valid_fonts(language)
        assert fonts
        assert all(
            _supports(Path(font.path), required_chars(language))
            for font in fonts
            if font.path
        )


def test_mixed_language_pool_is_turkic_first() -> None:
    for language, local_code in (("ru_kk", "kk"), ("ru_ky", "ky")):
        values = pool(language)
        local = read_lines(f"{local_code}_phrases.txt")
        mixed = read_lines(f"{language}_mixed_phrases.txt")
        russian = read_lines("ru_phrases.txt")
        assert len(values) == (
            len(local) * MIXED_LANGUAGE_REPETITION_COUNT
            + len(mixed) * MIXED_LANGUAGE_REPETITION_COUNT
            + len(russian)
        )
        assert values.count(local[0]) >= MIXED_LANGUAGE_REPETITION_COUNT
        assert values.count(mixed[0]) >= MIXED_LANGUAGE_REPETITION_COUNT
        assert values.count(russian[0]) >= 1


def test_dense_table_text_stays_inside_cells(tmp_path: Path) -> None:
    for index, layout in enumerate(
        ("simple_table_page", "invoice_like_page", "schedule_table_page")
    ):
        plan = build_page_plan(
            index,
            "visual_300",
            20260612,
            layout_override=layout,
            effect_override="clean",
        )
        render_plan(plan, tmp_path / f"{layout}.png")
        table = next(zone for zone in plan.zones if zone.zone_type == "table")
        assert table.bbox[0] >= TABLE_LEFT_MARGIN_PX
        assert table.bbox[2] <= plan.width - TABLE_RIGHT_MARGIN_PX
        for cell in table.cells:
            assert cell.text.strip()
            lines = cell.metadata["rendered_lines"]
            font_size = int(cell.metadata["rendered_font_size"])
            line_height = int(cell.metadata["rendered_line_height"])
            text_y = int(cell.metadata["rendered_text_y"])
            font = choose_font(cell.language, 0, category="mono_or_table_safe")
            rendered_font = ImageFont.truetype(
                table.style.font_path or font.path, font_size
            )
            available_width = cell.bbox[2] - cell.bbox[0] - 14
            available_height = cell.bbox[3] - cell.bbox[1] - 8
            assert len(lines) * line_height <= available_height
            assert TABLE_MIN_FONT_SIZE_PX <= font_size <= TABLE_MAX_FONT_SIZE_PX
            assert text_y - cell.bbox[1] >= TABLE_MIN_TEXT_PADDING_PX
            assert (
                cell.bbox[3] - (text_y + len(lines) * line_height)
                >= TABLE_MIN_TEXT_PADDING_PX
            )
            assert all(
                rendered_font.getlength(line) <= available_width for line in lines
            )


def test_newspaper_rules_and_title_alignment_are_varied() -> None:
    for layout in ("bulletin_or_newspaper_page", "historical_newspaper_page"):
        plan = build_page_plan(
            4,
            "visual_300",
            20260612,
            layout_override=layout,
            effect_override="clean",
        )
        horizontal_rules = [
            zone
            for zone in plan.zones
            if zone.zone_type == "decorative_non_text"
            and zone.metadata.get("orientation") == "horizontal"
        ]
        assert len(horizontal_rules) >= NEWSPAPER_HORIZONTAL_RULES_MIN

    aligns: set[str] = set()
    for index in range(120):
        plan = build_page_plan(index, "visual_300", 20260612)
        aligns.update(
            zone.style.align for zone in plan.zones if zone.zone_type == "title"
        )
    assert {"left", "center", "right"}.issubset(aligns)


def test_titles_remain_large_and_use_all_alignments_after_render(
    tmp_path: Path,
) -> None:
    aligns: set[str] = set()
    for index in range(120):
        plan = build_page_plan(index, "visual_300", 20260615)
        render_plan(plan, tmp_path / f"title_{index:03d}.png")
        for zone in plan.zones:
            if zone.zone_type != "title":
                continue
            aligns.add(zone.style.align)
            assert zone.style.font_size_px >= TITLE_MIN_FONT_SIZE_PX
    assert aligns == {"left", "center", "right"}


def test_repeated_title_layouts_receive_deterministic_variants() -> None:
    titles_by_layout: dict[str, list[str]] = {}
    all_titles: list[str] = []
    for index in range(300):
        plan = build_page_plan(index, "visual_300", 20260611)
        titles = [zone.text for zone in plan.zones if zone.zone_type == "title"]
        titles_by_layout.setdefault(plan.layout_id, []).extend(titles)
        all_titles.extend(titles)

    for layout_id, titles in titles_by_layout.items():
        counts = Counter(titles)
        assert max(counts.values(), default=0) <= 3, layout_id
    assert max(Counter(all_titles).values(), default=0) <= max(
        1, int(len(all_titles) * 0.05)
    )


def test_generation_summary_contains_dataset_level_qa(tmp_path: Path) -> None:
    generate_dataset("visual_300", tmp_path, seed=42, count=8, force=True)
    summary = json.loads(
        (tmp_path / "reports" / "generation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["accepted"] + summary["rejected"] == SUMMARY_EXPECTED_SAMPLE_COUNT
    assert sum(summary["layouts"].values()) == SUMMARY_EXPECTED_SAMPLE_COUNT
    assert sum(summary["languages"].values()) == SUMMARY_EXPECTED_SAMPLE_COUNT
    assert sum(summary["effects"].values()) == SUMMARY_EXPECTED_SAMPLE_COUNT
    assert isinstance(summary["reject_reasons"], dict)
    assert isinstance(summary["top_repeated_lines"], list)
    assert isinstance(summary["table_fonts"], dict)
    assert isinstance(summary["table_text_fit"], dict)
    assert isinstance(summary["title_repeats_by_layout"], dict)


def _qa_zone(
    zone_id: str,
    zone_type: str,
    bbox: tuple[int, int, int, int],
    text: str,
    *,
    font_size: int = 20,
    metadata: dict | None = None,
    lines: list[LineBox] | None = None,
    cells: list[TableCell] | None = None,
) -> Zone:
    return Zone(
        zone_id=zone_id,
        zone_type=zone_type,
        bbox=bbox,
        polygon=[
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]),
        ],
        text=text,
        language="kk",
        reading_order=1,
        style=TextStyle(font_family="sans", font_size_px=font_size),
        lines=lines or [],
        cells=cells or [],
        metadata=metadata or {},
    )


def test_validate_page_plan_flags_new_qa_issues() -> None:
    cell = TableCell(
        row=0,
        col=0,
        bbox=(20, 20, 80, 50),
        text="Long table value",
        language="kk",
        reading_order=1,
        metadata={
            "rendered_lines": ["Long", "table"],
            "rendered_font_size": 12,
            "rendered_line_height": 14,
            "rendered_text_y": 15,
            "value_type": "amount",
        },
    )
    plan = PagePlan(
        page_id="qa",
        width=200,
        height=200,
        layout_id="simple_table_page",
        language_mix="kk",
        quality_profile="visual_300",
        zones=[
            _qa_zone(
                "title",
                "title",
                (10, 10, 180, 30),
                "Дата выдачи документа: 12.04.2026",
            ),
            _qa_zone(
                "table",
                "table",
                (10, 40, 190, 180),
                "",
                font_size=16,
                cells=[cell],
            ),
            _qa_zone(
                "form_a",
                "form",
                (10, 120, 190, 170),
                "A: B",
                metadata={
                    "rendered_fields": [
                        {
                            "label_bbox": [20, 125, 100, 150],
                            "value_bbox": [90, 125, 170, 150],
                            "rendered_complete": False,
                        }
                    ]
                },
            ),
        ],
    )
    plan.zones[1].metadata["text_truncated"] = True
    report = validate_page_plan(plan)
    codes = {issue.code for issue in report.issues}
    assert "table_cell_text_truncated" in codes
    assert "table_cell_font_too_small" in codes
    assert "rendered_text_outside_cell" in codes
    assert "form_field_overlap" in codes
    assert "title_semantic_mismatch" in codes


def test_duplicate_text_warning_includes_value_type() -> None:
    repeated = "Repeated long note"
    plan = PagePlan(
        page_id="dup",
        width=200,
        height=200,
        layout_id="simple_table_page",
        language_mix="kk",
        quality_profile="visual_300",
        zones=[
            _qa_zone(
                "table",
                "table",
                (10, 10, 190, 190),
                "",
                cells=[
                    TableCell(
                        row=0,
                        col=0,
                        bbox=(10, 10, 60, 40),
                        text=repeated,
                        language="kk",
                        reading_order=1,
                        metadata={"value_type": "note"},
                    ),
                    TableCell(
                        row=0,
                        col=1,
                        bbox=(60, 10, 110, 40),
                        text=repeated,
                        language="kk",
                        reading_order=2,
                        metadata={"value_type": "note"},
                    ),
                    TableCell(
                        row=1,
                        col=0,
                        bbox=(10, 40, 60, 70),
                        text=repeated,
                        language="kk",
                        reading_order=3,
                        metadata={"value_type": "note"},
                    ),
                    TableCell(
                        row=1,
                        col=1,
                        bbox=(60, 40, 110, 70),
                        text=repeated,
                        language="kk",
                        reading_order=4,
                        metadata={"value_type": "note"},
                    ),
                    TableCell(
                        row=2,
                        col=0,
                        bbox=(10, 70, 60, 100),
                        text=repeated,
                        language="kk",
                        reading_order=5,
                        metadata={"value_type": "note"},
                    ),
                ],
            )
        ],
    )
    report = validate_page_plan(plan)
    duplicate = next(
        issue for issue in report.issues if issue.code == "excessive_duplicates"
    )
    assert "note" in duplicate.message


def test_short_repeated_body_fragments_do_not_trigger_duplicate_warning() -> None:
    repeated = "Short repeated fragment."
    plan = PagePlan(
        page_id="body-dup",
        width=600,
        height=800,
        layout_id="book_page_single_column",
        language_mix="kk",
        quality_profile="clean",
        zones=[
            _qa_zone(
                "body",
                "body",
                (20, 20, 580, 780),
                repeated,
                lines=[
                    LineBox(
                        line_id=f"line-{index}",
                        bbox=(30, 30 + index * 30, 300, 55 + index * 30),
                        text=repeated,
                        reading_order=index,
                    )
                    for index in range(5)
                ],
            )
        ],
    )
    report = validate_page_plan(plan)
    assert "excessive_duplicates" not in {issue.code for issue in report.issues}
