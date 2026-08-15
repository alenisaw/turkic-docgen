from __future__ import annotations

import json

from turkicdocgen.page_planning.content.audit import generate_iteration3_reports
from turkicdocgen.qa import validate_page_plan
from turkicdocgen.schema import LineBox, PagePlan, TextStyle, Zone


def test_latin_confusable_validation() -> None:
    text_with_confusable = "каб\u0079л"
    zone = Zone(
        zone_id="test_zone",
        zone_type="body",
        bbox=(0, 0, 100, 100),
        polygon=[],
        reading_order=0,
        text=text_with_confusable,
        language="kk",
        style=TextStyle("DejaVuSans", 20),
        lines=[
            LineBox(
                line_id="l1",
                bbox=(0, 0, 100, 20),
                text=text_with_confusable,
                reading_order=0,
            )
        ],
        cells=[],
    )
    plan = PagePlan(
        page_id="test_page",
        width=800,
        height=1200,
        layout_id="official_statement_page",
        language_mix="kk",
        quality_profile="clean",
        zones=[zone],
        effects=[],
    )
    report = validate_page_plan(plan)
    assert not report.ok
    assert any(i.code == "latin_confusable_detected" for i in report.issues)


def test_character_inventory_validation() -> None:
    zone = Zone(
        zone_id="test_zone",
        zone_type="body",
        bbox=(0, 0, 100, 100),
        polygon=[],
        reading_order=0,
        text="Әлем",
        language="ky",
        style=TextStyle("DejaVuSans", 20),
        lines=[
            LineBox(line_id="l1", bbox=(0, 0, 100, 20), text="Әлем", reading_order=0)
        ],
        cells=[],
    )
    plan = PagePlan(
        page_id="test_page",
        width=800,
        height=1200,
        layout_id="official_statement_page",
        language_mix="ky",
        quality_profile="clean",
        zones=[zone],
        effects=[],
    )
    report = validate_page_plan(plan)
    assert not report.ok
    assert any(i.code == "unsupported_character_detected" for i in report.issues)


def test_nfc_normalization_validation() -> None:
    non_nfc_text = "и\u0306"
    zone = Zone(
        zone_id="test_zone",
        zone_type="body",
        bbox=(0, 0, 100, 100),
        polygon=[],
        reading_order=0,
        text=non_nfc_text,
        language="kk",
        style=TextStyle("DejaVuSans", 20),
        lines=[
            LineBox(
                line_id="l1", bbox=(0, 0, 100, 20), text=non_nfc_text, reading_order=0
            )
        ],
        cells=[],
    )
    plan = PagePlan(
        page_id="test_page",
        width=800,
        height=1200,
        layout_id="official_statement_page",
        language_mix="kk",
        quality_profile="clean",
        zones=[zone],
        effects=[],
    )
    report = validate_page_plan(plan)
    assert not report.ok
    assert any(i.code == "text_not_nfc_normalized" for i in report.issues)


def test_character_coverage_includes_lines_cells_and_form_values(tmp_path) -> None:
    rows = [
        {
            "page_id": "p1",
            "language_mix": "kk",
            "zones": [
                {"zone_type": "body", "lines": [{"text": "\u04d8\u04d9"}]},
                {"zone_type": "table", "cells": [{"text": "\u0492\u0493"}]},
                {
                    "zone_type": "form",
                    "metadata": {
                        "rendered_fields": [
                            {
                                "label_text": "\u049a",
                                "value_text": "\u049b",
                            }
                        ]
                    },
                },
            ],
        }
    ]
    generate_iteration3_reports(tmp_path, rows, "visual_300")
    report = json.loads(
        (tmp_path / "reports" / "character_coverage_report.json").read_text(
            encoding="utf-8"
        )
    )
    counts = report["required_special_character_counts"]
    assert counts["\u04d8"] == 1
    assert counts["\u0493"] == 1
    assert counts["\u049a"] == 1
    assert counts["\u049b"] == 1
