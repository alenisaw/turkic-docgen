import json
from pathlib import Path

from turkicdocgen.export import export_page
from turkicdocgen.hf.dataset_card import dataset_release_name
from turkicdocgen.profiles import load_profiles
from turkicdocgen.qa import validate_page_plan
from turkicdocgen.release_config import RELEASE_CONFIG_TARGETS
from turkicdocgen.schema import LineBox, PagePlan, TableCell, TextStyle, Zone


def test_ocr_cleanups_and_consistency(tmp_path: Path):
    style = TextStyle("Inter", 12)
    # Construct a plan with some empty texts, some tiny boxes, and some normal lines
    lines = [
        LineBox("l1", (10, 10, 100, 30), "Valid text", 1),
        LineBox("l2", (10, 35, 100, 55), "   ", 2),  # Empty/whitespace text
        LineBox(
            "l3", (10, 60, 12, 62), "Tiny box", 3
        ),  # Width = 2, height = 2 (under threshold 4x6)
        LineBox("l4", (10, 65, 100, 85), None, 4),  # None text
    ]
    cells = [
        TableCell(0, 0, (20, 20, 120, 40), "Cell text", "kk", 1),
        TableCell(0, 1, (120, 20, 220, 40), " ", "kk", 2),  # Empty/whitespace cell
        TableCell(0, 2, (220, 20, 222, 22), "Tiny cell", "kk", 3),  # Tiny bbox
    ]
    zone_normal = Zone(
        "z1",
        "paragraph",
        (10, 10, 300, 300),
        [],
        "Valid text",
        "kk",
        1,
        style,
        lines=lines,
        cells=cells,
    )
    zone_decorative = Zone(
        "z2",
        "decorative_non_text",
        (10, 350, 100, 400),
        [],
        "",
        "kk",
        2,
        style,
        lines=[LineBox("ldec1", (20, 360, 80, 380), "Decorative text", 1)],
    )

    plan = PagePlan(
        page_id="test_sample_01",
        width=1000,
        height=1000,
        layout_id="simple_table_page",
        language_mix="kk",
        quality_profile="clean",
        zones=[zone_normal, zone_decorative],
    )

    # 1. QA Checks
    qa = validate_page_plan(plan)
    # Should report empty_ocr_label and tiny_ocr_box
    issues_codes = {issue.code for issue in qa.issues}
    assert "empty_ocr_label" in issues_codes
    assert "tiny_ocr_box" in issues_codes

    # 2. Export page checks
    export_page(tmp_path, plan, qa, "images/test_sample_01.png")

    # Check that det.jsonl and rec.jsonl exist
    det_file = tmp_path / "ocr_det.jsonl"
    rec_file = tmp_path / "ocr_rec.jsonl"
    assert det_file.exists()
    assert rec_file.exists()

    det_lines = [
        json.loads(line)
        for line in det_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rec_lines = [
        json.loads(line)
        for line in rec_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Check that empty lines/cells and tiny boxes are filtered out
    # Also decorative zone line ldec1 is skipped
    det_line_ids = {item["line_id"] for item in det_lines}
    rec_line_ids = {item["line_id"] for item in rec_lines}

    # "l1" (normal line) and "z1_cell_0_0" (normal cell) should be exported
    assert "l1" in det_line_ids
    assert "l1" in rec_line_ids
    assert "z1_cell_0_0" in det_line_ids
    assert "z1_cell_0_0" in rec_line_ids

    # Empty / tiny / decorative lines should NOT be in det or rec
    assert "l2" not in det_line_ids
    assert "l3" not in det_line_ids
    assert "l4" not in det_line_ids
    assert "z1_cell_0_1" not in det_line_ids
    assert "z1_cell_0_2" not in det_line_ids
    assert "ldec1" not in det_line_ids

    # 4. Consistency: det and rec lines should match exactly in count and ids
    assert len(det_lines) == len(rec_lines)
    assert det_line_ids == rec_line_ids
    assert all(item["page_id"] == plan.page_id for item in det_lines + rec_lines)
    assert all(item["region_id"] == item["line_id"] for item in det_lines + rec_lines)


def test_jpeg_output_format_and_profiles():
    # 5. JPEG output format test:
    cfg = load_profiles()
    assert cfg.get("image_format") == "jpg"

    # 6. Profile names test:
    profiles = cfg.get("profiles", {})
    assert "tiny_25k" in profiles
    assert "medium_50k" in profiles
    assert "large_100k" in profiles
    assert "medium_100k" not in profiles
    assert "large_250k" not in profiles
    assert "small_50k" not in profiles
    assert "large_200k" not in profiles


def test_public_dataset_release_names():
    assert RELEASE_CONFIG_TARGETS == {
        "tiny": 25_000,
        "medium": 50_000,
        "large": 100_000,
    }
    assert dataset_release_name(25_000).endswith("Tiny 25,000")
    assert dataset_release_name(50_000).endswith("Medium 50,000")
    assert dataset_release_name(100_000).endswith("Large 100,000")
    assert "250" not in dataset_release_name(100_000)
