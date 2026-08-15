from __future__ import annotations

import json
from collections import Counter

from turkicdocgen.page_planning.content.audit import normalize_text, run_diversity_audit


def test_run_diversity_audit() -> None:
    sample_rows = [
        {
            "layout_id": "official_statement_page",
            "language_mix": "kk",
            "zones": [
                {
                    "zone_id": "title",
                    "zone_type": "title",
                    "text": "ӨТІНІШ",
                    "metadata": {},
                },
                {
                    "zone_id": "body",
                    "zone_type": "body",
                    "text": "Бұл құжат мәтіні. Тексеру үшін осында жазылды. Қосымша ақпарат жоқ.",
                    "metadata": {},
                },
            ],
        }
    ]
    report = run_diversity_audit(sample_rows)
    assert report["layouts"]["official_statement_page"] == 1
    assert report["languages"]["kk"] == 1
    assert report["title_distribution"]["official_statement_page"]["ӨТІНІШ"] == 1


def test_diversity_audit_normalizes_duplicates_and_ignores_status_as_department() -> (
    None
):
    repeated = "Ұзақ ресми мәтін тексеру үшін қайталанады және тыныс белгілері өзгерсе де бір мазмұн болып қалады."
    rows = [
        {
            "layout_id": "application_form_page",
            "language_mix": "kk",
            "orientation": "landscape",
            "qa_flags": ["excessive_duplicates"],
            "zones": [
                {
                    "zone_type": "body",
                    "text": repeated,
                    "metadata": {
                        "rendered_fields": [
                            {"field_key": "office_status", "value_text": "Қабылданды"},
                            {
                                "field_key": "request_summary",
                                "value_text": "Құжаттарды тіркеу туралы",
                            },
                        ]
                    },
                }
            ],
        },
        {
            "layout_id": "application_form_page",
            "language_mix": "kk",
            "zones": [
                {
                    "zone_type": "body",
                    "text": repeated.upper() + "!",
                    "metadata": {},
                }
            ],
        },
    ]
    report = run_diversity_audit(rows)
    assert report["orientations"] == {"landscape": 1, "portrait": 1}
    assert report["qa_flags"] == {"excessive_duplicates": 1}
    assert report["fields"]["department"]["count"] == 0
    assert report["fields"]["subject"]["count"] == 1
    assert report["normalized_duplicate_clusters"][0]["count"] == 2
    assert report["title_repeat_violations"] == []
    assert normalize_text("  ТЕСТ, мәтіні! ") == "тест мәтіні"


def test_generate_iteration6_reports(tmp_path) -> None:
    import json

    from turkicdocgen.page_planning.content.audit import generate_iteration6_reports

    rows = [
        {
            "layout_id": "simple_form_page",
            "layout_density": "dense",
            "qa_flags": ["excessive_overlap", "rendered_text_truncated"],
            "zones": [
                {
                    "zone_id": "form_zone",
                    "zone_type": "form",
                    "text": "Аты: Ален\nЖұмысы: Бағдарламашы",
                    "style": {"font_size_px": 20, "font_family": "DejaVuSans"},
                    "metadata": {
                        "form_label_width": 100,
                        "rendered_fields": [
                            {
                                "kind": "field",
                                "field_key": "name",
                                "label_text": "Аты",
                                "value_text": "Ален",
                                "label_bbox": [10, 10, 50, 30],
                                "value_bbox": [110, 10, 150, 30],
                                "row_bbox": [0, 5, 200, 35],
                                "rendered_complete": True,
                                "font_size": 20,
                                "wrap_state": False,
                            }
                        ],
                        "rendered_entry_count": 1,
                        "wrap_state": "no_wrap",
                        "completion_state": "complete",
                    },
                }
            ],
        }
    ]

    generate_iteration6_reports(tmp_path, rows)
    report_file = tmp_path / "reports" / "quality_report.json"
    assert report_file.exists()

    with open(report_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["font_size_distribution"]["20"] == 2
    assert data["wrap_and_fit_distribution"]["total_elements"] == 2
    assert data["truncation_and_overlap_counts"]["total_overlaps"] == 1
    assert data["truncation_and_overlap_counts"]["total_truncations"] == 1
    assert data["density_distribution_by_family"]["form"]["dense"] == 1


def test_iteration6_hard_gate_blocks_release_profile(tmp_path) -> None:
    import pytest

    from turkicdocgen.page_planning.content.audit import generate_iteration6_reports

    rows = [
        {
            "layout_id": "simple_table_page",
            "qa_flags": ["table_cell_text_truncated"],
            "zones": [],
        }
    ]
    with pytest.raises(ValueError, match="Hard rendering quality gate"):
        generate_iteration6_reports(tmp_path, rows, "tiny_25k")


def test_visual_audit_manifest_is_deterministic_and_stratified(tmp_path) -> None:
    from turkicdocgen.page_planning.content.audit import (
        generate_visual_audit_manifest,
    )

    rows = [
        {
            "page_id": f"page-{index}",
            "layout_id": "book_page_single_column"
            if index % 2
            else "simple_table_page",
            "layout_variant": f"variant-{index % 3}",
            "quality_profile": "clean" if index % 2 else "office_scan",
            "effect_chain": ["noise"] if index % 2 else [],
            "language_mix": "kk" if index % 2 else "ky",
            "orientation": "portrait",
            "layout_density": "standard",
            "zones": [
                {
                    "style": {
                        "font_family": "DejaVuSans",
                        "font_size_px": 22,
                    }
                }
            ],
        }
        for index in range(12)
    ]
    generate_visual_audit_manifest(tmp_path, rows)
    first = (tmp_path / "reports" / "visual_audit_manifest.json").read_text(
        encoding="utf-8"
    )
    generate_visual_audit_manifest(tmp_path, list(reversed(rows)))
    second = (tmp_path / "reports" / "visual_audit_manifest.json").read_text(
        encoding="utf-8"
    )
    report = json.loads(second)
    assert first == second
    assert report["selected_count"] == len(rows)
    assert {sample["layout_id"] for sample in report["samples"]} == {
        "book_page_single_column",
        "simple_table_page",
    }


def test_visual_audit_manifest_balances_layouts_before_deep_strata(tmp_path) -> None:
    from turkicdocgen.page_planning.content.audit import (
        generate_visual_audit_manifest,
    )

    rows = [
        {
            "page_id": f"page-{index:05d}",
            "layout_id": f"layout-{index % 10}",
            "layout_variant": f"variant-{index % 4}",
            "quality_profile": f"quality-{index % 3}",
            "effect_chain": [f"effect-{index % 7}"],
            "language_mix": f"lang-{index % 4}",
            "orientation": "portrait",
            "layout_density": "standard",
            "zones": [],
        }
        for index in range(10_000)
    ]

    generate_visual_audit_manifest(tmp_path, rows)

    report = json.loads(
        (tmp_path / "reports" / "visual_audit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    layout_counts = Counter(sample["layout_id"] for sample in report["samples"])
    assert report["selected_count"] == 100
    assert layout_counts == {f"layout-{index}": 10 for index in range(10)}
