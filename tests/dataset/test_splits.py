from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from turkicdocgen.splits import (
    assign_splits,
    build_components,
    check_leakage,
    has_rare_characters,
    load_duplicate_clusters,
    process_dataset_splits,
    stratify_and_rank,
)


def _create_dummy_row(
    page_id: str,
    layout_id: str = "lecture_notes_page",
    language_mix: str = "kk",
    primary_language: str = "kk",
    quality_profile: str = "clean",
    content_lineage_id: str | None = None,
    corpus_record_ids: list[str] | None = None,
    grammar_source: str | None = None,
    text: str = "Қазақстан",
) -> dict[str, Any]:
    return {
        "page_id": page_id,
        "layout_id": layout_id,
        "language_mix": language_mix,
        "primary_language": primary_language,
        "quality_profile": quality_profile,
        "content_lineage_id": content_lineage_id,
        "content_record_ids": corpus_record_ids or [],
        "grammar_source": grammar_source,
        "zones": [
            {
                "zone_id": "z1",
                "zone_type": "body",
                "text": text,
                "metadata": {
                    "corpus_record_id": corpus_record_ids[0]
                    if corpus_record_ids
                    else None
                },
            }
        ],
    }


def test_has_rare_characters() -> None:
    # Kazakh/Kyrgyz Cyrillic rare character check
    sample_with_rare = _create_dummy_row("p1", text="Сәлем әлем")
    sample_without_rare = _create_dummy_row("p2", text="Привет мир")

    assert has_rare_characters(sample_with_rare) is True
    assert has_rare_characters(sample_without_rare) is False


def test_build_components_and_leakage_safe_splits() -> None:
    # Create 100 dummy samples with some sharing lineages and corpus records
    rows = []
    for i in range(100):
        # We define a few shared lineages and corpus records to force grouping
        # Components:
        # Group 1: 0, 1, 2 share lineage_1
        # Group 2: 3, 4 share corpus_1
        # Group 3: 5, 6 share grammar_1
        # Group 4: 7, 8 in same duplicate cluster (we will define later)
        # Groups 5+: single pages

        lineage = f"lineage_{i // 10}" if i < 10 else None
        corpus_ids = [f"corpus_{i // 5}"] if i < 20 else []
        grammar = "grammar_source_1" if (20 <= i < 25) else None

        rows.append(
            _create_dummy_row(
                page_id=f"page_{i:03d}",
                content_lineage_id=lineage,
                corpus_record_ids=corpus_ids,
                grammar_source=grammar,
            )
        )

    # Let's add duplicate clusters: page_007 and page_008
    dup_clusters = [["page_007", "page_008"]]

    components = build_components(rows, dup_clusters)

    # Assert that linked pages are indeed in the same component
    # e.g., page_000 to page_009 should be in the same component due to shared lineage and corpus records
    assert any("page_000" in comp and "page_001" in comp for comp in components)
    assert any("page_007" in comp and "page_008" in comp for comp in components)

    # Assign splits
    page_to_split = assign_splits(components, len(rows))

    # Verify split ratios are exactly 90/5/5
    split_counts = {"train": 0, "val": 0, "test": 0}
    for _pid, split in page_to_split.items():
        split_counts[split] += 1

    assert split_counts["train"] == 90
    assert split_counts["val"] == 5
    assert split_counts["test"] == 5

    # Check leakage
    leak_chk = check_leakage(rows, page_to_split, dup_clusters)
    assert leak_chk["overlap_detected"] is False
    assert leak_chk["common_lineages"] == 0
    assert leak_chk["common_corpus_records"] == 0
    assert leak_chk["common_grammar_sources"] == 0
    assert leak_chk["common_duplicate_clusters"] == 0


def test_assign_splits_rejects_impossible_component_partition() -> None:
    components = [[f"p-{group}-{index}" for index in range(6)] for group in range(16)]
    components.extend([[f"single-{index}"] for index in range(4)])
    try:
        assign_splits(components, 100)
    except ValueError as exc:
        assert "impossible" in str(exc) or "singleton" in str(exc)
    else:
        raise AssertionError("Impossible exact component partition was accepted")


def test_lineage_is_not_discarded_when_it_contains_page_id() -> None:
    rows = [
        _create_dummy_row("page-1", content_lineage_id="lineage-page-1-shared"),
        _create_dummy_row("page-2", content_lineage_id="lineage-page-1-shared"),
    ]
    components = build_components(rows, [])
    assert components == [["page-1", "page-2"]]


def test_load_duplicate_clusters_includes_all_visual_and_text_modes(
    tmp_path: Path,
) -> None:
    fields = (
        "exact_meaningful_text_duplicates",
        "normalized_meaningful_text_duplicates",
        "near_meaningful_text_duplicates",
        "exact_full_page_duplicates",
        "near_full_page_duplicates",
        "structural_layout_clusters",
        "page_mask_clusters",
    )
    report = {field: [[f"{field}-a", f"{field}-b"]] for field in fields}
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "duplicate_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert len(load_duplicate_clusters(tmp_path)) == len(fields)


def test_overlapping_duplicate_modes_form_one_transitive_component() -> None:
    rows = [_create_dummy_row(page_id) for page_id in ("a", "b", "c")]
    components = build_components(rows, [["a", "b"], ["b", "c"]])
    assert components == [["a", "b", "c"]]


def test_assign_splits_is_stable_after_component_shuffle() -> None:
    components = [[f"linked-{index}-a", f"linked-{index}-b"] for index in range(5)]
    components.extend([[f"single-{index}"] for index in range(90)])
    first = assign_splits(components, 100)
    second = assign_splits(list(reversed(components)), 100)
    assert first == second


def test_stratify_and_rank() -> None:
    # Create pages belonging to different strata
    samples = []
    # 20 samples from layout_family book, kk language, clean quality
    for i in range(20):
        samples.append(
            _create_dummy_row(
                page_id=f"page_book_{i:02d}",
                layout_id="official_statement_page",  # official layout family
                language_mix="kk",
                quality_profile="clean",
            )
        )
    # 20 samples from layout_family table, ru_kk language, corrupted quality
    for i in range(20):
        samples.append(
            _create_dummy_row(
                page_id=f"page_table_{i:02d}",
                layout_id="simple_table_page",  # table layout family
                language_mix="ru_kk",
                quality_profile="corrupted",
            )
        )

    ranked = stratify_and_rank(samples)
    assert len(ranked) == 40

    # Assert deterministic ordering of the same input list
    ranked_2 = stratify_and_rank(samples[::-1])
    assert [r["page_id"] for r in ranked] == [r["page_id"] for r in ranked_2]

    # Verify round-robin alternating behavior:
    # Since there are two main strata (official vs table), the merged ranked list should alternate
    # between the two strata or preserve similar proportions throughout the list.
    half_size = len(ranked) // 2
    first_half = ranked[:half_size]

    official_count = sum(
        1 for s in first_half if s["layout_id"] == "official_statement_page"
    )
    table_count = sum(1 for s in first_half if s["layout_id"] == "simple_table_page")

    # In a round-robin merge of two equally sized strata, they should be split ~50/50 in the first half
    assert abs(official_count - table_count) <= 1


def test_process_dataset_splits_nesting_and_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # Create a dummy duplicate_report.json
        (tmp_dir / "reports").mkdir(parents=True)
        dup_report = {
            "exact_meaningful_text_duplicates": [["p_001", "p_002"]],
            "exact_full_page_duplicates": [],
            "near_full_page_duplicates": [],
            "structural_layout_clusters_concentration": {},
        }
        (tmp_dir / "reports" / "duplicate_report.json").write_text(
            json.dumps(dup_report), encoding="utf-8"
        )

        # Create 100 manifest rows
        manifest_rows = []
        for i in range(100):
            # Alternate some parameters
            layout = "lecture_notes_page" if i % 2 == 0 else "simple_table_page"
            lang = "kk" if i % 3 == 0 else "ky"
            quality = "clean" if i % 5 == 0 else "corrupted"

            row = _create_dummy_row(
                page_id=f"p_{i:03d}",
                layout_id=layout,
                language_mix=lang,
                quality_profile=quality,
                content_lineage_id=f"lin_{i // 5}" if i > 2 else None,
            )
            # Add metadata_groups
            row["metadata_groups"] = {
                "layout": {"layout_id": layout, "orientation": "portrait"},
                "release": {},
            }
            manifest_rows.append(row)

        # Write dummy manifest.jsonl and metadata.jsonl
        manifest_path = tmp_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in manifest_rows:
                disk_row = {
                    **row,
                    "preservation_marker": {
                        "full_only": True,
                        "payload": ["glyph-trace"] * 10,
                    },
                }
                f.write(json.dumps(disk_row, ensure_ascii=False) + "\n")

        metadata_path = tmp_dir / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(
                    json.dumps(
                        {
                            "page_id": row["page_id"],
                            "layout_id": row["layout_id"],
                            "language_mix": row["language_mix"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # Run process_dataset_splits
        # We pass manifest_rows directly, which will be updated in-place and written to disk
        process_dataset_splits(tmp_dir, manifest_rows)

        # Assert output files exist
        assert (tmp_dir / "train_manifest.jsonl").exists()
        assert (tmp_dir / "val_manifest.jsonl").exists()
        assert (tmp_dir / "test_manifest.jsonl").exists()
        assert (tmp_dir / "tiny_index.json").exists()
        assert (tmp_dir / "medium_index.json").exists()
        assert (tmp_dir / "reports" / "leakage_report.json").exists()

        # Load updated manifest
        updated_rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
        ]

        # Verify split, nested_rank, tiny/medium/large membership, and subsets.
        p1 = updated_rows[0]
        assert "split" in p1
        assert "nested_rank" in p1
        assert "in_tiny" in p1
        assert "in_medium" in p1
        assert "in_large" in p1
        assert "in_compact_25k" not in p1
        assert "in_standard_100k" not in p1
        assert "subsets" in p1
        assert isinstance(p1["subsets"], list)
        assert p1["preservation_marker"]["full_only"] is True
        assert "preservation_marker" not in manifest_rows[0]
        for row in updated_rows:
            subsets = row["subsets"]
            assert subsets == [
                name for name in ("tiny", "medium", "large") if name in subsets
            ]
            assert row["in_large"] is True
            assert (not row["in_tiny"]) or row["in_medium"]
            assert (not row["in_medium"]) or row["in_large"]
            assert ("tiny" in subsets) == row["in_tiny"]
            assert ("medium" in subsets) == row["in_medium"]
            assert ("large" in subsets) == row["in_large"]

        # Load indexes to verify nesting property
        tiny_index = json.loads(
            (tmp_dir / "tiny_index.json").read_text(encoding="utf-8")
        )
        medium_index = json.loads(
            (tmp_dir / "medium_index.json").read_text(encoding="utf-8")
        )

        tiny_pids = set(tiny_index["page_ids"])
        medium_pids = set(medium_index["page_ids"])

        # Tiny (10%) must be a subset of Medium (40%)
        assert tiny_pids.issubset(medium_pids)

        # Check sizes: 100 total, 90 train, 5 val, 5 test
        # Nested targets use exact largest-remainder allocation from the master size.
        assert len(tiny_pids) == 10
        assert len(medium_pids) == 40

        # Verify zero leakage report asserts no overlaps
        leakage_report = json.loads(
            (tmp_dir / "reports" / "leakage_report.json").read_text(encoding="utf-8")
        )
        assert leakage_report["overlap_check"]["overlap_detected"] is False
        assert leakage_report["split_sizes"]["train"] == 90
        assert leakage_report["split_sizes"]["val"] == 5
        assert leakage_report["split_sizes"]["test"] == 5


def test_split_rewrite_failure_preserves_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from turkicdocgen.splits import _rewrite_manifests_with_split_info

    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"page_id":"page-a"}\n{"page_id":"page-b"}\n',
        encoding="utf-8",
    )
    old_split_contents = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}_manifest.jsonl"
        path.write_text(f"old-{split}\n", encoding="utf-8")
        old_split_contents[split] = path.read_text(encoding="utf-8")

    real_loads = json.loads
    calls = 0

    def fail_second_row(value: str):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated parse failure")
        return real_loads(value)

    monkeypatch.setattr("turkicdocgen.splits.json.loads", fail_second_row)

    with pytest.raises(ValueError, match="simulated parse failure"):
        _rewrite_manifests_with_split_info(
            tmp_path,
            [],
            {
                "page-a": {"split": "train"},
                "page-b": {"split": "test"},
            },
        )

    assert manifest_path.read_text(encoding="utf-8") == (
        '{"page_id":"page-a"}\n{"page_id":"page-b"}\n'
    )
    for split, expected in old_split_contents.items():
        assert (tmp_path / f"{split}_manifest.jsonl").read_text(
            encoding="utf-8"
        ) == expected
    assert not list(tmp_path.glob("*.tmp"))
