from __future__ import annotations

import json
import random
from dataclasses import asdict

from turkicdocgen.page_planning.layouts.registry import build_layout
from turkicdocgen.page_planning.layouts.variants import (
    LAYOUT_VARIANTS,
    choose_variant,
)


def test_registry_uniqueness_and_minimums() -> None:
    # Ensure every family has at least 12 variants, since all target >= 2%
    for family, variants in LAYOUT_VARIANTS.items():
        assert len(variants) >= 12, f"Family {family} has less than 12 variants"
        # Check uniqueness of variant IDs
        assert len(set(variants.keys())) == len(variants), (
            f"Family {family} has duplicate variant IDs"
        )


def test_structural_difference() -> None:
    # Proves each registered variant changes at least two declared properties
    for family, variants in LAYOUT_VARIANTS.items():
        vkeys = list(variants.keys())
        for i in range(len(vkeys)):
            for j in range(i + 1, len(vkeys)):
                v1 = variants[vkeys[i]]
                v2 = variants[vkeys[j]]
                # Find differing keys
                diff_count = 0
                all_keys = set(v1.keys()) | set(v2.keys())
                for k in all_keys:
                    if v1.get(k) != v2.get(k):
                        diff_count += 1
                assert diff_count >= 2, (
                    f"Variants {vkeys[i]} and {vkeys[j]} in family {family} "
                    f"differ by only {diff_count} properties (expected >= 2)"
                )


def test_deterministic_selection() -> None:
    # Check choose_variant determinism
    for family in LAYOUT_VARIANTS.keys():
        for orientation in ("portrait", "landscape"):
            v1 = choose_variant(family, orientation, seed=42)
            v2 = choose_variant(family, orientation, seed=42)
            assert v1 == v2

            v3 = choose_variant(family, orientation, seed=123)
            # Should choose something deterministically based on seed
            v4 = choose_variant(family, orientation, seed=123)
            assert v3 == v4


def test_build_layout_integration() -> None:
    rng = random.Random(42)
    # Test built zones contain content_schema_id matching variant_id
    layout_id = "book_page_single_column"
    variant_id = "book_var_03"
    zones = build_layout(
        layout_id,
        index=0,
        language="kk",
        rng=rng,
        bounds=(100, 100, 700, 1100),
        variant_id=variant_id,
    )
    assert len(zones) > 0
    for z in zones:
        assert z.metadata.get("layout_variant_id") == variant_id


def test_generic_family_variants_change_rendered_structure() -> None:
    representatives = {
        "specialized": "certificate_page",
        "reference": "glossary_page",
        "structured": "bulletin_or_newspaper_page",
    }
    bounds = (120, 180, 1534, 2179)
    for family, layout_id in representatives.items():
        fingerprints = set()
        for variant_id in LAYOUT_VARIANTS[family]:
            zones = build_layout(
                layout_id,
                index=1,
                language="kk",
                rng=random.Random(1),
                bounds=bounds,
                variant_id=variant_id,
            )
            fingerprints.add(
                json.dumps([asdict(zone) for zone in zones], sort_keys=True)
            )
        assert len(fingerprints) == len(LAYOUT_VARIANTS[family])


def test_simple_table_titles_are_instance_specific() -> None:
    titles = set()
    bounds = (120, 180, 1534, 2179)
    for index in range(20):
        zones = build_layout(
            "simple_table_page",
            index=index,
            language="kk",
            rng=random.Random(index),
            bounds=bounds,
            variant_id="table_var_01",
        )
        title = next(zone.text for zone in zones if zone.zone_type == "title")
        titles.add(title)
    assert len(titles) == 20
