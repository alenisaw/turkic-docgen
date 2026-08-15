from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from turkicdocgen.page_planning.layout_policy import (
    get_layout_policy,
    get_page_geometry,
    resolve_country,
    select_orientation,
)


def test_resolve_country() -> None:
    assert resolve_country("kk") == "kz"
    assert resolve_country("ru_kk") == "kz"
    assert resolve_country("ky") == "kg"
    assert resolve_country("ru_ky") == "kg"
    assert resolve_country("other") == "generic"


def test_page_geometry() -> None:
    portrait = get_page_geometry("portrait")
    assert portrait.width == 1654
    assert portrait.height == 2339
    assert portrait.margin_left == 120
    assert portrait.margin_top == 180

    landscape = get_page_geometry("landscape")
    assert landscape.width == 2339
    assert landscape.height == 1654
    assert landscape.margin_left == 180
    assert landscape.margin_top == 120


def test_select_orientation_determinism() -> None:
    # Deterministic selection based on seed and index
    o1 = select_orientation("schedule_table_page", index=0, seed=42)
    o2 = select_orientation("schedule_table_page", index=0, seed=42)
    assert o1 == o2

    o3 = select_orientation("schedule_table_page", index=1, seed=42)
    o4 = select_orientation("schedule_table_page", index=1, seed=42)
    assert o3 == o4


def test_select_orientation_distributions() -> None:
    # Check that for schedule_table_page we get some landscape pages
    schedule_landscape_count = sum(
        1
        for i in range(100)
        if select_orientation("schedule_table_page", index=i, seed=123) == "landscape"
    )
    # Target range is 20-35%. With 100 samples, 15-40 is expected/reasonable range
    assert 15 <= schedule_landscape_count <= 40

    # Non-eligible layout should always be portrait
    official_landscape_count = sum(
        1
        for i in range(100)
        if select_orientation("official_statement_page", index=i, seed=123)
        == "landscape"
    )
    assert official_landscape_count == 0


def test_wide_schedule_can_be_landscape_and_policy_is_descriptive() -> None:
    orientations = {
        select_orientation("wide_schedule_page", index=index, seed=123)
        for index in range(100)
    }
    assert orientations == {"landscape"}

    policy = get_layout_policy("official_statement_page", "kz", "portrait")
    assert policy.country == "kz"
    assert policy.decoration_profile == "plain_official"
    assert {rule.role for rule in policy.zones} >= {
        "recipient_block",
        "title",
        "body",
        "signature_zone",
    }


def test_layout_specs_have_no_duplicate_layout_keys() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "turkicdocgen"
        / "configs"
        / "layout_specs.yaml"
    )
    layout_keys = re.findall(
        r"^  ([a-z][a-z0-9_]+):$",
        config_path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    duplicates = [key for key, count in Counter(layout_keys).items() if count > 1]
    assert duplicates == []
