import hashlib
import importlib.resources
import random
from collections import Counter
from pathlib import Path

import pytest
import yaml

from turkicdocgen.page_planning.planner import (
    OFFICIAL_STAMP_COMPATIBLE_LAYOUTS,
    PlannerOverrides,
    _resolve_layout_lang_quality,
    build_page_plan,
    load_dataset_profile,
    quality_distribution_for_layout,
)

DATASET_PROFILE_PATH = Path(
    str(importlib.resources.files("turkicdocgen") / "configs" / "dataset_profile.yaml")
)
FLOAT_TOLERANCE = 1e-6


def test_dataset_family_profiles_exist():
    cfg = yaml.safe_load(DATASET_PROFILE_PATH.read_text(encoding="utf-8"))
    for name in [
        "visual_300",
        "internal_10k",
        "tiny_25k",
        "medium_50k",
        "large_100k",
    ]:
        assert name in cfg["profiles"]


def test_only_100k_profile_is_publication_source() -> None:
    cfg = yaml.safe_load(DATASET_PROFILE_PATH.read_text(encoding="utf-8"))
    assert cfg["profiles"]["tiny_25k"]["public"] is False
    assert cfg["profiles"]["medium_50k"]["public"] is False
    assert cfg["profiles"]["large_100k"]["public"] is True
    assert "large_250k" not in cfg["profiles"]


def test_distributions_sum_to_one():
    cfg = yaml.safe_load(DATASET_PROFILE_PATH.read_text(encoding="utf-8"))
    assert abs(sum(cfg["languages"].values()) - 1.0) < FLOAT_TOLERANCE
    assert abs(sum(cfg["layouts"].values()) - 1.0) < FLOAT_TOLERANCE
    assert abs(sum(cfg["quality"].values()) - 1.0) < FLOAT_TOLERANCE


def test_language_mix_profiles_are_turkic_or_bilingual():
    cfg = yaml.safe_load(DATASET_PROFILE_PATH.read_text(encoding="utf-8"))
    assert "ru" not in cfg["languages"]
    assert {"ru_kk", "ru_ky"}.issubset(cfg["languages"])


def test_controlled_language_mixing_distribution_is_configured():
    cfg = yaml.safe_load(DATASET_PROFILE_PATH.read_text(encoding="utf-8"))
    mixing = cfg["language_mixing"]
    assert mixing["enabled"] is True
    assert set(mixing["feature_distribution"]) == {
        "field_level",
        "header_footer",
        "table_level",
        "parallel_lines",
        "abbreviation_level",
        "entity_level",
        "stamp_level",
        "section_level",
    }
    assert abs(sum(mixing["feature_distribution"].values()) - 1.0) < FLOAT_TOLERANCE


def test_normalized_effect_profiles_are_configured():
    cfg = yaml.safe_load(DATASET_PROFILE_PATH.read_text(encoding="utf-8"))
    assert set(cfg["quality"]) == {
        "clean",
        "office_scan",
        "low_dpi_scan",
        "photocopy",
        "phone_photo",
        "old_paper",
        "official_stamped",
    }


def test_retry_preserves_distribution_dimensions() -> None:
    first = build_page_plan(17, "internal_10k", 20260613, attempt=0)
    retry = build_page_plan(17, "internal_10k", 20260613, attempt=3)

    assert retry.layout_id == first.layout_id
    assert retry.language_mix == first.language_mix
    assert retry.quality_profile == first.quality_profile


def test_conditional_official_stamp_sampling_preserves_quality_targets() -> None:
    cfg = load_dataset_profile()
    profile_cfg = cfg["profiles"]["internal_10k"]
    counts: Counter[str] = Counter()
    sample_count = 20_000

    for index in range(sample_count):
        page_id = f"internal_10k_{index:06d}"
        selection_seed = int.from_bytes(
            hashlib.sha256(f"20260613:{page_id}:0".encode()).digest()[:8],
            "big",
        )
        layout, _, quality = _resolve_layout_lang_quality(
            index,
            profile_cfg,
            cfg,
            random.Random(selection_seed),
            PlannerOverrides(),
        )
        counts[quality] += 1
        if quality == "official_stamped":
            assert layout in OFFICIAL_STAMP_COMPATIBLE_LAYOUTS

    for quality, target in cfg["quality"].items():
        assert abs(counts[quality] / sample_count - target) < 0.015


def test_conditional_quality_distributions_recover_global_targets() -> None:
    cfg = load_dataset_profile()
    recovered = Counter()
    for layout, layout_share in cfg["layouts"].items():
        for quality, conditional_share in quality_distribution_for_layout(
            cfg, layout
        ).items():
            recovered[quality] += layout_share * conditional_share

    for quality, target in cfg["quality"].items():
        assert recovered[quality] == pytest.approx(target)


def test_incompatible_official_stamp_override_is_rejected() -> None:
    cfg = load_dataset_profile()
    with pytest.raises(ValueError, match="incompatible"):
        _resolve_layout_lang_quality(
            0,
            cfg["profiles"]["internal_10k"],
            cfg,
            random.Random(1),
            PlannerOverrides(
                layout="book_page_single_column",
                effect="official_stamped",
            ),
        )
