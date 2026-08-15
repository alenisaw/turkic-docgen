from __future__ import annotations

import hashlib
import importlib.resources
import random
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from turkicdocgen.config_validation import validate_dataset_profile_config
from turkicdocgen.languages import canonical_language_mix
from turkicdocgen.schema import EffectSpec, PagePlan, Zone

from .language_mixing import (
    attach_mixing_metadata,
    estimate_language_mix_ratio,
    resolve_primary_secondary,
    sample_mixing_features,
)
from .layout_policy import (
    get_layout_policy,
    get_page_geometry,
    resolve_country,
    select_orientation,
)
from .layouts.common import sample_weighted
from .layouts.registry import CORE_LAYOUTS, build_layout
from .layouts.variants import choose_variant

__all__ = [
    "CORE_LAYOUTS",
    "SamplingProfile",
    "build_page_plan",
    "resolve_profile",
    "PlannerOverrides",
]


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    name: str
    count: int
    seed: int


DATASET_PROFILE = Path(
    str(importlib.resources.files("turkicdocgen") / "configs" / "dataset_profile.yaml")
)
TITLE_VARIANT_LAYOUTS = {
    "academic_abstract_page",
    "application_form_page",
    "book_page_single_column",
    "book_page_two_columns",
    "bulletin_or_newspaper_page",
    "catalog_entry_page",
    "certificate_page",
    "dictionary_entry_page",
    "exam_sheet_page",
    "historical_newspaper_page",
    "index_page",
    "invoice_like_page",
    "lecture_notes_page",
    "meeting_minutes_page",
    "memo_page",
    "official_letter_page",
    "official_statement_page",
    "receipt_like_page",
    "registry_extract_page",
    "schedule_table_page",
    "simple_form_page",
    "syllabus_page",
    "worksheet_page",
    "archival_notice_page",
    "exam_register_page",
    "inventory_sheet_page",
    "attendance_sheet_page",
    "wide_schedule_page",
    "glossary_page",
}

LAYOUT_FAMILIES = {
    "book_page_single_column": "book",
    "book_page_two_columns": "book",
    "academic_abstract_page": "book",
    "official_statement_page": "official",
    "official_letter_page": "official",
    "simple_form_page": "form",
    "application_form_page": "form",
    "exam_sheet_page": "form",
    "worksheet_page": "form",
    "receipt_like_page": "form",
    "simple_table_page": "table",
    "registry_extract_page": "table",
    "syllabus_page": "table",
    "catalog_entry_page": "table",
    "invoice_like_page": "table",
    "schedule_table_page": "table",
    "exam_register_page": "table",
    "inventory_sheet_page": "table",
    "attendance_sheet_page": "table",
    "wide_schedule_page": "table",
    "certificate_page": "specialized",
    "memo_page": "specialized",
    "meeting_minutes_page": "specialized",
    "lecture_notes_page": "specialized",
    "archival_notice_page": "specialized",
    "historical_newspaper_page": "specialized",
    "glossary_page": "reference",
    "dictionary_entry_page": "reference",
    "index_page": "reference",
    "bulletin_or_newspaper_page": "structured",
}
OFFICIAL_STAMP_COMPATIBLE_LAYOUTS = {
    "official_statement_page",
    "official_letter_page",
    "simple_form_page",
    "application_form_page",
    "certificate_page",
    "memo_page",
    "meeting_minutes_page",
    "registry_extract_page",
    "invoice_like_page",
    "receipt_like_page",
    "schedule_table_page",
    "exam_register_page",
    "inventory_sheet_page",
    "attendance_sheet_page",
    "wide_schedule_page",
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_dataset_profile() -> dict[str, Any]:
    return validate_dataset_profile_config(load_yaml(DATASET_PROFILE))


def resolve_profile(name: str, seed: int | None = None) -> SamplingProfile:
    cfg = load_dataset_profile()
    profiles = cfg.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"unknown dataset profile: {name}")
    return SamplingProfile(
        name=name,
        count=int(profiles[name]["count"]),
        seed=int(
            seed
            if seed is not None
            else cfg.get("seed_policy", {}).get("default_seed", 42)
        ),
    )


@dataclass(frozen=True, slots=True)
class PlannerOverrides:
    language: str | None = None
    layout: str | None = None
    effect: str | None = None


def quality_distribution_for_layout(
    cfg: dict[str, Any], layout: str
) -> dict[str, float]:
    quality_weights = {name: float(weight) for name, weight in cfg["quality"].items()}
    official_target = quality_weights.pop("official_stamped", 0.0)
    compatible_share = sum(
        float(weight)
        for candidate, weight in cfg["layouts"].items()
        if candidate in OFFICIAL_STAMP_COMPATIBLE_LAYOUTS
    )
    non_official_total = 1.0 - official_target
    if layout in OFFICIAL_STAMP_COMPATIBLE_LAYOUTS and official_target > 0:
        official_given_compatible = min(1.0, official_target / compatible_share)
        other_scale = (
            (1.0 - official_given_compatible) / non_official_total
            if non_official_total
            else 0.0
        )
        quality_weights = {
            name: weight * other_scale for name, weight in quality_weights.items()
        }
        quality_weights["official_stamped"] = official_given_compatible
    elif non_official_total:
        quality_weights = {
            name: weight / non_official_total
            for name, weight in quality_weights.items()
        }
    return quality_weights


def _resolve_layout_lang_quality(
    index: int,
    profile_cfg: dict[str, Any],
    cfg: dict[str, Any],
    rng: random.Random,
    overrides: PlannerOverrides,
) -> tuple[str, str, str]:
    stratified = bool(profile_cfg.get("stratified_layouts"))
    per_layout = max(1, int(profile_cfg.get("count", 160)) // len(CORE_LAYOUTS))
    layout = overrides.layout or (
        CORE_LAYOUTS[(index // per_layout) % len(CORE_LAYOUTS)]
        if stratified
        else sample_weighted(rng, cfg["layouts"])
    )
    if layout not in CORE_LAYOUTS:
        raise ValueError(f"unknown dataset layout override: {layout}")
    language = canonical_language_mix(
        overrides.language
        or (
            ("kk", "ky", "ru_kk", "ru_ky")[index % 4]
            if stratified
            else sample_weighted(rng, cfg["languages"])
        )
    )
    if language not in {"kk", "ky", "ru_kk", "ru_ky"}:
        raise ValueError(f"unknown language override: {language}")
    if overrides.effect:
        quality = overrides.effect
    elif stratified:
        quality = tuple(cfg["quality"])[index % len(cfg["quality"])]
    else:
        quality = sample_weighted(rng, quality_distribution_for_layout(cfg, layout))
    if quality not in cfg["quality"]:
        raise ValueError(f"unknown effect override: {quality}")
    if (
        quality == "official_stamped"
        and layout not in OFFICIAL_STAMP_COMPATIBLE_LAYOUTS
    ):
        if overrides.effect:
            raise ValueError(
                f"effect official_stamped is incompatible with layout {layout}"
            )
        quality_weights = {
            name: float(weight)
            for name, weight in cfg["quality"].items()
            if name != "official_stamped"
        }
        quality = sample_weighted(rng, quality_weights)
    return layout, language, quality


def _extract_date_roles(zones: list[Zone]) -> dict[str, str]:
    date_roles: dict[str, str] = {}
    for zone in zones:
        if zone.metadata.get("date_role"):
            date_roles[zone.zone_id] = str(zone.metadata["date_role"])
        nested_roles = zone.metadata.get("date_roles")
        if isinstance(nested_roles, dict):
            date_roles.update(
                {
                    f"{zone.zone_id}.{key}": str(value)
                    for key, value in nested_roles.items()
                }
            )
        elif isinstance(nested_roles, list):
            date_roles.update(
                {
                    f"{zone.zone_id}.{index}": str(value)
                    for index, value in enumerate(nested_roles)
                }
            )
    return date_roles


def _diversify_repeated_titles(
    zones: list[Zone],
    layout: str,
    index: int,
) -> None:
    if layout not in TITLE_VARIANT_LAYOUTS:
        return
    variant = f"{index + 1:04d}"
    for zone in zones:
        if zone.zone_type != "title" or not zone.text.strip():
            continue
        zone.metadata["base_title"] = zone.text
        zone.metadata["title_variant_id"] = variant
        zone.text = f"{zone.text} · {variant}"


def _build_page_plan(
    index: int,
    profile_name: str,
    seed: int,
    overrides: PlannerOverrides,
    attempt: int = 0,
) -> PagePlan:
    cfg = load_dataset_profile()
    page_id = f"{profile_name}_{index:06d}"
    hash_input = f"{seed}:{page_id}:{attempt}".encode()
    sample_seed = int.from_bytes(hashlib.sha256(hash_input).digest()[:8], "big")
    rng = random.Random(sample_seed)
    selection_hash = f"{seed}:{page_id}:0".encode()
    selection_seed = int.from_bytes(hashlib.sha256(selection_hash).digest()[:8], "big")
    selection_rng = random.Random(selection_seed)
    profile_cfg = cfg.get("profiles", {}).get(profile_name, {})
    layout, language, quality = _resolve_layout_lang_quality(
        index, profile_cfg, cfg, selection_rng, overrides
    )
    orientation = select_orientation(layout, index, seed)
    geom = get_page_geometry(orientation)
    layout_policy = get_layout_policy(layout, resolve_country(language), orientation)
    width, height = geom.width, geom.height
    m_left, m_top = geom.margin_left, geom.margin_top
    right, bottom = width - geom.margin_right, height - geom.margin_bottom
    layout_seed = int.from_bytes(
        hashlib.sha256(f"{sample_seed}:layout".encode()).digest()[:8], "big"
    )
    layout_family = LAYOUT_FAMILIES.get(layout, "other")
    layout_variant_id = choose_variant(layout_family, orientation, layout_seed)
    zones = build_layout(
        layout,
        index=index,
        language=language,
        rng=rng,
        bounds=(m_left, m_top, right, bottom),
        variant_id=layout_variant_id,
    )
    primary_language, secondary_language = resolve_primary_secondary(language)
    sampled_mixing_features = sample_mixing_features(
        language, cfg.get("language_mixing"), rng
    )
    if quality != "official_stamped":
        sampled_mixing_features = [
            feature for feature in sampled_mixing_features if feature != "stamp_level"
        ]
    mixing_features = attach_mixing_metadata(zones, language, sampled_mixing_features)
    _diversify_repeated_titles(zones, layout, index)
    language_mix_ratio = estimate_language_mix_ratio(zones, language)
    layout_density = next(
        (
            str(zone.metadata["layout_density"])
            for zone in zones
            if zone.metadata.get("layout_density")
        ),
        "standard",
    )
    date_roles = _extract_date_roles(zones)
    signature_roles = {
        zone.zone_id: str(zone.metadata["signature_role"])
        for zone in zones
        if zone.metadata.get("signature_role")
    }

    layout_variant = next(
        (
            zone.metadata.get("layout_variant_id")
            for zone in zones
            if zone.metadata.get("layout_variant_id")
        ),
        next(
            (
                zone.metadata.get("content_schema_id")
                for zone in zones
                if zone.metadata.get("content_schema_id")
            ),
            "default",
        ),
    )
    content_record_ids = sorted(
        {
            str(zone.metadata["corpus_record_id"])
            for zone in zones
            if zone.metadata.get("corpus_record_id")
        }
    )
    if content_record_ids:
        content_lineage_id = "lineage_" + "_".join(content_record_ids)
    else:
        content_lineage_id = f"lineage_layout_{layout}_{page_id}"

    for zone in zones:
        if zone.text:
            zone.text = unicodedata.normalize("NFC", zone.text)
        for line in zone.lines:
            if line.text:
                line.text = unicodedata.normalize("NFC", line.text)
        for cell in zone.cells:
            if cell.text:
                cell.text = unicodedata.normalize("NFC", cell.text)

    return PagePlan(
        page_id=page_id,
        width=width,
        height=height,
        layout_id=layout,
        language_mix=language,
        quality_profile=quality,
        zones=zones,
        effects=[EffectSpec(quality, quality, {})],
        metadata={
            "profile": profile_name,
            "index": index,
            "seed": seed,
            "sample_id": page_id,
            "sample_seed": sample_seed,
            "layout_seed": layout_seed,
            "layout_family": layout_family,
            "layout_variant_id": layout_variant,
            "content_lineage_id": content_lineage_id,
            "language_mix": language,
            "primary_language": primary_language,
            "secondary_language": secondary_language,
            "mixing_features": mixing_features,
            "language_mix_ratio": language_mix_ratio,
            "orientation": orientation,
            "layout_policy": {
                "country": layout_policy.country,
                "orientation": layout_policy.orientation,
                "decoration_profile": layout_policy.decoration_profile,
                "typography_profile": layout_policy.typography_profile,
                "zone_roles": [rule.role for rule in layout_policy.zones],
            },
            "content_schema_id": next(
                (
                    zone.metadata.get("content_schema_id")
                    for zone in zones
                    if zone.metadata.get("content_schema_id")
                ),
                layout,
            ),
            "layout_variant": layout_variant,
            "layout_density": layout_density,
            "date_roles": date_roles,
            "signature_role": next(iter(signature_roles.values()), None),
            "content_record_ids": content_record_ids,
            "generator_schema_version": "2.1",
        },
    )


def build_page_plan(
    index: int,
    profile_name: str,
    seed: int,
    overrides: PlannerOverrides | None = None,
    *,
    language: str | None = None,
    layout: str | None = None,
    effect: str | None = None,
    language_override: str | None = None,
    layout_override: str | None = None,
    effect_override: str | None = None,
    attempt: int = 0,
) -> PagePlan:
    legacy_keys = {
        "language": language,
        "layout": layout,
        "effect": effect,
        "language_override": language_override,
        "layout_override": layout_override,
        "effect_override": effect_override,
    }
    has_legacy = any(v is not None for v in legacy_keys.values())
    if overrides is not None and has_legacy:
        raise ValueError(
            "Cannot specify both 'overrides' and legacy override arguments"
        )
    if overrides is None:
        overrides = PlannerOverrides(
            language=language_override or language,
            layout=layout_override or layout,
            effect=effect_override or effect,
        )
    return _build_page_plan(index, profile_name, seed, overrides, attempt=attempt)
