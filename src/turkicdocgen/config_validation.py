from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def require_positive_weights(weights: Any, label: str) -> None:
    mapping = require_mapping(weights, label)
    if not mapping:
        raise ValueError(f"{label} must not be empty")
    for key, value in mapping.items():
        if not isinstance(value, int | float) or value <= 0:
            raise ValueError(f"{label}.{key} must be a positive number")


REQUIRED_RANGE_SIZE = 2


def require_range(values: Any, label: str) -> None:
    if not isinstance(values, list | tuple) or len(values) != REQUIRED_RANGE_SIZE:
        raise ValueError(f"{label} must be a two-item range")
    lo, hi = values
    if not isinstance(lo, int | float) or not isinstance(hi, int | float):
        raise ValueError(f"{label} range values must be numeric")
    if lo > hi:
        raise ValueError(f"{label} range must satisfy min <= max")


def validate_dataset_profile_config(config: Any) -> dict[str, Any]:
    cfg = dict(require_mapping(config, "dataset profile config"))
    profiles = require_mapping(cfg.get("profiles"), "profiles")
    for name, profile in profiles.items():
        profile_map = require_mapping(profile, f"profiles.{name}")
        count = profile_map.get("count")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"profiles.{name}.count must be a positive integer")
    require_positive_weights(cfg.get("languages"), "languages")
    require_positive_weights(cfg.get("layouts"), "layouts")
    require_positive_weights(cfg.get("quality"), "quality")
    language_mixing = cfg.get("language_mixing")
    if language_mixing is not None:
        mixing = require_mapping(language_mixing, "language_mixing")
        if not isinstance(mixing.get("enabled"), bool):
            raise ValueError("language_mixing.enabled must be a boolean")
        distribution = require_mapping(
            mixing.get("feature_distribution"), "language_mixing.feature_distribution"
        )
        supported = {
            "field_level",
            "header_footer",
            "table_level",
            "parallel_lines",
            "abbreviation_level",
            "entity_level",
            "stamp_level",
            "section_level",
        }
        if set(distribution) != supported:
            raise ValueError(
                "language_mixing.feature_distribution must define all supported features"
            )
        require_positive_weights(distribution, "language_mixing.feature_distribution")
    return cfg


def validate_layout_specs_config(config: Any) -> dict[str, Any]:
    cfg = dict(require_mapping(config, "layout specs config"))
    layouts = require_mapping(cfg.get("layouts"), "layouts")
    for layout_id, spec in layouts.items():
        spec_map = require_mapping(spec, f"layouts.{layout_id}")
        density = require_mapping(
            spec_map.get("density", {}), f"layouts.{layout_id}.density"
        )
        for key, value in density.items():
            if isinstance(value, list | tuple):
                require_range(value, f"layouts.{layout_id}.density.{key}")
        zones = spec_map.get("zones")
        if not isinstance(zones, list) or not zones:
            raise ValueError(f"layouts.{layout_id}.zones must be a non-empty list")
    return cfg


def validate_effects_config(config: Any) -> dict[str, Any]:
    cfg = dict(require_mapping(config, "effects config"))
    profiles = require_mapping(cfg.get("profiles"), "profiles")
    parameters = require_mapping(cfg.get("parameters"), "parameters")
    for key, value in parameters.items():
        require_range(value, f"parameters.{key}")
    for name, profile in profiles.items():
        profile_map = require_mapping(profile, f"profiles.{name}")
        effects = profile_map.get("effects")
        required = profile_map.get("required_effects")
        optional = profile_map.get("weighted_optional_effects", {})
        if effects is None and not isinstance(required, list):
            raise ValueError(f"profiles.{name} must define effects or required_effects")
        if effects is not None and not isinstance(effects, list):
            raise ValueError(f"profiles.{name}.effects must be a list")
        if not isinstance(optional, Mapping):
            raise ValueError(
                f"profiles.{name}.weighted_optional_effects must be a mapping"
            )
        for effect_name, weight in optional.items():
            if not isinstance(effect_name, str) or not isinstance(weight, int | float):
                raise ValueError(
                    f"profiles.{name}.weighted_optional_effects must use numeric weights"
                )
    return cfg


def validate_qa_config(config: Any) -> dict[str, Any]:
    cfg = dict(require_mapping(config, "qa config"))
    qa = require_mapping(cfg.get("qa"), "qa")
    for key in ("min_body_font_px", "min_metadata_font_px", "min_table_font_px"):
        value = qa.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"qa.{key} must be a positive integer")
    if qa.get("min_table_font_px", 0) < 18:
        raise ValueError("qa.min_table_font_px must be at least 18")
    min_visible = require_mapping(
        qa.get("min_visible_chars", {}), "qa.min_visible_chars"
    )
    for key, value in min_visible.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"qa.min_visible_chars.{key} must be a non-negative integer"
            )
    ocr = require_mapping(qa.get("ocr", {}), "qa.ocr")
    for key in ("min_width_px", "min_height_px"):
        value = ocr.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"qa.ocr.{key} must be a positive integer")
    return cfg
