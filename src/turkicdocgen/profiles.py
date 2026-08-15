from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

import yaml

from .config_validation import validate_dataset_profile_config
from .page_planning.planner import CORE_LAYOUTS

DATASET_PROFILE = Path(
    str(importlib.resources.files("turkicdocgen") / "configs" / "dataset_profile.yaml")
)


def load_profiles() -> dict[str, Any]:
    return validate_dataset_profile_config(
        yaml.safe_load(DATASET_PROFILE.read_text(encoding="utf-8"))
    )


def dataset_family() -> set[str]:
    return {
        name
        for name, profile in load_profiles()["profiles"].items()
        if not profile.get("qa_only", False)
    }


def profile_count(profile: str) -> int:
    profiles = load_profiles()["profiles"]
    if profile not in profiles:
        raise KeyError(profile)
    return int(profiles[profile]["count"])


def active_layouts() -> tuple[str, ...]:
    return CORE_LAYOUTS
