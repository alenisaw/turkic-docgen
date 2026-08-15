from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import random

import yaml

from turkicdocgen.config_validation import validate_effects_config


@dataclass(slots=True)
class EffectResult:
    image_path: str
    transformed_annotations: bool
    warnings: list[str]
    metadata: dict[str, object]


EFFECTS_CONFIG = Path(
    str(importlib.resources.files("turkicdocgen") / "configs" / "effects_profile.yaml")
)
STAMP_PHRASES = Path(
    str(
        importlib.resources.files("turkicdocgen")
        / "data"
        / "corpus"
        / "stamp_phrases.jsonl"
    )
)


@lru_cache(maxsize=1)
def _load_config() -> dict:
    return validate_effects_config(
        yaml.safe_load(EFFECTS_CONFIG.read_text(encoding="utf-8"))
    )


def _range(rng: random.Random, values: list[float | int]) -> float:
    return float(values[0]) + rng.random() * (float(values[1]) - float(values[0]))


def _int_range(rng: random.Random, values: list[float | int]) -> int:
    return rng.randint(int(values[0]), int(values[1]))


def _selected_effects(profile: dict[str, Any], rng: random.Random) -> list[str]:
    if "effects" in profile:
        return list(profile["effects"])
    selected = list(profile.get("required_effects", []))
    optional = [
        name
        for name, probability in profile.get("weighted_optional_effects", {}).items()
        if rng.random() < float(probability)
    ]
    min_optional_len = 3
    if len(selected) + len(optional) < min_optional_len and profile.get(
        "weighted_optional_effects"
    ):
        ranked = sorted(
            profile["weighted_optional_effects"].items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
        for name, _ in ranked:
            if name not in selected and name not in optional:
                optional.append(name)
            if len(selected) + len(optional) >= min_optional_len:
                break
    return selected + optional
