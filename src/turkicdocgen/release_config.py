from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseConfig:
    name: str
    target_rows: int
    description: str


TINY_CONFIG = ReleaseConfig("tiny", 25_000, "Tiny subset of 25,000 pages.")
MEDIUM_CONFIG = ReleaseConfig("medium", 50_000, "Medium subset of 50,000 pages.")
LARGE_CONFIG = ReleaseConfig("large", 100_000, "Large release of 100,000 pages.")

RELEASE_CONFIGS = (TINY_CONFIG.name, MEDIUM_CONFIG.name, LARGE_CONFIG.name)
RELEASE_CONFIG_DETAILS = {
    config.name: config for config in (TINY_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG)
}
RELEASE_CONFIG_TARGETS = {
    config.name: config.target_rows
    for config in (TINY_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG)
}
RELEASE_NESTING = {
    TINY_CONFIG.name: MEDIUM_CONFIG.name,
    MEDIUM_CONFIG.name: LARGE_CONFIG.name,
}


def normalize_release_config_name(name: str) -> str:
    return name


def normalize_release_subsets(subsets: list[str] | tuple[str, ...] | None) -> list[str]:
    if not subsets:
        return [LARGE_CONFIG.name]
    normalized: list[str] = []
    for subset in subsets:
        name = normalize_release_config_name(str(subset))
        if name in RELEASE_CONFIGS and name not in normalized:
            normalized.append(name)
    if LARGE_CONFIG.name not in normalized:
        normalized.append(LARGE_CONFIG.name)
    return [name for name in RELEASE_CONFIGS if name in normalized]
