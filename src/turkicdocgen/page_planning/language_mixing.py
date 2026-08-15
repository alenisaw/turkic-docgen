from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from turkicdocgen.languages import canonical_language_mix

if TYPE_CHECKING:
    import random
    from collections.abc import Mapping

    from turkicdocgen.schema import TableCell, Zone

SUPPORTED_MIXING_FEATURES = (
    "field_level",
    "header_footer",
    "table_level",
    "parallel_lines",
    "abbreviation_level",
    "entity_level",
    "stamp_level",
    "section_level",
)

_BILINGUAL_LANGUAGE = {
    "ru_kk": "bilingual_kk_ru",
    "ru_ky": "bilingual_ky_ru",
}

_FIELD_LABELS = {
    "ru_kk": {
        "Аты": "Аты / Имя",
        "Тегі": "Тегі / Фамилия",
        "Туған күні": "Туған күні / Дата рождения",
        "ЖСН": "ЖСН / ИИН",
        "Мекенжайы": "Мекенжайы / Адрес",
        "Телефон": "Телефон / Телефон",
    },
    "ru_ky": {
        "Аты": "Аты / Имя",
        "Фамилиясы": "Фамилиясы / Фамилия",
        "Туулган күнү": "Туулган күнү / Дата рождения",
        "Жеке номер": "Жеке номер / ИИН",
        "Дареги": "Дареги / Адрес",
        "Телефон": "Телефон / Телефон",
    },
}

_TABLE_HEADERS = {
    "ru_kk": (
        "№",
        "Пән атауы / Название дисциплины",
        "Баға / Оценка",
        "Сағат саны / Количество часов",
    ),
    "ru_ky": (
        "№",
        "Сабактын аталышы / Название дисциплины",
        "Баасы / Оценка",
        "Саат саны / Количество часов",
    ),
}

_PARALLEL_LINES = {
    "ru_kk": (
        "Құжат берілген күні: 12.04.2026",
        "Дата выдачи документа: 12.04.2026",
    ),
    "ru_ky": (
        "Документ берилген күн: 12.04.2026",
        "Дата выдачи документа: 12.04.2026",
    ),
}

_HEADER_FOOTER = {
    "ru_kk": (
        "Қазақстан Республикасы",
        "Республика Казахстан",
    ),
    "ru_ky": (
        "Кыргыз Республикасы",
        "Кыргызская Республика",
    ),
}

_ENTITIES = {
    "ru_kk": "Оқу орны: Astana IT University\nҰйым: Министерство цифрового развития",
    "ru_ky": "Окуу жайы: Кыргыз мамлекеттик университети\nУюм: Министерство цифрового развития",
}

_STAMPS = {
    "ru_kk": "ҚАБЫЛДАНДЫ / ПРИНЯТО",
    "ru_ky": "КАБЫЛ АЛЫНДЫ / ПРИНЯТО",
}

_SECTIONS = {
    "ru_kk": "НЕГІЗГІ БӨЛІМ / ОСНОВНОЙ РАЗДЕЛ",
    "ru_ky": "НЕГИЗГИ БӨЛҮМ / ОСНОВНОЙ РАЗДЕЛ",
}

_ABBREVIATIONS = {
    "ru_kk": "ЖСН / ИИН    ID    QR    PDF",
    "ru_ky": "Жеке номер / ИИН    ID    QR    PDF",
}

MIN_MIXING_ZONE_HEIGHT = 70


def resolve_primary_secondary(language_mix: str) -> tuple[str, str | None]:
    language = canonical_language_mix(language_mix)
    mapping = {
        "kk": ("kk", None),
        "ky": ("ky", None),
        "ru_kk": ("kk", "ru"),
        "ru_ky": ("ky", "ru"),
    }
    return mapping[language]


def sample_mixing_features(
    language_mix: str,
    config: Mapping[str, Any] | None,
    rng: random.Random,
) -> list[str]:
    language = canonical_language_mix(language_mix)
    if language not in _BILINGUAL_LANGUAGE or not config or not config.get("enabled"):
        return []
    distribution = config.get("feature_distribution", {})
    weighted = [
        (name, float(distribution.get(name, 0.0)))
        for name in SUPPORTED_MIXING_FEATURES
        if float(distribution.get(name, 0.0)) > 0
    ]
    if not weighted:
        return []
    feature_count = rng.choices((1, 2, 3), weights=(0.55, 0.30, 0.15), k=1)[0]
    selected: list[str] = []
    available = weighted[:]
    for _ in range(min(feature_count, len(available))):
        total = sum(weight for _, weight in available)
        pick = rng.random() * total
        cumulative = 0.0
        for index, (name, weight) in enumerate(available):
            cumulative += weight
            if pick <= cumulative:
                selected.append(name)
                available.pop(index)
                break
    return selected


def _mark_zone(zone: Zone, language_mix: str, feature: str) -> None:
    zone.language = _BILINGUAL_LANGUAGE[language_mix]
    features = zone.metadata.setdefault("mixing_features", [])
    if feature not in features:
        features.append(feature)
    zone.metadata.setdefault("mixing_feature", feature)


def _mark_cell(cell: TableCell, language_mix: str, feature: str) -> None:
    cell.language = _BILINGUAL_LANGUAGE[language_mix]
    features = cell.metadata.setdefault("mixing_features", [])
    if feature not in features:
        features.append(feature)
    cell.metadata.setdefault("mixing_feature", feature)


def _first_zone(
    zones: list[Zone],
    *,
    zone_types: set[str] | None = None,
    roles: set[str] | None = None,
    require_unmixed: bool = False,
) -> Zone | None:
    for zone in zones:
        if require_unmixed and (
            zone.metadata.get("mixing_feature") or zone.metadata.get("mixing_features")
        ):
            continue
        if zone_types and zone.zone_type in zone_types:
            return zone
        if roles and str(zone.metadata.get("role", "")) in roles:
            return zone
    return None


def _apply_field_level(zones: list[Zone], language_mix: str) -> bool:
    target = _first_zone(zones, zone_types={"form"})
    if target is None:
        return False
    labels = _FIELD_LABELS[language_mix]
    lines = []
    changed = False
    for line in target.text.splitlines():
        if ":" not in line:
            lines.append(line)
            continue
        label, value = line.split(":", 1)
        replacement = labels.get(label.strip())
        if replacement is None and label.strip() and not label.startswith("["):
            replacement = f"{label.strip()} / Поле"
        lines.append(f"{replacement or label}: {value.strip()}")
        changed = changed or replacement is not None
    if not changed:
        return False
    target.text = "\n".join(lines)
    target.metadata["field_labels_mixed"] = True
    _mark_zone(target, language_mix, "field_level")
    return True


def _apply_table_level(zones: list[Zone], language_mix: str) -> bool:
    target = _first_zone(zones, zone_types={"table"})
    if target is None:
        return False
    headers = [cell for cell in target.cells if cell.row == 0]
    if not headers:
        return False
    values = _TABLE_HEADERS[language_mix]
    for index, cell in enumerate(headers):
        cell.text = values[index % len(values)]
        _mark_cell(cell, language_mix, "table_level")
    target.metadata["table_headers_mixed"] = True
    _mark_zone(target, language_mix, "table_level")
    return True


def _eligible_metadata_zone(zones: list[Zone]) -> Zone | None:
    return next(
        (
            zone
            for zone in zones
            if zone.zone_type == "metadata"
            and not zone.metadata.get("date_role")
            and not zone.metadata.get("signature_role")
            and not zone.metadata.get("mixing_feature")
            and not zone.metadata.get("mixing_features")
            and zone.bbox[3] - zone.bbox[1] >= MIN_MIXING_ZONE_HEIGHT
        ),
        None,
    )


def _finish_text_feature(target: Zone | None, language_mix: str, feature: str) -> bool:
    if target is None:
        return False
    _mark_zone(target, language_mix, feature)
    return True


def _mark_existing_bilingual_content(
    zones: list[Zone],
    language_mix: str,
) -> bool:
    def has_bilingual_separator(text: str) -> bool:
        for left, right in zip(text.split(" / "), text.split(" / ")[1:], strict=False):
            if (
                len(re.findall(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]", left)) >= 3
                and len(re.findall(r"[А-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]", right)) >= 3
            ):
                return True
        return False

    for zone in zones:
        if has_bilingual_separator(zone.text):
            _mark_zone(zone, language_mix, "section_level")
            return True
        for cell in zone.cells:
            if has_bilingual_separator(cell.text):
                _mark_cell(cell, language_mix, "table_level")
                _mark_zone(zone, language_mix, "table_level")
                return True
    return False


def _apply_header_footer(zones: list[Zone], language_mix: str) -> bool:
    target = _first_zone(
        zones,
        roles={"agency_header", "footer", "table_footer"},
        require_unmixed=True,
    ) or _first_zone(zones, zone_types={"header", "footer"}, require_unmixed=True)
    if target is not None:
        target.text = " / ".join(_HEADER_FOOTER[language_mix])
    return _finish_text_feature(target, language_mix, "header_footer")


def _apply_parallel_lines(zones: list[Zone], language_mix: str) -> bool:
    target = _eligible_metadata_zone(zones)
    if target is not None:
        date_match = re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", target.text)
        if date_match is None:
            return False
        date = date_match.group(0)
        target.text = "\n".join(
            re.sub(r"\b\d{2}\.\d{2}\.\d{4}\b", date, line)
            for line in _PARALLEL_LINES[language_mix]
        )
    return _finish_text_feature(target, language_mix, "parallel_lines")


def _replace_form_abbreviation(target: Zone, language_mix: str) -> bool:
    lines = target.text.splitlines()
    for index, line in enumerate(lines):
        if ":" in line and not line.startswith("["):
            _, value = line.split(":", 1)
            lines[index] = f"{_ABBREVIATIONS[language_mix]}: {value.strip()}"
            target.text = "\n".join(lines)
            return True
    return False


def _apply_abbreviation_level(zones: list[Zone], language_mix: str) -> bool:
    target = _first_zone(zones, zone_types={"form"}, require_unmixed=True)
    if target is not None and not _replace_form_abbreviation(target, language_mix):
        target = None
    if target is None:
        target = _eligible_metadata_zone(zones)
        if target is not None:
            target.text = f"{target.text}\n{_ABBREVIATIONS[language_mix]}".strip()
    return _finish_text_feature(target, language_mix, "abbreviation_level")


def _apply_entity_level(zones: list[Zone], language_mix: str) -> bool:
    target = _first_zone(
        zones,
        roles={"agency_header", "recipient_block", "sender_block"},
        require_unmixed=True,
    ) or _eligible_metadata_zone(zones)
    if target is not None:
        target.text = _ENTITIES[language_mix]
    return _finish_text_feature(target, language_mix, "entity_level")


def _apply_stamp_level(zones: list[Zone], language_mix: str) -> bool:
    target = _first_zone(zones, zone_types={"stamp"})
    if target is not None:
        target.text = _STAMPS[language_mix]
        target.metadata["stamp_text"] = target.text
    return _finish_text_feature(target, language_mix, "stamp_level")


def _apply_section_level(zones: list[Zone], language_mix: str) -> bool:
    target = _first_zone(zones, zone_types={"form"}, require_unmixed=True)
    if target is not None:
        lines = target.text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("[") and line.endswith("]"):
                lines[index] = f"[{_SECTIONS[language_mix]}]"
                target.text = "\n".join(lines)
                target.metadata["section_heading_mixed"] = True
                return _finish_text_feature(target, language_mix, "section_level")

    target = _first_zone(
        zones,
        roles={"agenda", "decisions", "outline", "summary", "subject"},
        require_unmixed=True,
    ) or _first_zone(zones, zone_types={"subtitle"}, require_unmixed=True)
    if target is not None:
        target.text = _SECTIONS[language_mix]
        return _finish_text_feature(target, language_mix, "section_level")

    target = max(
        (
            zone
            for zone in zones
            if zone.zone_type in {"body", "paragraph"}
            and not zone.metadata.get("mixing_feature")
            and not zone.metadata.get("mixing_features")
        ),
        key=lambda zone: (zone.bbox[2] - zone.bbox[0]) * (zone.bbox[3] - zone.bbox[1]),
        default=None,
    )
    if target is not None:
        target.text = f"{_SECTIONS[language_mix]}\n{target.text}"
    return _finish_text_feature(target, language_mix, "section_level")


_FEATURE_HANDLERS = {
    "field_level": _apply_field_level,
    "header_footer": _apply_header_footer,
    "table_level": _apply_table_level,
    "parallel_lines": _apply_parallel_lines,
    "abbreviation_level": _apply_abbreviation_level,
    "entity_level": _apply_entity_level,
    "stamp_level": _apply_stamp_level,
    "section_level": _apply_section_level,
}
FALLBACK_MIXING_FEATURES = (
    "field_level",
    "table_level",
    "parallel_lines",
    "section_level",
    "entity_level",
    "header_footer",
    "abbreviation_level",
)


def attach_mixing_metadata(
    zones: list[Zone],
    language_mix: str,
    features: list[str],
) -> list[str]:
    language = canonical_language_mix(language_mix)
    if language not in _BILINGUAL_LANGUAGE:
        return []
    applied: list[str] = []
    for feature in features:
        handler = _FEATURE_HANDLERS.get(feature)
        if handler is not None and handler(zones, language):
            applied.append(feature)
    if not applied:
        if _mark_existing_bilingual_content(zones, language):
            return ["section_level"]
        for fallback in FALLBACK_MIXING_FEATURES:
            if _FEATURE_HANDLERS[fallback](zones, language):
                applied.append(fallback)
                break
    return applied


def _latin_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]", text)
    if not letters:
        return 0.0
    return sum(letter.isascii() for letter in letters) / len(letters)


def estimate_language_mix_ratio(
    zones: list[Zone],
    language_mix: str | None = None,
) -> dict[str, float]:
    primary, secondary = resolve_primary_secondary(language_mix or "kk")
    if secondary is None:
        return {primary: 1.0, "ru": 0.0, "en": 0.0}

    primary_weight = 0.0
    secondary_weight = 0.0
    english_weight = 0.0
    for zone in zones:
        items: list[tuple[str, str]] = [(zone.text, zone.language)]
        items.extend((cell.text, cell.language) for cell in zone.cells)
        for text, language in items:
            length = max(1, len(text.strip()))
            english = length * _latin_ratio(text)
            cyrillic = max(0.0, length - english)
            english_weight += english
            if language == "ru":
                secondary_weight += cyrillic
            elif str(language).startswith("bilingual_"):
                primary_weight += cyrillic * 0.68
                secondary_weight += cyrillic * 0.32
            else:
                primary_weight += cyrillic
    total = primary_weight + secondary_weight + english_weight
    if total <= 0:
        return {primary: 0.68, secondary: 0.29, "en": 0.03}
    ratios = {
        primary: primary_weight / total,
        secondary: secondary_weight / total,
        "en": english_weight / total,
    }
    rounded = {key: round(value, 3) for key, value in ratios.items()}
    rounded[primary] = round(rounded[primary] + (1.0 - sum(rounded.values())), 3)
    return rounded
