from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from turkicdocgen.page_planning.content.document_models import (
    bilingual,
    build_document_context,
    choose_density,
)

from .common import ZoneConfig, style, text, zone

if TYPE_CHECKING:
    import random

    from turkicdocgen.schema import Zone


def single_column(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    from .variants import get_variant_properties

    props = get_variant_properties("book", variant_id) if variant_id else {}
    density = props.get("density", choose_density(rng))
    has_header = props.get("has_header", False)
    has_page_num = props.get("has_page_num", True)
    has_lines = props.get("has_lines", False)
    has_frame = props.get("has_frame", False)
    has_title = props.get("has_title", True)

    (m_left, m_top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    char_ranges = {
        "standard": (3200, 3700),
        "dense": (3600, 4200),
        "extended": (4000, 4600),
    }

    zones = []
    # Frame
    if has_frame:
        zones.append(
            zone(
                ZoneConfig(
                    "page_frame",
                    "decorative_non_text",
                    (m_left - 20, m_top - 20, right + 20, bottom + 20),
                    "[frame]",
                    language,
                    100,
                    style("note", rng, language),
                )
            )
        )
    # Header
    if has_header:
        zones.append(
            zone(
                ZoneConfig(
                    "running_header",
                    "metadata",
                    (m_left, m_top - 60, right, m_top - 20),
                    context.organization,
                    language,
                    101,
                    style("note", rng, language),
                )
            )
        )
    # Separator Line
    if has_lines:
        zones.append(
            zone(
                ZoneConfig(
                    "header_line",
                    "decorative_non_text",
                    (m_left, m_top - 10, right, m_top - 5),
                    "---",
                    language,
                    102,
                    style("note", rng, language),
                )
            )
        )

    # Title
    t_bottom = m_top
    b_bottom = bottom - 40
    if has_title:
        zones.append(
            zone(
                ZoneConfig(
                    "title",
                    "title",
                    (m_left, m_top, right, m_top + 78),
                    context.subject.capitalize(),
                    language,
                    1,
                    style("title", rng, language),
                )
            )
        )
        t_bottom = m_top + 115
    else:
        t_bottom = m_top + 20
        b_bottom = bottom - 135

    # Body
    zones.append(
        zone(
            ZoneConfig(
                "body",
                "body",
                (m_left, t_bottom, right, b_bottom),
                text(language, rng, *char_ranges[density]),
                language,
                2,
                style("body", rng, language),
            )
        )
    )

    # Page number
    if has_page_num:
        zones.append(
            zone(
                ZoneConfig(
                    "page_number",
                    "metadata",
                    (760, bottom, 900, bottom + 34),
                    str(index + 1),
                    language,
                    3,
                    style("note", rng, language),
                )
            )
        )

    for item in zones:
        item.metadata.setdefault("layout_density", density)
    return zones


def two_columns(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    from .variants import get_variant_properties

    props = get_variant_properties("book", variant_id) if variant_id else {}
    density = props.get("density", choose_density(rng))
    has_header = props.get("has_header", False)
    has_page_num = props.get("has_page_num", True)
    has_lines = props.get("has_lines", False)
    has_frame = props.get("has_frame", False)
    has_title = props.get("has_title", True)

    (m_left, m_top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    gutter = rng.randint(60, 82)
    col_w = (right - m_left - gutter) // 2
    char_ranges = {
        "standard": (3800, 4200),
        "dense": (4000, 4500),
        "extended": (4400, 5000),
    }
    body_text = text(language, rng, *char_ranges[density], max_paragraphs=24)
    mid = len(body_text) // 2
    space_idx = body_text.rfind(" ", 0, mid)
    cut = space_idx if space_idx != -1 else mid
    col_left_text = body_text[:cut]
    col_right_text = body_text[cut:].lstrip()
    column_style = style("body", rng, language)

    zones = []
    # Frame
    if has_frame:
        zones.append(
            zone(
                ZoneConfig(
                    "page_frame",
                    "decorative_non_text",
                    (m_left - 20, m_top - 20, right + 20, bottom + 20),
                    "[frame]",
                    language,
                    100,
                    style("note", rng, language),
                )
            )
        )
    # Header
    if has_header:
        zones.append(
            zone(
                ZoneConfig(
                    "running_header",
                    "metadata",
                    (m_left, m_top - 60, right, m_top - 20),
                    context.organization,
                    language,
                    101,
                    style("note", rng, language),
                )
            )
        )
    # Separator Line
    if has_lines:
        zones.append(
            zone(
                ZoneConfig(
                    "header_line",
                    "decorative_non_text",
                    (m_left, m_top - 10, right, m_top - 5),
                    "---",
                    language,
                    102,
                    style("note", rng, language),
                )
            )
        )

    # Title
    t_bottom = m_top
    b_bottom = bottom - 45
    if has_title:
        zones.append(
            zone(
                ZoneConfig(
                    "title",
                    "title",
                    (m_left, m_top, right, m_top + 70),
                    context.subject.capitalize(),
                    language,
                    1,
                    style("title", rng, language),
                )
            )
        )
        t_bottom = m_top + 110
    else:
        t_bottom = m_top + 20
        b_bottom = bottom - 135

    # Left Column
    zones.append(
        zone(
            ZoneConfig(
                "column_left",
                "body",
                (m_left, t_bottom, m_left + col_w, b_bottom),
                col_left_text,
                language,
                2,
                copy.deepcopy(column_style),
            )
        )
    )
    # Right Column
    zones.append(
        zone(
            ZoneConfig(
                "column_right",
                "body",
                (m_left + col_w + gutter, t_bottom, right, b_bottom),
                col_right_text,
                language,
                3,
                copy.deepcopy(column_style),
            )
        )
    )

    # Page number
    if has_page_num:
        zones.append(
            zone(
                ZoneConfig(
                    "page_number",
                    "metadata",
                    (760, bottom, 900, bottom + 34),
                    str(index + 1),
                    language,
                    4,
                    style("note", rng, language),
                )
            )
        )

    for item in zones:
        item.metadata.setdefault("layout_density", density)
    return zones


def academic_abstract(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    from .variants import get_variant_properties

    props = get_variant_properties("book", variant_id) if variant_id else {}
    density = props.get("density", choose_density(rng))
    has_header = props.get("has_header", False)
    has_page_num = props.get("has_page_num", True)
    has_lines = props.get("has_lines", False)
    has_frame = props.get("has_frame", False)
    has_title = props.get("has_title", True)

    (m_left, m_top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    char_ranges = {
        "standard": (3800, 4200),
        "dense": (4200, 4700),
        "extended": (4600, 5200),
    }

    zones = []
    # Frame
    if has_frame:
        zones.append(
            zone(
                ZoneConfig(
                    "page_frame",
                    "decorative_non_text",
                    (m_left - 20, m_top - 20, right + 20, bottom + 20),
                    "[frame]",
                    language,
                    100,
                    style("note", rng, language),
                )
            )
        )
    # Header
    if has_header:
        zones.append(
            zone(
                ZoneConfig(
                    "running_header",
                    "metadata",
                    (m_left, m_top - 60, right, m_top - 20),
                    context.organization,
                    language,
                    101,
                    style("note", rng, language),
                )
            )
        )
    # Separator Line
    if has_lines:
        zones.append(
            zone(
                ZoneConfig(
                    "header_line",
                    "decorative_non_text",
                    (m_left, m_top - 10, right, m_top - 5),
                    "---",
                    language,
                    102,
                    style("note", rng, language),
                )
            )
        )

    # Title
    t_bottom = m_top
    b_bottom = bottom - 80
    if has_title:
        zones.append(
            zone(
                ZoneConfig(
                    "title",
                    "title",
                    (m_left, m_top, right, m_top + 90),
                    context.subject.capitalize(),
                    language,
                    1,
                    style("title", rng, language),
                )
            )
        )
        t_bottom = m_top + 105
    else:
        t_bottom = m_top + 20
        b_bottom = bottom - 165

    # Authors
    zones.append(
        zone(
            ZoneConfig(
                "authors",
                "metadata",
                (m_left, t_bottom, right, t_bottom + 50),
                f"{context.person_name}, {context.organization}",
                language,
                2,
                style("metadata", rng, language),
            )
        )
    )

    # Keywords
    zones.append(
        zone(
            ZoneConfig(
                "keywords",
                "metadata",
                (m_left, t_bottom + 75, right, t_bottom + 135),
                f"{bilingual(language, 'Түйін сөздер', 'Ачкыч сөздөр', 'Ключевые слова')}: {bilingual(language, 'OCR, құжаттар, мәтінді тану', 'OCR, документтер, текстти таануу', 'OCR, документы, распознавание текста')}",
                language,
                3,
                style("metadata", rng, language),
            )
        )
    )

    # Abstract Body
    zones.append(
        zone(
            ZoneConfig(
                "abstract",
                "body",
                (m_left, t_bottom + 170, right, b_bottom),
                text(language, rng, *char_ranges[density]),
                language,
                4,
                style("body", rng, language),
            )
        )
    )

    # Page number
    if has_page_num:
        zones.append(
            zone(
                ZoneConfig(
                    "page_number",
                    "metadata",
                    (760, bottom, 900, bottom + 34),
                    str(index + 1),
                    language,
                    5,
                    style("note", rng, language),
                )
            )
        )

    for item in zones:
        item.metadata.setdefault("layout_density", density)
    return zones
