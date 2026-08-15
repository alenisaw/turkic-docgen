from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING

from turkicdocgen.page_planning.content.document_models import (
    bilingual,
    build_document_context,
    choose_density,
)
from turkicdocgen.page_planning.content.phrase_builder import pool

from .common import ZoneConfig, style, text, zone

if TYPE_CHECKING:
    import random

    from turkicdocgen.schema import Zone

MIN_WORD_LENGTH = 4
MIN_SPECIAL_WORD_LENGTH = 5


@dataclass(frozen=True)
class SeparatorConfig:
    zone_id: str
    bbox: tuple[int, int, int, int]
    language: str
    order: int
    rng: random.Random
    orientation: str


def _split_text(value: str, parts: int) -> list[str]:
    words = value.split()
    chunk_size = max(1, len(words) // parts)
    chunks = [
        " ".join(words[index * chunk_size : (index + 1) * chunk_size])
        for index in range(parts - 1)
    ]
    chunks.append(" ".join(words[(parts - 1) * chunk_size :]))
    return chunks


def _separator(config: SeparatorConfig) -> Zone:
    zone_id = config.zone_id
    bbox = config.bbox
    language = config.language
    order = config.order
    rng = config.rng
    orientation = config.orientation
    return zone(
        ZoneConfig(
            zone_id,
            "decorative_non_text",
            bbox,
            "",
            language,
            order,
            style("note", rng, language),
        ),
        role="layout_separator",
        orientation=orientation,
        stroke_width=1,
        color=(118, 118, 118),
    )


def _reference_entries(
    language: str, rng: random.Random, *, count: int, numbered: bool
) -> list[str]:
    sentences = [item.strip().rstrip(".") for item in pool(language) if item.strip()]
    rng.shuffle(sentences)
    entries: list[str] = []
    for index, sentence in enumerate(sentences[:count]):
        words = [word.strip(".,:;!?()[]") for word in sentence.split()]
        words = [word for word in words if len(word) >= MIN_WORD_LENGTH]
        if not words:
            continue
        term = words[index % len(words)].capitalize()
        prefix = f"{index + 1}. " if numbered else ""
        entries.append(f"{prefix}{term} — {sentence}.")
    return entries


def _get_chronological_dates(index: int, rng: random.Random) -> dict[str, str]:
    year = 2022 + index % 5
    month = index % 12 + 1
    day = index * 7 % 27 + 1

    doc_dt = datetime.datetime(year, month, day)
    reg_dt = doc_dt + datetime.timedelta(days=rng.randint(0, 2))
    issue_dt = reg_dt + datetime.timedelta(days=rng.randint(0, 3))

    return {
        "document_date": doc_dt.strftime("%d.%m.%Y"),
        "registration_date": reg_dt.strftime("%d.%m.%Y"),
        "issue_date": issue_dt.strftime("%d.%m.%Y"),
        "meeting_date": doc_dt.strftime("%d.%m.%Y"),
    }


def certificate(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    (left, top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    dates = _get_chronological_dates(index, rng)

    # 5.3 Certificate placement
    title_style = style("title", rng, language)
    title_style.align = "center"

    body_style = style("body", rng, language)
    body_text = text(language, rng, 700, 1000)

    # switches to left alignment for body longer than 240 chars
    body_style.align = "left"

    recipient_style = style("subtitle", rng, language)
    # short recipient centered, long left-aligned
    recipient_style.align = "center" if len(context.person_name) < 25 else "left"

    footer_y = bottom - 115

    # issue_date role in the number/date row
    ref_text = f"№ {100 + index}      {bilingual(language, 'Берілген күні', 'Берилген күнү', 'Дата выдачи')}: {dates['issue_date']}"

    return [
        zone(
            ZoneConfig(
                "organization",
                "metadata",
                (left + 120, top, right - 120, top + 85),
                context.organization,
                language,
                1,
                style("metadata", rng, language),
            ),
            role="agency_header",
        ),
        zone(
            ZoneConfig(
                "title",
                "title",
                (left + 180, top + 155, right - 180, top + 255),
                bilingual(language, "АНЫҚТАМА", "МААЛЫМКАТ", "СПРАВКА"),
                language,
                2,
                title_style,
            ),
            role="title",
        ),
        zone(
            ZoneConfig(
                "reference",
                "metadata",
                (left + 220, top + 285, right - 220, top + 345),
                ref_text,
                language,
                3,
                style("metadata", rng, language),
            ),
            role="ref_number",
            date_role="issue_date",
        ),
        zone(
            ZoneConfig(
                "recipient",
                "subtitle",
                (left + 150, top + 400, right - 150, top + 460),
                context.person_name,
                language,
                4,
                recipient_style,
            ),
            role="recipient_name",
        ),
        zone(
            ZoneConfig(
                "body",
                "body",
                (left + 120, top + 490, right - 120, bottom - 380),
                body_text,
                language,
                5,
                body_style,
            ),
            role="body",
        ),
        zone(
            ZoneConfig(
                "purpose",
                "metadata",
                (left + 120, bottom - 365, right - 120, bottom - 300),
                f"{bilingual(language, 'Анықтама берілген жері', 'Маалымкат берилген жери', 'Справка выдана для представления в')}: {context.organization}",
                language,
                19,
                style("note", rng, language),
            ),
            role="metadata_block",
        ),
        zone(
            ZoneConfig(
                "signature",
                "metadata",
                (left + 80, bottom - 245, right - 390, footer_y + 10),
                f"{context.department}\n{context.person_name}",
                language,
                6,
                style("metadata", rng, language),
            ),
            role="signature_zone",
            signature_role="issuing_officer",
            footer_baseline_y=footer_y,
        ),
        zone(
            ZoneConfig(
                "stamp_safe",
                "stamp",
                (right - 350, bottom - 300, right - 40, bottom - 45),
                "",
                language,
                7,
                style("note", rng, language),
            ),
            role="stamp_zone",
            safe_overlay=True,
        ),
    ]


def memo(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    (left, top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    dates = _get_chronological_dates(index, rng)
    content_width = right - left

    # 5.4 Memo alignment
    return [
        zone(
            ZoneConfig(
                "sender",
                "metadata",
                (left, top, left + int(content_width * 0.45), top + 105),
                f"{context.organization}\n{context.department}",
                language,
                1,
                style("metadata", rng, language),
            ),
            role="sender_block",
        ),
        zone(
            ZoneConfig(
                "recipient",
                "metadata",
                (right - int(content_width * 0.45), top, right, top + 135),
                f"{bilingual(language, 'Кімге', 'Кимге', 'Кому')}:\n{context.recipient_name}",
                language,
                2,
                style("metadata", rng, language),
            ),
            role="recipient_block",
        ),
        zone(
            ZoneConfig(
                "title",
                "title",
                (left + 260, top + 190, right - 260, top + 270),
                bilingual(
                    language, "ҚЫЗМЕТТІК ЖАЗБА", "КЫЗМАТТЫК КАТ", "СЛУЖЕБНАЯ ЗАПИСКА"
                ),
                language,
                3,
                style("title", rng, language),
            ),
            role="title",
        ),
        zone(
            ZoneConfig(
                "subject",
                "subtitle",
                (left, top + 315, right, top + 385),
                f"{bilingual(language, 'Тақырыбы', 'Темасы', 'Тема')}: {context.subject.capitalize()}",
                language,
                4,
                style("subtitle", rng, language),
            ),
            role="subject",
        ),
        # horizontal rule between header requisites and body
        _separator(
            SeparatorConfig(
                "memo_separator",
                (left, top + 400, right, top + 404),
                language,
                92,
                rng,
                orientation="horizontal",
            )
        ),
        zone(
            ZoneConfig(
                "body",
                "body",
                (left, top + 430, right, bottom - 300),
                text(language, rng, 1350, 2100),
                language,
                5,
                style("body", rng, language),
            ),
            role="body",
        ),
        zone(
            ZoneConfig(
                "signature",
                "metadata",
                (right - 560, bottom - 245, right, bottom - 55),
                f"{bilingual(language, 'Қолы', 'Колу', 'Подпись')}: _______________\n{context.person_name}",
                language,
                6,
                style("metadata", rng, language),
            ),
            role="signature_zone",
            signature_role="author",
        ),
        zone(
            ZoneConfig(
                "date",
                "metadata",
                (left, bottom - 180, left + 300, bottom - 120),
                dates["document_date"],
                language,
                20,
                style("metadata", rng, language),
            ),
            role="date",
            date_role="document_date",
        ),
    ]


def meeting_minutes(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    (left, top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    dates = _get_chronological_dates(index, rng)
    agenda = text(language, rng, 300, 450)
    decisions = text(language, rng, 500, 800)
    return [
        zone(
            ZoneConfig(
                "organization",
                "metadata",
                (left, top, right, top + 65),
                context.organization,
                language,
                1,
                style("metadata", rng, language),
            ),
            role="agency_header",
        ),
        zone(
            ZoneConfig(
                "title",
                "title",
                (left + 180, top + 95, right - 180, top + 185),
                bilingual(language, "ХАТТАМА", "ПРОТОКОЛ", "ПРОТОКОЛ"),
                language,
                2,
                style("title", rng, language),
            ),
            role="title",
        ),
        zone(
            ZoneConfig(
                "meeting_metadata",
                "metadata",
                (left, top + 210, right, top + 295),
                f"№ {1000 + index}       {dates['meeting_date']}\n{context.department}",
                language,
                3,
                style("metadata", rng, language),
            ),
            role="meeting_metadata",
            date_role="meeting_date",
        ),
        zone(
            ZoneConfig(
                "participants",
                "body",
                (left, top + 330, right, top + 510),
                f"{bilingual(language, 'Қатысқандар', 'Катышкандар', 'Присутствовали')}: {context.person_name}; {context.recipient_name}",
                language,
                4,
                style("body", rng, language),
            ),
            role="participants",
        ),
        # section rule before agenda
        _separator(
            SeparatorConfig(
                "agenda_rule",
                (left, top + 520, right, top + 524),
                language,
                93,
                rng,
                orientation="horizontal",
            )
        ),
        zone(
            ZoneConfig(
                "agenda",
                "body",
                (left, top + 545, right, top + 890),
                f"{bilingual(language, 'Күн тәртібі', 'Күн тартиби', 'Повестка')}:\n{agenda}",
                language,
                5,
                style("body", rng, language),
            ),
            role="agenda",
        ),
        # section rule before decisions
        _separator(
            SeparatorConfig(
                "decisions_rule",
                (left, top + 905, right, top + 909),
                language,
                94,
                rng,
                orientation="horizontal",
            )
        ),
        zone(
            ZoneConfig(
                "decisions",
                "body",
                (left, top + 930, right, bottom - 335),
                f"{bilingual(language, 'Шешімдер', 'Чечимдер', 'Решения')}:\n{decisions}",
                language,
                6,
                style("body", rng, language),
            ),
            role="decisions",
        ),
        zone(
            ZoneConfig(
                "signature",
                "metadata",
                (left, bottom - 260, right, bottom - 105),
                f"{bilingual(language, 'Төраға', 'Төрага', 'Председатель')}: _______________ {context.person_name}\n"
                f"{bilingual(language, 'Хатшы', 'Секретарь', 'Секретарь')}: _______________ {context.recipient_name}",
                language,
                7,
                style("metadata", rng, language),
            ),
            role="signature_zone",
            signature_role="chair_and_secretary",
        ),
        _separator(
            SeparatorConfig(
                "header_rule",
                (left, top + 305, right, top + 309),
                language,
                90,
                rng,
                orientation="horizontal",
            )
        ),
        _separator(
            SeparatorConfig(
                "footer_rule",
                (left, bottom - 275, right, bottom - 271),
                language,
                91,
                rng,
                orientation="horizontal",
            )
        ),
    ]


def archival_notice(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    (left, top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    title_style = style("title", rng, language)
    title_style.font_size_px = min(title_style.font_size_px, 34)
    return [
        zone(
            ZoneConfig(
                "archive_code",
                "metadata",
                (left, top, left + 620, top + 85),
                f"F.{12 + index % 80} / OP.{1 + index % 9} / D.{100 + index}",
                language,
                1,
                style("metadata", rng, language),
            ),
            role="archive_reference",
        ),
        zone(
            ZoneConfig(
                "title",
                "title",
                (left + 100, top + 125, right - 100, top + 265),
                bilingual(
                    language,
                    "МҰРАҒАТТЫҚ ХАБАРЛАМА",
                    "АРХИВДИК БИЛДИРҮҮ",
                    "АРХИВНОЕ ИЗВЕЩЕНИЕ",
                ),
                language,
                2,
                title_style,
            ),
            role="title",
        ),
        zone(
            ZoneConfig(
                "catalog_metadata",
                "metadata",
                (left, top + 300, right, top + 430),
                f"{context.organization}\n{context.document_number}    {context.date}",
                language,
                3,
                style("metadata", rng, language),
            ),
            role="catalog_metadata",
        ),
        zone(
            ZoneConfig(
                "body",
                "body",
                (left, top + 475, right, bottom - 260),
                text(language, rng, 1750, 2700),
                language,
                4,
                style("body", rng, language),
            ),
            role="body",
        ),
        zone(
            ZoneConfig(
                "archive_footer",
                "metadata",
                (left, bottom - 210, right, bottom - 70),
                f"{context.department}\n{context.person_name}    {context.date}",
                language,
                5,
                style("note", rng, language),
            ),
            role="archive_footer",
        ),
    ]


def lecture_notes(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    (left, top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    from .variants import get_variant_properties

    props = get_variant_properties("specialized", variant_id) if variant_id else {}
    density = props.get("density", choose_density(rng))
    body = text(language, rng, 2800, 3800)
    (first, second, third) = _split_text(body, 3)
    outline_items = [
        " ".join(chunk.split()[:18]).rstrip(".,;:") for chunk in (first, second, third)
    ]
    return [
        zone(
            ZoneConfig(
                "course",
                "metadata",
                (left, top, right, top + 55),
                f"{context.organization} · {context.department}",
                language,
                1,
                style("metadata", rng, language),
            ),
            role="running_header",
        ),
        zone(
            ZoneConfig(
                "title",
                "title",
                (left, top + 80, right, top + 165),
                context.subject.capitalize(),
                language,
                2,
                style("title", rng, language),
            ),
            role="lecture_title",
        ),
        zone(
            ZoneConfig(
                "outline",
                "body",
                (left, top + 205, right, top + 490),
                "\n".join(
                    f"{position}. {item}"
                    for (position, item) in enumerate(outline_items, start=1)
                ),
                language,
                3,
                style("body", rng, language),
            ),
            role="outline",
        ),
        zone(
            ZoneConfig(
                "notes",
                "body",
                (left, top + 530, right, bottom - 310),
                body,
                language,
                4,
                style("body", rng, language),
            ),
            role="lecture_notes",
            min_render_font_px=16,
        ),
        zone(
            ZoneConfig(
                "summary",
                "metadata",
                (left, bottom - 265, right, bottom - 85),
                text(language, rng, 180, 300),
                language,
                5,
                style("note", rng, language),
            ),
            role="summary",
            layout_density=density,
        ),
    ]


def reference_page(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    layout_id: str,
    variant_id: str | None = None,
) -> list[Zone]:
    (left, top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    gutter = 58
    col_width = (right - left - gutter) // 2
    layout_offsets = {
        "glossary_page": (190, 80),
        "dictionary_entry_page": (215, 105),
        "index_page": (165, 65),
    }
    (entries_top, bottom_space) = layout_offsets[layout_id]
    labels = {
        "glossary_page": bilingual(language, "ГЛОССАРИЙ", "ГЛОССАРИЙ", "ГЛОССАРИЙ"),
        "dictionary_entry_page": bilingual(language, "СӨЗДІК", "СӨЗДҮК", "СЛОВАРЬ"),
        "index_page": bilingual(language, "КӨРСЕТКІШ", "КӨРСӨТКҮЧ", "УКАЗАТЕЛЬ"),
    }
    if layout_id == "index_page":
        words = [
            word.strip(".,:;!?()[]").capitalize()
            for sentence in pool(language)
            for word in sentence.split()
            if len(word.strip(".,:;!?()[]")) >= MIN_SPECIAL_WORD_LENGTH
        ]
        words = list(dict.fromkeys(words))
        rng.shuffle(words)
        entries = [
            f"{word} ........ {12 + position * 3}"
            for (position, word) in enumerate(words[:84])
        ]
    else:
        entries = _reference_entries(
            language,
            rng,
            count=48,
            numbered=layout_id == "glossary_page",
        )
    split_at = (len(entries) + 1) // 2
    left_text = "\n".join(entries[:split_at])
    right_text = "\n".join(entries[split_at:])
    column_style = style("body", rng, language)
    column_style.font_size_px = max(20, column_style.font_size_px)
    column_style.line_spacing = min(column_style.line_spacing, 1.22)
    return [
        zone(
            ZoneConfig(
                "title",
                "title",
                (left, top, right, top + 85),
                labels[layout_id],
                language,
                1,
                style("title", rng, language),
            ),
            role="reference_title",
            content_schema_id=layout_id.removesuffix("_page"),
        ),
        zone(
            ZoneConfig(
                "range",
                "metadata",
                (left, top + 100, right, top + 155),
                context.subject.capitalize(),
                language,
                2,
                style("metadata", rng, language),
            ),
            role="entry_range",
        ),
        zone(
            ZoneConfig(
                "entries_left",
                "body",
                (left, top + entries_top, left + col_width, bottom - bottom_space),
                left_text,
                language,
                3,
                copy.deepcopy(column_style),
            ),
            role="reference_entries",
            min_render_font_px=20,
        ),
        _separator(
            SeparatorConfig(
                "column_rule",
                (
                    left + col_width + gutter // 2 - 1,
                    top + entries_top,
                    left + col_width + gutter // 2 + 1,
                    bottom - bottom_space,
                ),
                language,
                90,
                rng,
                orientation="vertical",
            )
        ),
        zone(
            ZoneConfig(
                "entries_right",
                "body",
                (
                    left + col_width + gutter,
                    top + entries_top,
                    right,
                    bottom - bottom_space,
                ),
                right_text,
                language,
                4,
                copy.deepcopy(column_style),
            ),
            role="reference_entries",
            min_render_font_px=20,
        ),
    ]


def historical_newspaper(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    (left, top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    gutter = 28
    width = (right - left - gutter * 2) // 3
    articles = _split_text(text(language, rng, 3900, 5100), 3)
    zones = [
        zone(
            ZoneConfig(
                "masthead",
                "title",
                (left, top, right, top + 105),
                bilingual(language, "ДАЛА ХАБАРШЫСЫ", "ЭЛ КАБАРЧЫСЫ", "ВЕСТНИК"),
                language,
                1,
                style("title", rng, language),
            ),
            role="masthead",
        ),
        zone(
            ZoneConfig(
                "issue_metadata",
                "metadata",
                (left, top + 120, right, top + 175),
                f"№ {index + 1}    {context.date}    {context.organization}",
                language,
                2,
                style("metadata", rng, language),
            ),
            role="issue_metadata",
        ),
        _separator(
            SeparatorConfig(
                "masthead_rule",
                (left, top + 111, right, top + 113),
                language,
                90,
                rng,
                orientation="horizontal",
            )
        ),
        _separator(
            SeparatorConfig(
                "issue_rule",
                (left, top + 188, right, top + 191),
                language,
                91,
                rng,
                orientation="horizontal",
            )
        ),
    ]
    column_style = style("body", rng, language)
    for column, article in enumerate(articles):
        x1 = left + column * (width + gutter)
        zones.append(
            zone(
                ZoneConfig(
                    f"column_{column + 1}",
                    "body",
                    (x1, top + 215, x1 + width, bottom - 110),
                    article,
                    language,
                    column + 3,
                    copy.deepcopy(column_style),
                ),
                role="newspaper_column",
                min_render_font_px=16,
            )
        )
    for separator_index in range(2):
        x = (
            left
            + (separator_index + 1) * width
            + separator_index * gutter
            + gutter // 2
        )
        zones.append(
            _separator(
                SeparatorConfig(
                    f"column_rule_{separator_index + 1}",
                    (x - 1, top + 215, x + 1, bottom - 110),
                    language,
                    92 + separator_index,
                    rng,
                    orientation="vertical",
                )
            )
        )
    return zones
