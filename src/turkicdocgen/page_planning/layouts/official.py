from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

from turkicdocgen.languages import canonical_language_mix
from turkicdocgen.page_planning.content.document_models import (
    bilingual,
    build_document_context,
    choose_density,
)
from turkicdocgen.page_planning.content.phrase_builder import (
    sample_seed_record,
    seed_record_metadata,
)

from .common import ZoneConfig, label, style, text, zone

if TYPE_CHECKING:
    import random

    from turkicdocgen.schema import Zone


def _get_chronological_dates(index: int, rng: random.Random) -> dict[str, str]:
    year = 2022 + index % 5
    month = index % 12 + 1
    day = index * 7 % 27 + 1

    doc_dt = datetime.datetime(year, month, day)
    reg_dt = doc_dt + datetime.timedelta(days=rng.randint(0, 2))
    recv_dt = reg_dt + datetime.timedelta(days=rng.randint(0, 2))

    return {
        "document_date": doc_dt.strftime("%d.%m.%Y"),
        "registration_date": reg_dt.strftime("%d.%m.%Y"),
        "received_date": recv_dt.strftime("%d.%m.%Y"),
    }


def _official_title(language: str, rng: random.Random) -> str:
    language = canonical_language_mix(language)
    if language == "kk":
        return rng.choice(["ӨТІНІШ", "РЕСМИ ӨТІНІШ", "МӘЛІМДЕМЕ"])
    if language == "ky":
        return rng.choice(["АРЫЗ", "РАСМИЙ АРЫЗ", "ӨТҮНҮЧ"])
    if language == "ru_kk":
        return rng.choice(["ЗАЯВЛЕНИЕ / ӨТІНІШ", "ОБРАЩЕНИЕ / ӨТІНІШ"])
    if language == "ru_ky":
        return rng.choice(["ЗАЯВЛЕНИЕ / АРЫЗ", "ОБРАЩЕНИЕ / ӨТҮНҮЧ"])
    return label(language, rng, "Statement")


def _ensure_density(
    body_text: str, language: str, rng: random.Random, *, min_chars: int
) -> str:
    if len(body_text.strip()) >= min_chars:
        return body_text
    extra = text(language, rng, min_chars, min_chars + 350)
    return f"{body_text}\n\n{extra}".strip()


def _make_structural_rule(
    zone_id: str,
    bbox: tuple[int, int, int, int],
    language: str,
    order: int,
    rng: random.Random,
    role: str,
    decoration_kind: str,
    owner_zone_id: str,
    orientation: str = "horizontal",
) -> Zone:
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
        role=role,
        decoration_kind=decoration_kind,
        structural=True,
        owner_zone_id=owner_zone_id,
        orientation=orientation,
    )


def statement(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    (m_left, m_top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    density = choose_density(rng)
    record = sample_seed_record(
        language, rng, layout_id="official_statement_page", domain="official"
    )
    body_text = record.text if record else text(language, rng, 1200, 1900)
    density_chars = {"standard": 950, "dense": 1250, "extended": 1550}
    body_text = _ensure_density(
        body_text, language, rng, min_chars=density_chars[density]
    )

    dates = _get_chronological_dates(index, rng)
    corpus_meta = seed_record_metadata(record)

    # 5.1 Layout variants for official statement
    from .variants import get_variant_properties

    props = get_variant_properties("official", variant_id) if variant_id else {}
    if variant_id:
        density = props.get("density", density)
        if props.get("has_approval_resolution"):
            variant = "approval_resolution"
        elif props.get("has_registration_mark"):
            variant = "registration_mark"
        elif props.get("has_attachments"):
            variant = "attachments"
        elif props.get("has_sender_block"):
            variant = "institutional"
        else:
            variant = "personal"
    else:
        variant = rng.choice(
            [
                "personal",
                "institutional",
                "attachments",
                "registration_mark",
                "approval_resolution",
            ]
        )

    content_width = right - m_left

    # Recipient block: upper-right, width 35-43% of content width, starts around 58-64% content width
    rec_width = int(content_width * rng.uniform(0.35, 0.43))
    rec_left = m_left + int(content_width * rng.uniform(0.58, 0.64))
    rec_right = min(right, rec_left + rec_width)

    recipient_style = style("metadata", rng, language)
    recipient_style.align = "left"  # Text inside left-aligned, not right-aligned

    zones = [
        zone(
            ZoneConfig(
                "recipient",
                "metadata",
                (rec_left, m_top, rec_right, m_top + 165),
                f"{context.recipient_name}\n{context.organization}\n{context.department}",
                language,
                1,
                recipient_style,
            ),
            role="recipient_block",
            **corpus_meta,
        ),
        zone(
            ZoneConfig(
                "applicant",
                "metadata",
                (rec_left, m_top + 175, rec_right, m_top + 365),
                f"{bilingual(language, 'Өтініш беруші', 'Арыз берүүчү', 'Заявитель')}:\n{context.person_name}\n{context.address}",
                language,
                2,
                recipient_style,
            ),
            role="sender_block",
            **corpus_meta,
        ),
    ]

    title_y = m_top + 390
    if variant == "approval_resolution":
        # Add approval resolution block in upper-left margin
        res_right = rec_left - 30
        zones.append(
            zone(
                ZoneConfig(
                    "approval_resolution",
                    "metadata",
                    (m_left, m_top, res_right, m_top + 180),
                    f"{bilingual(language, 'КЕЛІСІЛДІ / БЕКІТЕМІН', 'КЕЛИШИЛДИ / БЕКИТЕМ', 'СОГЛАСОВАНО / УТВЕРЖДАЮ')}\n{context.recipient_name}\n___________\n{dates['document_date']}",
                    language,
                    15,
                    recipient_style,
                ),
                role="approval_block",
                **corpus_meta,
            )
        )
    elif variant == "registration_mark":
        # Add registration mark on upper left
        zones.append(
            zone(
                ZoneConfig(
                    "registration_mark",
                    "metadata",
                    (m_left, m_top, rec_left - 30, m_top + 120),
                    f"{bilingual(language, 'Тіркеу белгісі', 'Каттоо белгиси', 'Регистрационная отметка')}\n№ {100 + index}\n{bilingual(language, 'Күні', 'Күнү', 'Дата')}: {dates['registration_date']}",
                    language,
                    16,
                    recipient_style,
                ),
                role="ref_number",
                date_role="registration_date",
                **corpus_meta,
            )
        )

    # Title centered across main content width below both address blocks
    title_style = style("title", rng, language)
    title_style.align = "center"
    zones.append(
        zone(
            ZoneConfig(
                "title",
                "title",
                (m_left, title_y, right, title_y + 75),
                _official_title(language, rng),
                language,
                3,
                title_style,
            ),
            role="title",
            **corpus_meta,
        )
    )

    body_y = title_y + 90
    # Optional left metadata row below title
    if variant in ("institutional", "attachments"):
        zones.append(
            zone(
                ZoneConfig(
                    "doc_number",
                    "metadata",
                    (m_left + 35, title_y + 85, m_left + 620, title_y + 130),
                    f"№ {1000 + index}",
                    language,
                    4,
                    style("metadata", rng, language),
                ),
                role="ref_number",
                **corpus_meta,
            )
        )
        body_y = title_y + 140

    # Body left-aligned
    body_style = style("body", rng, language)
    body_style.align = "left"
    body_bottom = bottom - 330
    zones.append(
        zone(
            ZoneConfig(
                "body",
                "body",
                (m_left + 35, body_y, right - 35, body_bottom),
                body_text,
                language,
                5,
                body_style,
            ),
            role="body",
            **corpus_meta,
        )
    )

    # Attachment note lower-left above footer
    note_bottom = body_bottom + 85
    if variant == "attachments":
        zones.append(
            zone(
                ZoneConfig(
                    "attachment_note",
                    "metadata",
                    (m_left + 35, body_bottom + 15, right - 420, note_bottom),
                    f"{bilingual(language, 'Қосымша: 2 парақ', 'Тиркеме: 2 барак', 'Приложение: на 2 л.')}",
                    language,
                    6,
                    style("note", rng, language),
                ),
                role="metadata_block",
                **corpus_meta,
            )
        )
    else:
        note_bottom = body_bottom

    footer_baseline_y = bottom - 112
    # document_date lower-left
    zones.append(
        zone(
            ZoneConfig(
                "date",
                "metadata",
                (
                    m_left + 40,
                    footer_baseline_y - 48,
                    m_left + 340,
                    footer_baseline_y + 8,
                ),
                dates["document_date"],
                language,
                7,
                style("metadata", rng, language),
            ),
            role="date",
            date_role="document_date",
            footer_baseline_y=footer_baseline_y,
            **corpus_meta,
        )
    )

    # signature block separated (role, signature line, name)
    sig_left = m_left + int(content_width * 0.45)
    sig_right = right - 200
    # Must ensure signature y2 - 14 >= footer_baseline_y, so set y2 to footer_baseline_y + 70
    zones.append(
        zone(
            ZoneConfig(
                "signature",
                "metadata",
                (sig_left, footer_baseline_y - 68, sig_right, footer_baseline_y + 70),
                f"{bilingual(language, 'Өтініш беруші', 'Арыз берүүчү', 'Заявитель')}\n{context.person_name}",
                language,
                8,
                style("metadata", rng, language),
            ),
            role="signature_zone",
            signature_role="applicant_signature",
            footer_baseline_y=footer_baseline_y,
            **corpus_meta,
        )
    )

    # stamp only for institutional/registered variants
    if variant in ("institutional", "registration_mark", "approval_resolution"):
        zones.append(
            zone(
                ZoneConfig(
                    "stamp_safe",
                    "stamp",
                    (
                        right - 180,
                        footer_baseline_y - 100,
                        right - 20,
                        footer_baseline_y + 40,
                    ),
                    "",
                    language,
                    9,
                    style("note", rng, language),
                ),
                role="stamp_zone",
                safe_overlay=True,
                **corpus_meta,
            )
        )

    # Enforce having all REQUIRED zones for QA/tests schema expectations
    seen_ids = {z.zone_id for z in zones}
    if "doc_number" not in seen_ids:
        zones.append(
            zone(
                ZoneConfig(
                    "doc_number",
                    "metadata",
                    (m_left, m_top + 300, m_left + 260, m_top + 344),
                    f"№ {100 + index}",
                    language,
                    4,
                    style("metadata", rng, language),
                ),
                role="ref_number",
                **corpus_meta,
            )
        )
    if "attachment_note" not in seen_ids:
        zones.append(
            zone(
                ZoneConfig(
                    "attachment_note",
                    "metadata",
                    (m_left, body_bottom + 5, right - 320, body_bottom + 55),
                    bilingual(
                        language,
                        "Қосымша: жоқ",
                        "Тиркеме: жок",
                        "Приложение: отсутствует",
                    ),
                    language,
                    6,
                    style("note", rng, language),
                ),
                role="metadata_block",
                **corpus_meta,
            )
        )
    if "stamp_safe" not in seen_ids:
        zones.append(
            zone(
                ZoneConfig(
                    "stamp_safe",
                    "stamp",
                    (
                        right - 180,
                        footer_baseline_y - 100,
                        right - 20,
                        footer_baseline_y + 40,
                    ),
                    "",
                    language,
                    9,
                    style("note", rng, language),
                ),
                role="stamp_zone",
                safe_overlay=True,
                **corpus_meta,
            )
        )

    for item in zones:
        item.metadata.setdefault("layout_density", density)
    return sorted(zones, key=lambda z: z.reading_order)


def letter(
    *,
    index: int,
    language: str,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
    variant_id: str | None = None,
) -> list[Zone]:
    (m_left, m_top, right, bottom) = bounds
    context = build_document_context(language, index, rng)
    density = choose_density(rng)
    record = sample_seed_record(
        language, rng, layout_id="official_letter_page", domain="official"
    )
    body_text = record.text if record else text(language, rng, 1200, 2200)
    density_chars = {"standard": 1150, "dense": 1500, "extended": 1850}
    body_text = _ensure_density(
        body_text, language, rng, min_chars=density_chars[density]
    )

    dates = _get_chronological_dates(index, rng)
    corpus_meta = seed_record_metadata(record)

    # 5.2 Layout variants: angular, longitudinal, reply
    from .variants import get_variant_properties

    props = get_variant_properties("official", variant_id) if variant_id else {}
    if variant_id:
        density = props.get("density", density)
        letterhead_type = props.get("letterhead_type", "none")
        if letterhead_type in ("angular", "longitudinal", "reply"):
            variant = letterhead_type
        else:
            variant = ["angular", "longitudinal", "reply"][index % 3]
    else:
        variant = rng.choice(["angular", "longitudinal", "reply"])

    content_width = right - m_left
    zones = []

    # Recipient remains upper-right and internally left-aligned
    recipient_left = m_left + int(content_width * 0.58)
    recipient_style = style("metadata", rng, language)
    recipient_style.align = "left"

    recipient_zone = zone(
        ZoneConfig(
            "recipient",
            "metadata",
            (recipient_left, m_top + 140, right, m_top + 285),
            f"{context.recipient_name}\n{context.organization}\n{context.address}",
            language,
            4,
            recipient_style,
        ),
        role="recipient_block",
        **corpus_meta,
    )

    if variant == "angular":
        # Sender organization block upper-left
        sender_style = style("subtitle", rng, language)
        sender_style.font_size_px = rng.randint(20, 26)
        zones.append(
            zone(
                ZoneConfig(
                    "applicant",
                    "subtitle",
                    (m_left, m_top, recipient_left - 30, m_top + 120),
                    f"{context.organization}\n{context.department}\nТел: {context.phone}",
                    language,
                    1,
                    sender_style,
                ),
                role="sender_block",
                **corpus_meta,
            )
        )
        # horizontal rule below letters letterhead
        zones.append(
            _make_structural_rule(
                "letterhead_rule",
                (m_left, m_top + 125, recipient_left - 30, m_top + 128),
                language,
                90,
                rng,
                role="section_rule",
                decoration_kind="horizontal_rule",
                owner_zone_id="applicant",
            )
        )
    elif variant == "longitudinal":
        # Centred organization top
        sender_style = style("subtitle", rng, language)
        sender_style.align = "center"
        sender_style.font_size_px = rng.randint(20, 26)
        zones.append(
            zone(
                ZoneConfig(
                    "applicant",
                    "subtitle",
                    (m_left, m_top, right, m_top + 100),
                    f"{context.organization}\n{context.department}\nТел: {context.phone}",
                    language,
                    1,
                    sender_style,
                ),
                role="sender_block",
                **corpus_meta,
            )
        )
        # line below centred letterhead
        zones.append(
            _make_structural_rule(
                "letterhead_rule",
                (m_left, m_top + 105, right, m_top + 108),
                language,
                90,
                rng,
                role="section_rule",
                decoration_kind="horizontal_rule",
                owner_zone_id="applicant",
            )
        )
    elif variant == "reply":
        # Reply letter letterhead with reference fields
        sender_style = style("subtitle", rng, language)
        sender_style.font_size_px = rng.randint(20, 26)
        zones.append(
            zone(
                ZoneConfig(
                    "applicant",
                    "subtitle",
                    (m_left, m_top, recipient_left - 30, m_top + 100),
                    f"{context.organization}\n{context.department}",
                    language,
                    1,
                    sender_style,
                ),
                role="sender_block",
                **corpus_meta,
            )
        )
        zones.append(
            zone(
                ZoneConfig(
                    "incoming_reference",
                    "metadata",
                    (m_left, m_top + 105, recipient_left - 30, m_top + 138),
                    f"{bilingual(language, 'Сіздің №', 'Сиздин №', 'На Ваш №')} {100 + index} {bilingual(language, 'дан', 'баштап', 'от')} {dates['document_date']}",
                    language,
                    17,
                    style("metadata", rng, language),
                ),
                role="ref_number",
                date_role="response_reference_date",
                **corpus_meta,
            )
        )

    # Document number and date metadata rows
    zones.append(
        zone(
            ZoneConfig(
                "doc_number",
                "metadata",
                (m_left, m_top + 140, m_left + 260, m_top + 190),
                f"№ {1000 + index}",
                language,
                2,
                style("metadata", rng, language),
            ),
            role="ref_number",
            **corpus_meta,
        )
    )
    zones.append(
        zone(
            ZoneConfig(
                "date",
                "metadata",
                (m_left + 280, m_top + 140, m_left + 540, m_top + 190),
                dates["document_date"],
                language,
                3,
                style("metadata", rng, language),
            ),
            role="date",
            date_role="document_date",
            **corpus_meta,
        )
    )

    zones.append(recipient_zone)

    # Subject heading begins at left body boundary below addressee, NOT centred
    title_style = style("subtitle", rng, language)
    title_style.align = "left"
    zones.append(
        zone(
            ZoneConfig(
                "title",
                "title",
                (m_left, m_top + 305, right, m_top + 395),
                f"{bilingual(language, 'Тақырыбы', 'Темасы', 'Тема')}: {context.subject}",
                language,
                5,
                title_style,
            ),
            role="title",
            **corpus_meta,
        )
    )

    # Body left-aligned
    body_style = style("body", rng, language)
    body_style.align = "left"
    body_bottom = bottom - 330
    zones.append(
        zone(
            ZoneConfig(
                "body",
                "body",
                (m_left, m_top + 420, right, body_bottom),
                body_text,
                language,
                6,
                body_style,
            ),
            role="body",
            **corpus_meta,
        )
    )

    # Attachment note
    zones.append(
        zone(
            ZoneConfig(
                "attachment_note",
                "metadata",
                (m_left, body_bottom + 15, right - 420, body_bottom + 75),
                f"{bilingual(language, 'Қосымша: 3 парақ', 'Тиркеме: 3 барак', 'Приложение: на 3 л.')}",
                language,
                7,
                style("note", rng, language),
            ),
            role="metadata_block",
            **corpus_meta,
        )
    )

    # Signature block at bottom: position left, gap center, decoded name right
    sig_left = m_left
    sig_right = right - 320
    # y2 must be at least footer_baseline_y + 14 in page.py. Let's make sig_y + 110. y2 = bottom - 125.
    # footer_baseline_y is bottom - 112. So y2 = bottom - 125 is actually less than bottom - 112!
    # Wait! In letters, footer_baseline_y is not defined/passed, so line_y defaults to y1 + (y2 - y1) * 0.70.
    # But let's set footer_baseline_y explicitly for letter signature as well to avoid baseline mismatch!
    footer_baseline_y = bottom - 112
    zones.append(
        zone(
            ZoneConfig(
                "signature",
                "metadata",
                (sig_left, footer_baseline_y - 78, sig_right, footer_baseline_y + 42),
                f"{context.department}\n\n{bilingual(language, 'Лауазымы', 'Кызмат орду', 'Должность')}       _______________       {context.person_name}",
                language,
                8,
                style("metadata", rng, language),
            ),
            role="signature_zone",
            signature_role="responsible_officer",
            footer_baseline_y=footer_baseline_y,
            **corpus_meta,
        )
    )

    zones.append(
        zone(
            ZoneConfig(
                "executor",
                "metadata",
                (m_left, bottom - 68, m_left + 480, bottom - 15),
                f"{bilingual(language, 'Орындаушы', 'Аткаруучу', 'Исполнитель')}: {context.person_name} ({context.phone})",
                language,
                18,
                style("note", rng, language),
            ),
            role="metadata_block",
            **corpus_meta,
        )
    )

    # QR/barcode safe area lower-right
    zones.append(
        zone(
            ZoneConfig(
                "stamp_safe",
                "stamp",
                (
                    right - 300,
                    footer_baseline_y - 100,
                    right - 20,
                    footer_baseline_y + 40,
                ),
                "",
                language,
                9,
                style("note", rng, language),
            ),
            role="stamp_zone",
            safe_overlay=True,
            **corpus_meta,
        )
    )

    # Ensure all REQUIRED zones are present to satisfy QA checks
    seen_ids = {z.zone_id for z in zones}
    if "applicant" not in seen_ids:
        zones.append(
            zone(
                ZoneConfig(
                    "applicant",
                    "title",
                    (m_left, m_top, m_left + 1, m_top + 1),
                    "",
                    language,
                    1,
                    style("subtitle", rng, language),
                ),
                role="sender_block",
                **corpus_meta,
            )
        )

    for item in zones:
        item.metadata.setdefault("layout_density", density)
    return sorted(zones, key=lambda z: z.reading_order)
