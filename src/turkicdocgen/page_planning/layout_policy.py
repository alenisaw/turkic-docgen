from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

CountryPolicy = Literal["kz", "kg", "generic"]
PageOrientation = Literal["portrait", "landscape"]
HorizontalAnchor = Literal["left", "center", "right", "span"]
VerticalBand = Literal["header", "upper", "body", "lower", "footer"]


@dataclass(frozen=True, slots=True)
class ZonePlacementRule:
    role: str
    horizontal_anchor: HorizontalAnchor
    vertical_band: VerticalBand
    text_align: Literal["left", "center", "right"]
    width_ratio: tuple[float, float]
    height_ratio: tuple[float, float]
    required: bool = True


@dataclass(frozen=True, slots=True)
class LayoutPolicy:
    layout_id: str
    country: CountryPolicy
    orientation: PageOrientation
    zones: tuple[ZonePlacementRule, ...]
    decoration_profile: str
    typography_profile: str


@dataclass(frozen=True, slots=True)
class PageGeometry:
    width: int
    height: int
    margin_left: int
    margin_top: int
    margin_right: int
    margin_bottom: int

    @property
    def content_width(self) -> int:
        return self.width - self.margin_left - self.margin_right

    @property
    def content_height(self) -> int:
        return self.height - self.margin_top - self.margin_bottom


def resolve_country(language: str) -> CountryPolicy:
    if language in ("kk", "ru_kk"):
        return "kz"
    elif language in ("ky", "ru_ky"):
        return "kg"
    return "generic"


def get_page_geometry(orientation: PageOrientation) -> PageGeometry:
    if orientation == "landscape":
        return PageGeometry(
            width=2339,
            height=1654,
            margin_left=180,
            margin_top=120,
            margin_right=160,
            margin_bottom=120,
        )
    else:
        return PageGeometry(
            width=1654,
            height=2339,
            margin_left=120,
            margin_top=180,
            margin_right=120,
            margin_bottom=160,
        )


def select_orientation(layout_id: str, index: int, seed: int) -> PageOrientation:
    thresholds = {
        "schedule_table_page": 0.25,
        "registry_extract_page": 0.20,
        "invoice_like_page": 0.15,
        "catalog_entry_page": 0.15,
        "simple_table_page": 0.10,
        "exam_register_page": 0.25,
        "inventory_sheet_page": 0.20,
        "attendance_sheet_page": 0.20,
        "wide_schedule_page": 1.0,
    }
    threshold = thresholds.get(layout_id, 0.0)
    digest = hashlib.blake2b(
        f"{seed}:{index}:{layout_id}".encode(), digest_size=8
    ).digest()
    bucket = int.from_bytes(digest, "big") / 2**64
    if bucket < threshold:
        return "landscape"
    return "portrait"


OFFICIAL_POLICY_ZONES: dict[str, tuple[ZonePlacementRule, ...]] = {
    "official_statement_page": (
        ZonePlacementRule(
            "recipient_block", "right", "header", "left", (0.35, 0.48), (0.05, 0.13)
        ),
        ZonePlacementRule(
            "sender_block", "right", "upper", "left", (0.35, 0.48), (0.05, 0.12)
        ),
        ZonePlacementRule(
            "title", "center", "upper", "center", (0.42, 0.72), (0.03, 0.07)
        ),
        ZonePlacementRule("body", "span", "body", "left", (0.82, 1.0), (0.35, 0.62)),
        ZonePlacementRule(
            "document_date", "left", "lower", "left", (0.16, 0.28), (0.02, 0.05)
        ),
        ZonePlacementRule(
            "signature_zone", "right", "lower", "left", (0.28, 0.48), (0.04, 0.10)
        ),
    ),
    "official_letter_page": (
        ZonePlacementRule(
            "agency_header", "left", "header", "left", (0.42, 0.70), (0.05, 0.13)
        ),
        ZonePlacementRule(
            "recipient_block", "right", "upper", "left", (0.34, 0.48), (0.05, 0.12)
        ),
        ZonePlacementRule(
            "document_date", "left", "upper", "left", (0.16, 0.30), (0.02, 0.05)
        ),
        ZonePlacementRule(
            "title", "center", "upper", "center", (0.45, 0.78), (0.03, 0.07)
        ),
        ZonePlacementRule("body", "span", "body", "left", (0.82, 1.0), (0.38, 0.66)),
        ZonePlacementRule(
            "signature_zone", "right", "lower", "left", (0.30, 0.50), (0.04, 0.10)
        ),
    ),
}

DECORATION_PROFILES = {
    "official_statement_page": "plain_official",
    "official_letter_page": "angular_or_longitudinal_letterhead",
    "certificate_page": "certificate_open",
    "memo_page": "plain_official",
    "meeting_minutes_page": "minutes_sectioned",
    "simple_form_page": "form_grid",
    "application_form_page": "form_grid",
    "exam_sheet_page": "form_grid",
    "worksheet_page": "form_grid",
    "receipt_like_page": "form_grid",
}


def get_layout_policy(
    layout_id: str, country: CountryPolicy, orientation: PageOrientation
) -> LayoutPolicy:
    table_profile = (
        "register_table"
        if "table" in layout_id
        or layout_id
        in {
            "registry_extract_page",
            "syllabus_page",
            "catalog_entry_page",
            "invoice_like_page",
            "exam_register_page",
            "inventory_sheet_page",
            "attendance_sheet_page",
            "wide_schedule_page",
        }
        else None
    )
    return LayoutPolicy(
        layout_id=layout_id,
        country=country,
        orientation=orientation,
        zones=OFFICIAL_POLICY_ZONES.get(layout_id, ()),
        decoration_profile=DECORATION_PROFILES.get(
            layout_id, table_profile or "plain_official"
        ),
        typography_profile="table_readable" if table_profile else "official_body",
    )
