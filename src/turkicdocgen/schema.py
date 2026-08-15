from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BBox = tuple[int, int, int, int]
Point = tuple[int, int]
Polygon = list[Point]


@dataclass(slots=True)
class TextStyle:
    font_family: str
    font_size_px: int
    align: str = "left"
    line_spacing: float = 1.2
    bold: bool = False
    italic: bool = False
    font_path: str | None = None


@dataclass(slots=True)
class LineBox:
    line_id: str
    bbox: BBox
    text: str
    reading_order: int
    polygon: Polygon = field(default_factory=list)


@dataclass(slots=True)
class TableCell:
    row: int
    col: int
    bbox: BBox
    text: str
    language: str
    reading_order: int
    metadata: dict[str, Any] = field(default_factory=dict)
    polygon: Polygon = field(default_factory=list)


@dataclass(slots=True)
class Zone:
    zone_id: str
    zone_type: str
    bbox: BBox
    polygon: Polygon
    text: str
    language: str
    reading_order: int
    style: TextStyle
    lines: list[LineBox] = field(default_factory=list)
    cells: list[TableCell] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EffectSpec:
    effect_id: str
    level: str
    params: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PagePlan:
    page_id: str
    width: int
    height: int
    layout_id: str
    language_mix: str
    quality_profile: str
    zones: list[Zone]
    effects: list[EffectSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QAIssue:
    code: str
    severity: Literal["warning", "error"]
    message: str
    zone_id: str | None = None


@dataclass(slots=True)
class QAReport:
    ok: bool
    issues: list[QAIssue] = field(default_factory=list)
