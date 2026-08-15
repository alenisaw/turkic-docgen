from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import ImageDraw

from .geometry import _bbox_from_points

if TYPE_CHECKING:
    import random


@dataclass(frozen=True, slots=True)
class CursiveStrokeConfig:
    start_x: int
    signature_end: int
    baseline: int
    amplitude: int
    stroke_count: int


MIN_STROKE_WIDTH = 2
MAX_STROKE_WIDTH = 4
MIN_STROKE_POINTS = 2
MIN_POINTS_FOR_DOT = 4
DOT_PROBABILITY = 0.35
MIN_SIGNATURE_ZONE_WIDTH = 100
MIN_SIGNATURE_ZONE_HEIGHT = 34
BODY_MARK_PROBABILITY = 0.18
MIN_PEN_MARK_ZONE_WIDTH = 60
MIN_PEN_MARK_ZONE_HEIGHT = 24


def _signature_strokes(
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> tuple[str, list[list[tuple[int, int]]]]:
    x1, y1, x2, y2 = bounds
    width = x2 - x1
    height = y2 - y1
    start_x = x1 + max(18, int(width * rng.uniform(0.12, 0.24)))
    end_x = x2 - max(16, int(width * rng.uniform(0.06, 0.14)))
    baseline = y1 + int(height * rng.uniform(0.52, 0.72))
    amplitude = max(7, min(20, int(height * rng.uniform(0.18, 0.32))))
    variant = rng.choice(
        ("short_cursive", "initials_sweep", "compact_name", "slanted_signature")
    )
    span = max(1, end_x - start_x)
    length_ratio = {
        "short_cursive": (0.46, 0.62),
        "initials_sweep": (0.55, 0.72),
        "compact_name": (0.66, 0.82),
        "slanted_signature": (0.78, 0.94),
    }[variant]
    signature_end = start_x + int(span * rng.uniform(*length_ratio))
    stroke_count = 2 if variant in {"initials_sweep", "compact_name"} else 1
    cfg = CursiveStrokeConfig(
        start_x=start_x,
        signature_end=signature_end,
        baseline=baseline,
        amplitude=amplitude,
        stroke_count=stroke_count,
    )
    strokes = _cursive_strokes(rng, cfg)
    if variant in {"initials_sweep", "slanted_signature"}:
        flourish_y = min(y2 - 5, baseline + rng.randint(6, max(7, amplitude)))
        strokes.append(
            [
                (start_x + int(span * 0.10), flourish_y),
                (start_x + int(span * 0.34), flourish_y + rng.randint(-3, 2)),
                (signature_end, flourish_y + rng.randint(-2, 3)),
            ]
        )
    return variant, [
        [
            (
                max(x1, min(x2 - 1, int(point_x))),
                max(y1, min(y2 - 1, int(point_y))),
            )
            for point_x, point_y in stroke
        ]
        for stroke in strokes
    ]


def _cursive_strokes(
    rng: random.Random,
    cfg: CursiveStrokeConfig,
) -> list[list[tuple[int, int]]]:
    strokes: list[list[tuple[int, int]]] = []
    cursor = cfg.start_x
    for stroke_index in range(cfg.stroke_count):
        remaining = cfg.signature_end - cursor
        stroke_end = (
            cfg.signature_end
            if stroke_index == cfg.stroke_count - 1
            else cursor + int(remaining * rng.uniform(0.36, 0.52))
        )
        point_count = rng.randint(9, 15)
        slant = rng.uniform(-0.16, 0.10)
        points = []
        for point_index in range(point_count):
            progress = point_index / max(1, point_count - 1)
            point_x = cursor + int((stroke_end - cursor) * progress)
            wave = math.sin(progress * math.pi * rng.uniform(2.2, 3.8))
            jitter = rng.randint(
                -max(2, cfg.amplitude // 4), max(2, cfg.amplitude // 4)
            )
            point_y = cfg.baseline + int(wave * cfg.amplitude * 0.55 + jitter)
            point_y += int((progress - 0.5) * cfg.amplitude * slant)
            points.append((point_x, point_y))
        strokes.append(points)
        cursor = stroke_end + rng.randint(8, 18)
    return strokes


def _paint_signature(
    draw: ImageDraw.ImageDraw,
    strokes: list[list[tuple[int, int]]],
    rng: random.Random,
) -> tuple[tuple[int, int, int], int, int]:
    color = rng.choice(((18, 48, 142), (24, 28, 38), (72, 38, 112)))
    opacity = rng.randint(118, 182)
    stroke_width = rng.randint(MIN_STROKE_WIDTH, MAX_STROKE_WIDTH)
    for stroke in strokes:
        if len(stroke) >= MIN_STROKE_POINTS:
            draw.line(
                stroke,
                fill=(*color, opacity),
                width=stroke_width,
                joint="curve",
            )
    points = strokes[0]
    if len(points) > MIN_POINTS_FOR_DOT and rng.random() < DOT_PROBABILITY:
        dot_x, dot_y = rng.choice(points[2:-2])
        radius = rng.randint(1, 3)
        draw.ellipse(
            (dot_x - radius, dot_y - radius, dot_x + radius, dot_y + radius),
            fill=(*color, opacity),
        )
    return color, opacity, stroke_width


def _draw_signature_marks(
    draw: ImageDraw.ImageDraw,
    plan: object | None,
    rng: random.Random,
    *,
    seed: str,
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    if plan is None:
        return artifacts
    for zone in getattr(plan, "zones", []):
        if zone.metadata.get("role") != "signature_zone":
            continue
        mark_bbox = zone.metadata.get("signature_mark_bbox", zone.bbox)
        x1, y1, x2, y2 = (int(value) for value in mark_bbox)
        width = x2 - x1
        height = y2 - y1
        if width < MIN_SIGNATURE_ZONE_WIDTH or height < MIN_SIGNATURE_ZONE_HEIGHT:
            continue
        variant, strokes = _signature_strokes((x1, y1, x2, y2), rng)
        points = strokes[0]
        all_points = [point for stroke in strokes for point in stroke]
        if len(all_points) > 1:
            color, opacity, stroke_width = _paint_signature(draw, strokes, rng)
            artifacts.append(
                {
                    "type": "handwritten_signature",
                    "variant": variant,
                    "color_rgb": list(color),
                    "opacity": opacity,
                    "stroke_width": stroke_width,
                    "target_zone": zone.zone_id,
                    "polygon": points,
                    "strokes": strokes,
                    "bbox": list(_bbox_from_points(all_points, 100000, 100000)),
                    "seed": seed,
                }
            )
    return artifacts


def _draw_pen_artifacts(
    draw: ImageDraw.ImageDraw,
    plan: object | None,
    rng: random.Random,
    *,
    count: int,
    seed: str,
) -> list[dict[str, object]]:
    if plan is None:
        return []
    preferred = [
        zone
        for zone in getattr(plan, "zones", [])
        if zone.zone_id in {"note", "fields"}
        or zone.metadata.get("role") in {"table_footer", "form_fields"}
    ]
    body = [
        zone
        for zone in getattr(plan, "zones", [])
        if zone.zone_type in {"body", "paragraph"}
    ]
    artifacts: list[dict[str, object]] = []
    body_uses = 0
    colors = ((24, 58, 154), (22, 28, 38), (86, 42, 122))
    mark_types = ("underline", "check", "short_note", "strike", "flourish")
    for index in range(count):
        candidates = preferred or body
        if body and body_uses == 0 and rng.random() < BODY_MARK_PROBABILITY:
            candidates = body
            body_uses += 1
        if not candidates:
            break
        zone = rng.choice(candidates)
        x1, y1, x2, y2 = zone.bbox
        if x2 - x1 < MIN_PEN_MARK_ZONE_WIDTH or y2 - y1 < MIN_PEN_MARK_ZONE_HEIGHT:
            continue
        mark_type = rng.choice(mark_types)
        color = rng.choice(colors)
        opacity = rng.randint(68, 112)
        stroke_width = rng.randint(2, 4)
        start_x = rng.randint(x1 + 8, max(x1 + 8, x2 - 110))
        start_y = rng.randint(y1 + 8, max(y1 + 8, y2 - 16))
        points: list[tuple[int, int]]
        if mark_type == "check":
            points = [
                (start_x, start_y),
                (start_x + 14, start_y + 16),
                (start_x + 42, start_y - 18),
            ]
        elif mark_type == "short_note":
            line_end_x = min(x2 - 5, start_x + rng.randint(42, 82))
            line_end_y = start_y + rng.randint(-5, 5)
            arc_x1 = start_x
            arc_y1 = start_y - 12
            arc_x2 = min(x2 - 2, start_x + 38)
            arc_y2 = start_y + 15
            points = [
                (start_x, start_y),
                (arc_x1, arc_y1),
                (arc_x2, arc_y1),
                (arc_x2, arc_y2),
                (arc_x1, arc_y2),
                (line_end_x, line_end_y),
            ]
            draw.arc(
                (
                    arc_x1,
                    arc_y1,
                    arc_x2,
                    arc_y2,
                ),
                15,
                325,
                fill=(*color, opacity),
                width=stroke_width,
            )
        elif mark_type == "flourish":
            points = []
            x = start_x
            while x < min(x2 - 6, start_x + 150):
                points.append((x, start_y + rng.randint(-9, 9)))
                x += rng.randint(20, 36)
        else:
            max_length = max(20, min(190, x2 - start_x - 4))
            length = rng.randint(min(70, max_length), max_length)
            points = [
                (start_x, start_y),
                (start_x + length, start_y + rng.randint(-5, 5)),
            ]
        draw.line(
            points,
            fill=(*color, opacity),
            width=stroke_width,
            joint="curve",
        )
        artifacts.append(
            {
                "id": f"pen_{index:02d}",
                "type": mark_type,
                "color_rgb": list(color),
                "opacity": opacity,
                "stroke_width": stroke_width,
                "target_zone": zone.zone_id,
                "polygon": points,
                "bbox": list(_bbox_from_points(points, 100000, 100000)),
                "seed": seed,
            }
        )
    return artifacts
