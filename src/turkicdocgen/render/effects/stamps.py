from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from turkicdocgen.languages import canonical_language_mix

from .common import STAMP_PHRASES
from .geometry import _bbox_from_points

if TYPE_CHECKING:
    import random


@dataclass(frozen=True, slots=True)
class StampBorderOptions:
    shape: str
    size: tuple[int, int]
    ink: tuple[int, int, int, int]
    rgb: tuple[int, int, int]
    alpha: int
    border_width: int


@dataclass(frozen=True, slots=True)
class StampRenderOptions:
    alpha: int
    angle: float
    color_pick: float | None = None
    shape_index: int | None = None


BLUR_THRESHOLD = 0.15
DOUBLE_IMPRESSION_PROBABILITY = 0.18


def _stamp_record(
    language_mix: str, rng: random.Random, effect_profile: str
) -> dict[str, str]:
    language_mix = canonical_language_mix(language_mix)
    if STAMP_PHRASES.exists():
        rows: list[dict[str, str]] = []
        for line in STAMP_PHRASES.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if (
                canonical_language_mix(str(payload.get("language_mix", "")))
                == language_mix
            ):
                text = str(payload.get("text", "")).strip()
                profiles = payload.get("recommended_effect_profiles", [])
                if text and (not profiles or effect_profile in profiles):
                    rows.append(
                        {
                            "id": str(payload.get("id", "stamp")),
                            "language_mix": language_mix,
                            "text": text,
                            "style": str(payload.get("style", "rectangular_stamp")),
                        }
                    )
        if rows:
            return rng.choice(rows)
    fallback = {
        "kk": "ҚАБЫЛДАНДЫ",
        "ky": "КАБЫЛ АЛЫНДЫ",
        "ru_kk": "ҚАБЫЛДАНДЫ / ПРИНЯТО",
        "ru_ky": "КАБЫЛ АЛЫНДЫ / ПРИНЯТО",
    }.get(language_mix, "МӨР")
    return {
        "id": f"fallback_{language_mix}",
        "language_mix": language_mix,
        "text": fallback,
        "style": "rectangular_stamp",
    }


def _safe_overlay_boxes(plan: object | None) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if plan is None:
        return boxes
    for zone in getattr(plan, "zones", []):
        if getattr(zone, "zone_type", "") == "stamp" or zone.metadata.get(
            "safe_overlay"
        ):
            bbox = getattr(zone, "bbox", None)
            bbox_len = 4
            if bbox and len(bbox) == bbox_len:
                # Use contextlib.suppress instead of try-except-pass (SIM105)
                with contextlib.suppress(ValueError, TypeError):
                    boxes.append(
                        (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    )
    return boxes


def _stamp_color(
    rng: random.Random, pick: float | None = None
) -> tuple[str, tuple[int, int, int]]:
    pick = rng.random() if pick is None else pick
    red_threshold = 0.45
    blue_threshold = 0.75
    violet_threshold = 0.85
    if pick < red_threshold:
        return "red", rng.choice([(150, 30, 38), (176, 38, 45), (132, 36, 42)])
    if pick < blue_threshold:
        return "blue", rng.choice([(30, 70, 145), (25, 86, 160), (42, 65, 126)])
    if pick < violet_threshold:
        return "violet", rng.choice([(103, 48, 128), (91, 48, 118)])
    return "black_gray", rng.choice([(45, 48, 54), (72, 72, 76), (32, 38, 44)])


def _stamp_font(
    plan: object | None, size: int
) -> tuple[ImageFont.ImageFont, str | None]:
    for zone in getattr(plan, "zones", []):
        if getattr(zone, "zone_type", "") != "stamp":
            continue
        path = getattr(getattr(zone, "style", None), "font_path", None)
        if path:
            with contextlib.suppress(OSError):
                return ImageFont.truetype(path, size), path
    for candidate in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"):
        with contextlib.suppress(OSError):
            return ImageFont.truetype(candidate, size), candidate
    return ImageFont.load_default(), None


def _draw_stamp_border(
    draw: ImageDraw.ImageDraw,
    opts: StampBorderOptions,
) -> None:
    width, height = opts.size
    margin = 8
    if opts.shape == "round_seal":
        diameter = min(width, height) - margin * 2
        left = (width - diameter) // 2
        top = (height - diameter) // 2
        draw.ellipse(
            (left, top, left + diameter, top + diameter),
            outline=opts.ink,
            width=opts.border_width,
        )
        draw.ellipse(
            (left + 12, top + 12, left + diameter - 12, top + diameter - 12),
            outline=(*opts.rgb, max(30, opts.alpha - 24)),
            width=2,
        )
        return
    if opts.shape == "oval_seal":
        draw.ellipse(
            (margin, height * 0.16, width - margin, height * 0.84),
            outline=opts.ink,
            width=opts.border_width,
        )
        draw.ellipse(
            (margin + 12, height * 0.23, width - margin - 12, height * 0.77),
            outline=(*opts.rgb, max(30, opts.alpha - 24)),
            width=2,
        )
        return
    draw.rounded_rectangle(
        (margin, margin, width - margin, height - margin),
        radius=3,
        outline=opts.ink,
        width=opts.border_width,
    )
    if opts.shape == "received_date":
        draw.line(
            (margin + 8, height * 0.62, width - margin - 8, height * 0.62),
            fill=opts.ink,
            width=2,
        )


def _draw_stamp_text(
    draw: ImageDraw.ImageDraw,
    plan: object | None,
    text: str,
    size: tuple[int, int],
    ink: tuple[int, int, int, int],
) -> str | None:
    width, height = size
    font, font_path = _stamp_font(plan, max(14, min(25, height // 5)))
    lines = text.split("/") if "/" in text else [text]
    lines = [line.strip()[:30] for line in lines[:2]]
    line_height = max(17, int(getattr(font, "size", 18) * 1.15))
    start_y = max(10, (height - line_height * len(lines)) // 2)
    for index, line in enumerate(lines):
        text_box = draw.textbbox((0, 0), line, font=font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            ((width - text_width) // 2, start_y + index * line_height),
            line,
            font=font,
            fill=ink,
        )
    return font_path


def _damage_stamp(
    layer: Image.Image,
    draw: ImageDraw.ImageDraw,
    rng: random.Random,
) -> tuple[Image.Image, float, float, bool]:
    width, height = layer.size
    margin = 8
    damage = rng.uniform(0.04, 0.18)
    damage_count = max(5, int((width + height) * damage / 8))
    for _ in range(damage_count):
        px = rng.randrange(margin, max(margin + 1, width - margin))
        py = rng.randrange(margin, max(margin + 1, height - margin))
        radius = rng.randint(1, 4)
        draw.ellipse((px, py, px + radius, py + radius), fill=(0, 0, 0, 0))
    local_blur = rng.uniform(0.0, 0.45)
    if local_blur > BLUR_THRESHOLD:
        layer = layer.filter(ImageFilter.GaussianBlur(local_blur))
    double_impression = rng.random() < DOUBLE_IMPRESSION_PROBABILITY
    if double_impression:
        shifted = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        shifted.alpha_composite(layer, (rng.randint(2, 5), rng.randint(1, 4)))
        layer = Image.alpha_composite(shifted, layer)
    return layer, damage, local_blur, double_impression


def _render_stamp_layer(
    plan: object | None,
    stamp_record: dict[str, str],
    box: tuple[int, int, int, int],
    rng: random.Random,
    opts: StampRenderOptions | None = None,
    *,
    alpha: int | None = None,
    angle: float | None = None,
    color_pick: float | None = None,
    shape_index: int | None = None,
) -> tuple[Image.Image, tuple[int, int], dict[str, object]]:
    if opts is None:
        opts = StampRenderOptions(
            alpha=255 if alpha is None else alpha,
            angle=0.0 if angle is None else angle,
            color_pick=0.0 if color_pick is None else color_pick,
            shape_index=0 if shape_index is None else shape_index,
        )
    else:
        supplied = {
            "alpha": alpha,
            "angle": angle,
            "color_pick": color_pick,
            "shape_index": shape_index,
        }
        if any(value is not None for value in supplied.values()):
            raise ValueError(
                "cannot pass both 'opts' and legacy stamp render parameters"
            )
    x1, y1, x2, y2 = box
    width = max(120, x2 - x1)
    height = max(70, y2 - y1)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    color_group, rgb = _stamp_color(rng, opts.color_pick)
    ink = (*rgb, opts.alpha)
    shapes = (
        "round_seal",
        "oval_seal",
        "rectangular_approval",
        "received_date",
        "archive_copy",
    )
    shape = (
        shapes[opts.shape_index % len(shapes)]
        if opts.shape_index is not None
        else rng.choice(shapes)
    )
    border_width = rng.randint(3, 6)
    border_opts = StampBorderOptions(
        shape=shape,
        size=layer.size,
        ink=ink,
        rgb=rgb,
        alpha=opts.alpha,
        border_width=border_width,
    )
    _draw_stamp_border(draw, border_opts)
    font_path = _draw_stamp_text(draw, plan, stamp_record["text"], layer.size, ink)
    layer, damage, local_blur, double_impression = _damage_stamp(layer, draw, rng)
    rotated = layer.rotate(opts.angle, resample=Image.Resampling.BICUBIC, expand=True)
    paste_x = int((x1 + x2 - rotated.width) / 2)
    paste_y = int((y1 + y2 - rotated.height) / 2)
    polygon = [
        (paste_x, paste_y),
        (paste_x + rotated.width, paste_y),
        (paste_x + rotated.width, paste_y + rotated.height),
        (paste_x, paste_y + rotated.height),
    ]
    theta = math.radians(opts.angle)
    transform = [
        [math.cos(theta), -math.sin(theta), paste_x],
        [math.sin(theta), math.cos(theta), paste_y],
        [0.0, 0.0, 1.0],
    ]
    return (
        rotated,
        (paste_x, paste_y),
        {
            "stamp_color_group": color_group,
            "stamp_color_rgb": rgb,
            "stamp_shape": shape,
            "stamp_font": font_path,
            "stamp_scale": 1.0,
            "stamp_border_damage": damage,
            "stamp_local_blur": local_blur,
            "stamp_double_impression": double_impression,
            "stamp_transform_matrix": transform,
            "stamp_polygon": polygon,
            "stamp_bbox": _bbox_from_points(polygon, 100000, 100000),
        },
    )
