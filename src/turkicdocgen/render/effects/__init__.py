from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from turkicdocgen.dedup import compute_dhash, compute_full_page_dhash

from .color_noise import (
    _apply_irregular_stains,
    _apply_paper_aging,
    _apply_scanline_jitter,
    _degradation_tier,
    _draw_roller_streaks,
    _draw_toner_dropout,
    _paper_texture,
)
from .common import (
    EffectResult,
    _int_range,
    _load_config,
    _range,
    _selected_effects,
)
from .geometry import (
    _apply_perspective,
    _apply_phone_geometry,
    _bbox_from_points,
    _phone_geometry_tier,
    _transform_annotations,
    _transform_artifacts,
    _transform_point,
)
from .signatures import (
    _draw_pen_artifacts,
    _draw_signature_marks,
)
from .stamps import (
    StampRenderOptions,
    _render_stamp_layer,
    _safe_overlay_boxes,
    _stamp_record,
)


def _apply_scan_stripes(
    draw: ImageDraw.ImageDraw, width: int, height: int, count: int, rng: random.Random
) -> None:
    for _ in range(count):
        y = rng.randrange(0, height)
        draw.line(
            (0, y, width, y),
            fill=(0, 0, 0, rng.randint(8, 22)),
            width=1,
        )


def _apply_dark_printer_streaks(
    draw: ImageDraw.ImageDraw, width: int, height: int, count: int, rng: random.Random
) -> None:
    for _ in range(count):
        x = rng.randrange(80, width - 80)
        draw.line(
            (x, 80, x + rng.randint(-18, 18), height - 80),
            fill=(0, 0, 0, rng.randint(16, 34)),
            width=2,
        )


def _apply_scanner_noise(
    draw: ImageDraw.ImageDraw, width: int, height: int, count: int, rng: random.Random
) -> None:
    for _ in range(count):
        x = rng.randrange(0, width)
        y = rng.randrange(0, height)
        shade = rng.randint(0, 80)
        draw.point((x, y), fill=(shade, shade, shade, rng.randint(24, 70)))


def _apply_colored_blotches(
    draw: ImageDraw.ImageDraw, width: int, height: int, count: int, rng: random.Random
) -> None:
    for _ in range(count):
        x = rng.randrange(100, width - 180)
        y = rng.randrange(100, height - 180)
        r = rng.randint(12, 34)
        color = rng.choice([(190, 60, 60, 38), (60, 90, 180, 34), (130, 120, 30, 32)])
        draw.ellipse((x, y, x + r, y + r), fill=color)


def _apply_ink_spots(
    draw: ImageDraw.ImageDraw, width: int, height: int, count: int, rng: random.Random
) -> None:
    for _ in range(count):
        x = rng.randrange(80, width - 80)
        y = rng.randrange(80, height - 80)
        radius = rng.randint(1, 3)
        draw.ellipse(
            (x, y, x + radius, y + radius), fill=(0, 0, 0, rng.randint(26, 75))
        )


def _apply_toner_speckles(
    draw: ImageDraw.ImageDraw, width: int, height: int, count: int, rng: random.Random
) -> None:
    for _ in range(count):
        x = rng.randrange(0, width)
        y = rng.randrange(0, height)
        shade = rng.randint(0, 45)
        draw.point((x, y), fill=(shade, shade, shade, rng.randint(36, 96)))


def _apply_paper_feed_bands(
    draw: ImageDraw.ImageDraw, width: int, height: int, count: int, rng: random.Random
) -> None:
    for _ in range(count):
        y = rng.randrange(40, height - 40)
        draw.rectangle(
            (0, y, width, y + rng.randint(2, 7)),
            fill=(70, 70, 70, rng.randint(6, 18)),
        )


def _apply_print_through(image: Image.Image, rng: random.Random) -> Image.Image:
    ghost = ImageOps.mirror(image).filter(ImageFilter.GaussianBlur(1.2))
    return Image.blend(image, ghost, rng.uniform(0.025, 0.055))


def _apply_repeated_copy_erosion(
    image: Image.Image, rng: random.Random, contrast: float
) -> Image.Image:
    enhanced = ImageEnhance.Contrast(image).enhance(contrast)
    return enhanced.filter(ImageFilter.MedianFilter(3))


def _apply_edge_curl_shadow(
    image: Image.Image, rng: random.Random, side: str
) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay, "RGBA")
    if side == "bottom":
        odraw.ellipse(
            (-80, image.height - 55, image.width + 80, image.height + 45),
            fill=(0, 0, 0, 24),
        )
    elif side == "left":
        odraw.ellipse((-45, -80, 55, image.height + 80), fill=(0, 0, 0, 22))
    else:
        odraw.ellipse(
            (image.width - 55, -80, image.width + 45, image.height + 80),
            fill=(0, 0, 0, 22),
        )
    overlay = overlay.filter(ImageFilter.GaussianBlur(18))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _apply_specular_highlight(
    image: Image.Image, rng: random.Random, x: int, y: int, radius: int
) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay, "RGBA")
    odraw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(255, 255, 245, rng.randint(18, 38)),
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius // 2))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _apply_moire(
    draw: ImageDraw.ImageDraw, width: int, height: int, spacing: int
) -> None:
    for x in range(-height, width, spacing):
        draw.line((x, 0, x + height, height), fill=(20, 45, 80, 5), width=1)


def _apply_uneven_illumination(image: Image.Image) -> Image.Image:
    gradient = Image.new("L", image.size, 0)
    gdraw = ImageDraw.Draw(gradient)
    for y in range(image.height):
        alpha = int(28 * y / image.height)
        gdraw.line((0, y, image.width, y), fill=alpha)
    shade = ImageOps.colorize(gradient, (255, 255, 255), (232, 232, 224))
    return ImageChops.multiply(image, shade)


DEFAULT_STAMP_OFFSET_W1 = 420
DEFAULT_STAMP_OFFSET_H1 = 420
DEFAULT_STAMP_OFFSET_W2 = 130
DEFAULT_STAMP_OFFSET_H2 = 140

MIN_FEED_BANDS = 2
MAX_FEED_BANDS = 5

DEFAULT_PERSPECTIVE_RATIO = [0.01, 0.03]
POINT_DIMENSION = 2

PAPER_TEXTURE_BLEND = 0.08
DEFAULT_JPEG_QUALITY = 92

MIN_EROSION_CONTRAST = 1.16
MAX_EROSION_CONTRAST = 1.34

SPECULAR_W_DIVISOR = 5
SPECULAR_W_MULTIPLIER = 4
SPECULAR_H_DIVISOR = 6
SPECULAR_H_HALF = 2
MIN_SPECULAR_RADIUS = 120
MAX_SPECULAR_RADIUS = 280

MIN_MOIRE_SPACING = 5
MAX_MOIRE_SPACING = 9

EDGE_SHADOW_ALPHA = 28
EDGE_SHADOW_WIDTH = 18
COPY_BORDER_WIDTH = 12

AGED_TONE_COLOR = 0.82
AGED_TONE_BRIGHTNESS = 0.96

PHOTOCOPY_CONTRAST = 1.18
PHOTOCOPY_BRIGHTNESS = 0.96

FADED_INK_CONTRAST = 0.92
FADED_INK_BRIGHTNESS = 1.02

SCAN_FINAL_CONTRAST = 1.05
SCAN_FINAL_BRIGHTNESS = 0.985

ROUND_DECIMALS = 3
SEVERITY_DIVISOR = 10
HEAVY_DEGRADATION_PENALTY = 0.18
EXTREME_GEOMETRY_PENALTY = 0.12
DEFAULT_PAPER_BASE_RGB = [250, 250, 249]
HALF = 2.0


@dataclass(slots=True)
class _PipelineContext:
    image: Image.Image
    rng: random.Random
    selected_effects: set[str] | list[str]
    profile: dict[str, Any]
    params: dict[str, Any]
    seed_digest: str
    heavy_variant: bool
    plan: Any | None
    quality_profile: str
    rng_seed: str | int | None
    geometry_tier: str
    degradation_tier: str

    draw: ImageDraw.ImageDraw = field(init=False)
    warnings: list[str] = field(default_factory=list, init=False)
    transformed: bool = field(default=False, init=False)
    applied: list[str] = field(default_factory=list, init=False)
    effect_chain: list[dict[str, object]] = field(default_factory=list, init=False)
    exact_parameters: dict[str, object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.draw = ImageDraw.Draw(self.image, "RGBA")

    def remember(self, name: str, values: dict[str, object] | None = None) -> None:
        parameters = values or {}
        self.applied.append(name)
        self.exact_parameters[name] = parameters
        self.effect_chain.append({"effect": name, "parameters": parameters})

    def update_draw(self) -> None:
        self.draw = ImageDraw.Draw(self.image, "RGBA")


def _apply_noise_effects_part1(ctx: _PipelineContext) -> None:
    if "paper_texture_light" in ctx.selected_effects:
        ctx.image = _paper_texture(ctx.image, ctx.rng)
        ctx.update_draw()
        ctx.remember("paper_texture_light", {"blend": PAPER_TEXTURE_BLEND})

    if "scan_stripes" in ctx.selected_effects:
        count = ctx.rng.randint(*ctx.params["scan_stripes"])
        _apply_scan_stripes(ctx.draw, ctx.image.width, ctx.image.height, count, ctx.rng)
        ctx.remember("scan_stripes", {"count": count})

    if "dark_printer_streaks" in ctx.selected_effects:
        count = ctx.rng.randint(*ctx.params["printer_streaks"])
        _apply_dark_printer_streaks(
            ctx.draw, ctx.image.width, ctx.image.height, count, ctx.rng
        )
        ctx.remember("dark_printer_streaks", {"count": count})

    if "roller_streaks" in ctx.selected_effects:
        count = ctx.rng.randint(*ctx.params["roller_streaks"])
        _draw_roller_streaks(ctx.image, ctx.draw, ctx.rng, count)
        ctx.remember("roller_streaks", {"count": count})

    if "scanner_noise" in ctx.selected_effects:
        count = ctx.rng.randint(*ctx.params["scanner_noise_points"])
        _apply_scanner_noise(
            ctx.draw, ctx.image.width, ctx.image.height, count, ctx.rng
        )
        ctx.remember("scanner_noise", {"points": count})


def _apply_noise_effects_part2(ctx: _PipelineContext) -> None:
    if "scanline_jitter" in ctx.selected_effects:
        amount = _int_range(ctx.rng, ctx.params["scanline_jitter_px"])
        ctx.image = _apply_scanline_jitter(ctx.image, ctx.rng, amount)
        ctx.update_draw()
        ctx.remember("scanline_jitter", {"max_shift_px": amount})

    if "colored_blotches" in ctx.selected_effects:
        count = ctx.rng.randint(*ctx.params["color_blotches"])
        _apply_colored_blotches(
            ctx.draw, ctx.image.width, ctx.image.height, count, ctx.rng
        )
        ctx.remember("colored_blotches", {"count": count})

    if "ink_spots" in ctx.selected_effects:
        count = ctx.rng.randint(*ctx.params["ink_spots"])
        _apply_ink_spots(ctx.draw, ctx.image.width, ctx.image.height, count, ctx.rng)
        ctx.remember("ink_spots", {"count": count})

    if "toner_speckles" in ctx.selected_effects:
        count = ctx.rng.randint(*ctx.params["toner_speckles"])
        _apply_toner_speckles(
            ctx.draw, ctx.image.width, ctx.image.height, count, ctx.rng
        )
        ctx.remember("toner_speckles", {"count": count})

    if "toner_dropout" in ctx.selected_effects:
        count = ctx.rng.randint(*ctx.params["toner_dropout"])
        _draw_toner_dropout(ctx.image, ctx.draw, ctx.rng, count)
        ctx.remember("toner_dropout", {"count": count})


def _apply_stains_and_stamps(ctx: _PipelineContext) -> None:
    if "stains" in ctx.selected_effects:
        count = _int_range(ctx.rng, ctx.params["stain_count"])
        ctx.image, stain_artifacts = _apply_irregular_stains(
            ctx.image, ctx.rng, count, ctx.params["stain_alpha"]
        )
        ctx.update_draw()
        ctx.remember("stains", {"count": count, "artifacts": stain_artifacts})

    if "language_stamp" in ctx.selected_effects:
        stamp_record = _stamp_record(
            getattr(ctx.plan, "language_mix", ""), ctx.rng, ctx.quality_profile
        )
        planned_stamp = next(
            (
                zone.metadata.get("stamp_text")
                for zone in getattr(ctx.plan, "zones", [])
                if zone.zone_type == "stamp" and zone.metadata.get("stamp_text")
            ),
            None,
        )
        if planned_stamp:
            stamp_record = {
                **stamp_record,
                "id": f"{stamp_record['id']}_controlled",
                "text": str(planned_stamp),
            }
        stamp = stamp_record["text"]
        alpha = _int_range(ctx.rng, ctx.params["stamp_alpha"])
        angle_key = (
            "stamp_heavy_rotation_degrees"
            if ctx.heavy_variant
            else "stamp_rotation_degrees"
        )
        stamp_angle = _range(ctx.rng, ctx.params[angle_key])
        boxes = _safe_overlay_boxes(ctx.plan)
        if boxes:
            box = ctx.rng.choice(boxes)
        else:
            box = (
                ctx.image.width - DEFAULT_STAMP_OFFSET_W1,
                ctx.image.height - DEFAULT_STAMP_OFFSET_H1,
                ctx.image.width - DEFAULT_STAMP_OFFSET_W2,
                ctx.image.height - DEFAULT_STAMP_OFFSET_H2,
            )
            ctx.warnings.append("stamp_used_default_safe_corner")

        stamp_opts = StampRenderOptions(
            alpha=alpha,
            angle=stamp_angle,
            color_pick=(int(ctx.seed_digest[8:16], 16) % 100) / 100,
            shape_index=int(ctx.seed_digest[32:40], 16) % 5,
        )
        stamp_layer, stamp_position, stamp_details = _render_stamp_layer(
            ctx.plan,
            stamp_record,
            box,
            ctx.rng,
            stamp_opts,
        )
        ctx.image = ctx.image.convert("RGBA")
        ctx.image.alpha_composite(stamp_layer, stamp_position)
        ctx.image = ctx.image.convert("RGB")
        ctx.update_draw()
        ctx.remember(
            "language_stamp",
            {
                "stamp_id": stamp_record["id"],
                "stamp_text": stamp,
                "stamp_language_mix": stamp_record["language_mix"],
                "stamp_style": stamp_record["style"],
                "stamp_alpha": alpha,
                "stamp_rotation_degrees": stamp_angle,
                "stamp_seed": str(ctx.rng_seed),
                **stamp_details,
            },
        )


def _apply_signatures_and_marks(ctx: _PipelineContext) -> None:
    if "signature_marks" in ctx.selected_effects:
        signature_artifacts = _draw_signature_marks(
            ctx.draw, ctx.plan, ctx.rng, seed=str(ctx.rng_seed)
        )
        ctx.remember(
            "signature_marks",
            {"safe_zone_only": True, "artifacts": signature_artifacts},
        )

    if "sparse_pen_marks" in ctx.selected_effects:
        count = ctx.rng.randint(*ctx.params["pen_marks"])
        pen_artifacts = _draw_pen_artifacts(
            ctx.draw, ctx.plan, ctx.rng, count=count, seed=str(ctx.rng_seed)
        )
        ctx.remember(
            "sparse_pen_marks",
            {"count": len(pen_artifacts), "artifacts": pen_artifacts},
        )

    if "underlines_checks" in ctx.selected_effects:
        check_artifacts = _draw_pen_artifacts(
            ctx.draw, ctx.plan, ctx.rng, count=1, seed=str(ctx.rng_seed)
        )
        ctx.remember(
            "underlines_checks",
            {"count": len(check_artifacts), "artifacts": check_artifacts},
        )


def _apply_distortion_effects(ctx: _PipelineContext) -> None:
    if "paper_feed_bands" in ctx.selected_effects:
        count = ctx.rng.randint(MIN_FEED_BANDS, MAX_FEED_BANDS)
        _apply_paper_feed_bands(
            ctx.draw, ctx.image.width, ctx.image.height, count, ctx.rng
        )
        ctx.remember("paper_feed_bands", {"count": count})

    if "print_through" in ctx.selected_effects:
        ctx.image = _apply_print_through(ctx.image, ctx.rng)
        ctx.update_draw()
        ctx.remember("print_through", {"opacity": "0.025-0.055"})

    if "repeated_copy_erosion" in ctx.selected_effects:
        contrast = ctx.rng.uniform(MIN_EROSION_CONTRAST, MAX_EROSION_CONTRAST)
        ctx.image = _apply_repeated_copy_erosion(ctx.image, ctx.rng, contrast)
        ctx.update_draw()
        ctx.remember(
            "repeated_copy_erosion", {"contrast": contrast, "median_filter": 3}
        )

    if "edge_curl_shadow" in ctx.selected_effects:
        side = ctx.rng.choice(("left", "right", "bottom"))
        ctx.image = _apply_edge_curl_shadow(ctx.image, ctx.rng, side)
        ctx.update_draw()
        ctx.remember("edge_curl_shadow", {"side": side})

    if "specular_highlight" in ctx.selected_effects:
        x = ctx.rng.randint(
            ctx.image.width // SPECULAR_W_DIVISOR,
            ctx.image.width * SPECULAR_W_MULTIPLIER // SPECULAR_W_DIVISOR,
        )
        y = ctx.rng.randint(
            ctx.image.height // SPECULAR_H_DIVISOR, ctx.image.height // SPECULAR_H_HALF
        )
        radius = ctx.rng.randint(MIN_SPECULAR_RADIUS, MAX_SPECULAR_RADIUS)
        ctx.image = _apply_specular_highlight(ctx.image, ctx.rng, x, y, radius)
        ctx.update_draw()
        ctx.remember("specular_highlight", {"center": [x, y], "radius": radius})

    if "moire" in ctx.selected_effects:
        spacing = ctx.rng.randint(MIN_MOIRE_SPACING, MAX_MOIRE_SPACING)
        _apply_moire(ctx.draw, ctx.image.width, ctx.image.height, spacing)
        ctx.remember("moire", {"spacing_px": spacing})

    if "blur" in ctx.selected_effects:
        radius = _range(ctx.rng, ctx.params["blur_radius"])
        ctx.image = ctx.image.filter(ImageFilter.GaussianBlur(radius=radius))
        ctx.update_draw()
        ctx.remember("blur", {"radius": radius})

    if "downsample_upsample" in ctx.selected_effects:
        target_width = _int_range(ctx.rng, ctx.params["low_dpi_width"])
        target_height = max(1, int(ctx.image.height * target_width / ctx.image.width))
        original_size = ctx.image.size
        ctx.image = ctx.image.resize(
            (target_width, target_height), Image.Resampling.BILINEAR
        )
        ctx.image = ctx.image.resize(original_size, Image.Resampling.BICUBIC)
        ctx.update_draw()
        ctx.remember("downsample_upsample", {"target_width": target_width})


def _apply_toning_effects(ctx: _PipelineContext) -> None:
    if "uneven_illumination" in ctx.selected_effects:
        ctx.image = _apply_uneven_illumination(ctx.image)
        ctx.update_draw()
        ctx.remember("uneven_illumination", {"gradient": "vertical"})

    if "edge_shadow" in ctx.selected_effects:
        ctx.draw.rectangle(
            (0, 0, ctx.image.width - 1, ctx.image.height - 1),
            outline=(0, 0, 0, EDGE_SHADOW_ALPHA),
            width=EDGE_SHADOW_WIDTH,
        )
        ctx.remember("edge_shadow", {"width": EDGE_SHADOW_WIDTH})

    if "copy_border_shadow" in ctx.selected_effects:
        alpha = _int_range(ctx.rng, ctx.params["copy_border_alpha"])
        ctx.draw.rectangle(
            (14, 14, ctx.image.width - 15, ctx.image.height - 15),
            outline=(0, 0, 0, alpha),
            width=COPY_BORDER_WIDTH,
        )
        ctx.remember("copy_border_shadow", {"alpha": alpha})

    if "aged_tone" in ctx.selected_effects:
        ctx.image = ImageEnhance.Color(ctx.image).enhance(AGED_TONE_COLOR)
        ctx.image = ImageEnhance.Brightness(ctx.image).enhance(AGED_TONE_BRIGHTNESS)
        ctx.update_draw()
        ctx.remember(
            "aged_tone", {"color": AGED_TONE_COLOR, "brightness": AGED_TONE_BRIGHTNESS}
        )

    if "paper_aging" in ctx.selected_effects:
        ctx.image, aging = _apply_paper_aging(
            ctx.image,
            ctx.seed_digest,
            getattr(ctx.plan, "metadata", {}).get("paper_aging_variant_override"),
        )
        ctx.update_draw()
        ctx.remember("paper_aging", aging)

    if "photocopy_contrast" in ctx.selected_effects:
        ctx.image = ImageEnhance.Contrast(ctx.image).enhance(PHOTOCOPY_CONTRAST)
        ctx.image = ImageEnhance.Brightness(ctx.image).enhance(PHOTOCOPY_BRIGHTNESS)
        ctx.update_draw()
        ctx.remember(
            "photocopy_contrast",
            {"contrast": PHOTOCOPY_CONTRAST, "brightness": PHOTOCOPY_BRIGHTNESS},
        )

    if "faded_ink" in ctx.selected_effects:
        ctx.image = ImageEnhance.Contrast(ctx.image).enhance(FADED_INK_CONTRAST)
        ctx.image = ImageEnhance.Brightness(ctx.image).enhance(FADED_INK_BRIGHTNESS)
        ctx.update_draw()
        ctx.remember(
            "faded_ink",
            {"contrast": FADED_INK_CONTRAST, "brightness": FADED_INK_BRIGHTNESS},
        )


def _apply_rotation_transform(ctx: _PipelineContext) -> None:
    if (
        "light_rotation" in ctx.selected_effects
        and ctx.quality_profile != "phone_photo"
    ):
        rotation_key = {
            "office_scan": "office_scan_rotation_degrees",
            "low_dpi_scan": "low_dpi_scan_rotation_degrees",
            "photocopy": "photocopy_rotation_degrees",
            "phone_photo": "phone_photo_rotation_degrees",
        }.get(ctx.quality_profile, "rotation_degrees")
        angle = _range(ctx.rng, ctx.params[rotation_key])
        ctx.image = ctx.image.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=tuple(
                getattr(ctx.plan, "metadata", {})
                .get("paper_base", {})
                .get("rgb", DEFAULT_PAPER_BASE_RGB)
            ),
        )
        ctx.update_draw()
        ctx.transformed = True
        theta = math.radians(-angle)
        cx, cy = ctx.image.width / HALF, ctx.image.height / HALF
        forward = [
            [
                math.cos(theta),
                -math.sin(theta),
                cx - cx * math.cos(theta) + cy * math.sin(theta),
            ],
            [
                math.sin(theta),
                math.cos(theta),
                cy - cx * math.sin(theta) - cy * math.cos(theta),
            ],
            [0.0, 0.0, 1.0],
        ]
        forward_matrix = np.asarray(forward)
        if ctx.plan is not None:
            _transform_annotations(
                ctx.plan, forward_matrix, ctx.image.width, ctx.image.height
            )
        for effect_name in (
            "signature_marks",
            "sparse_pen_marks",
            "underlines_checks",
            "stains",
        ):
            effect_parameters = ctx.exact_parameters.get(effect_name, {})
            if isinstance(effect_parameters, dict):
                artifacts = effect_parameters.get("artifacts", [])
                if isinstance(artifacts, list):
                    _transform_artifacts(
                        artifacts, forward_matrix, ctx.image.width, ctx.image.height
                    )
        stamp_params = ctx.exact_parameters.get("language_stamp", {})
        if isinstance(stamp_params, dict) and "stamp_polygon" in stamp_params:
            points = stamp_params["stamp_polygon"]
            if isinstance(points, list) and points:
                transformed_pts = [
                    _transform_point(
                        (float(p[0]), float(p[1])),
                        forward_matrix,
                        ctx.image.width,
                        ctx.image.height,
                    )
                    for p in points
                    if isinstance(p, list | tuple) and len(p) == POINT_DIMENSION
                ]
                if transformed_pts:
                    stamp_params["stamp_polygon"] = transformed_pts
                    stamp_params["stamp_bbox"] = list(
                        _bbox_from_points(
                            transformed_pts, ctx.image.width, ctx.image.height
                        )
                    )
        ctx.remember(
            "light_rotation",
            {
                "angle_degrees": angle,
                "forward_matrix": forward,
                "inverse_matrix": np.linalg.inv(forward_matrix).tolist(),
            },
        )


def _apply_perspective_transform(ctx: _PipelineContext) -> None:
    if "light_perspective" in ctx.selected_effects:
        if ctx.quality_profile == "phone_photo":
            ctx.image, perspective = _apply_phone_geometry(
                ctx.image, ctx.plan, ctx.rng, ctx.params, ctx.geometry_tier
            )
        else:
            ctx.image, perspective = _apply_perspective(
                ctx.image,
                ctx.plan,
                ctx.rng,
                (
                    ctx.params["perspective_corner_offset_ratio"]
                    if ctx.heavy_variant
                    else DEFAULT_PERSPECTIVE_RATIO
                ),
            )
        forward_matrix = np.asarray(perspective["forward_matrix"])
        for effect_name in (
            "signature_marks",
            "sparse_pen_marks",
            "underlines_checks",
            "stains",
        ):
            effect_parameters = ctx.exact_parameters.get(effect_name, {})
            if isinstance(effect_parameters, dict):
                artifacts = effect_parameters.get("artifacts", [])
                if isinstance(artifacts, list):
                    _transform_artifacts(
                        artifacts, forward_matrix, ctx.image.width, ctx.image.height
                    )
        stamp_params = ctx.exact_parameters.get("language_stamp", {})
        if isinstance(stamp_params, dict) and "stamp_polygon" in stamp_params:
            points = stamp_params["stamp_polygon"]
            if isinstance(points, list) and points:
                transformed_pts = [
                    _transform_point(
                        (float(p[0]), float(p[1])),
                        forward_matrix,
                        ctx.image.width,
                        ctx.image.height,
                    )
                    for p in points
                    if isinstance(p, list | tuple) and len(p) == POINT_DIMENSION
                ]
                if transformed_pts:
                    stamp_params["stamp_polygon"] = transformed_pts
                    stamp_params["stamp_bbox"] = list(
                        _bbox_from_points(
                            transformed_pts, ctx.image.width, ctx.image.height
                        )
                    )
        ctx.update_draw()
        ctx.transformed = True
        ctx.remember("light_perspective", {"geometric_transform": True, **perspective})


def _apply_final_adjustments_and_save(
    ctx: _PipelineContext,
    path: Path,
    degradation_tier: str,
    geometry_tier: str,
) -> EffectResult:
    if ctx.quality_profile in {"office_scan", "phone_photo"}:
        ctx.image = ImageEnhance.Contrast(ctx.image).enhance(SCAN_FINAL_CONTRAST)
        ctx.image = ImageEnhance.Brightness(ctx.image).enhance(SCAN_FINAL_BRIGHTNESS)

    is_png = str(path).lower().endswith(".png")
    full_page_dhash_32 = (
        format(compute_dhash(ctx.image, hash_size=32), "0256x") if is_png else ""
    )

    if is_png:
        ctx.image.save(path, optimize=True)
    elif "jpeg_compression" in ctx.selected_effects:
        quality = int(_range(ctx.rng, ctx.params["jpeg_quality"]))
        ctx.image.convert("RGB").save(path, quality=quality, optimize=True)
        ctx.remember("jpeg_compression", {"quality": quality})
    else:
        ctx.image.convert("RGB").save(path, quality=DEFAULT_JPEG_QUALITY, optimize=True)
    if not is_png:
        full_page_dhash_32 = format(
            compute_full_page_dhash(path, hash_size=32),
            "0256x",
        )

    stamp_metadata = {
        "stamp_id": None,
        "stamp_text": None,
        "stamp_language_mix": None,
        "stamp_style": None,
        "stamp_bbox": None,
        "stamp_alpha": None,
        "stamp_rotation_degrees": None,
        "stamp_color_group": None,
        "stamp_color_rgb": None,
        "stamp_shape": None,
        "stamp_transform_matrix": None,
        "stamp_polygon": None,
        "stamp_font": None,
        "stamp_seed": None,
    }
    stamp_metadata.update(ctx.exact_parameters.get("language_stamp", {}))
    perspective_meta = ctx.exact_parameters.get("light_perspective", {})
    rotation_meta = ctx.exact_parameters.get("light_rotation", {})
    transform_metadata = {
        "kind": (
            "cumulative_homography"
            if perspective_meta and ctx.quality_profile == "phone_photo"
            else (
                "homography"
                if perspective_meta
                else ("rotation" if rotation_meta else "identity")
            )
        ),
        "forward": perspective_meta.get("forward_matrix")
        or rotation_meta.get("forward_matrix"),
        "inverse": perspective_meta.get("inverse_matrix")
        or rotation_meta.get("inverse_matrix"),
        "components": (
            ["rotation", "perspective", "fit_scale", "translation"]
            if perspective_meta and ctx.quality_profile == "phone_photo"
            else (
                ["perspective"]
                if perspective_meta
                else (["rotation"] if rotation_meta else [])
            )
        ),
    }
    severity_score = min(
        1.0,
        round(
            len([effect for effect in ctx.applied if effect != "paper_texture_light"])
            / SEVERITY_DIVISOR
            + (HEAVY_DEGRADATION_PENALTY if degradation_tier == "heavy" else 0.0)
            + (EXTREME_GEOMETRY_PENALTY if geometry_tier == "extreme" else 0.0),
            ROUND_DECIMALS,
        ),
    )

    return EffectResult(
        image_path=str(path),
        transformed_annotations=ctx.transformed,
        warnings=ctx.warnings,
        metadata={
            "effect_profile": ctx.quality_profile,
            "quality_profile": ctx.quality_profile,
            "seed": str(ctx.rng_seed),
            "effect_chain": ctx.effect_chain,
            "effects": ctx.applied,
            "exact_parameters": ctx.exact_parameters,
            "stamp_metadata": stamp_metadata,
            "selected_effects": ctx.selected_effects,
            "sampled_optional_effects": [
                effect
                for effect in ctx.selected_effects
                if effect not in ctx.profile.get("required_effects", [])
            ],
            "transform_budget": ctx.profile.get("transform_budget", 0.0),
            "transform": transform_metadata,
            "severity_score": severity_score,
            "geometry_tier": geometry_tier,
            "degradation_tier": degradation_tier,
            "heavy_variant": ctx.heavy_variant,
            "paper_base": getattr(ctx.plan, "metadata", {}).get("paper_base"),
            "transformed_annotations": ctx.transformed,
            "warnings": ctx.warnings,
            "full_page_dhash_32": full_page_dhash_32,
        },
    )


def apply_effect_pipeline(
    image_path: str,
    quality_profile: str,
    plan: Any | None = None,
    *,
    seed: str | int | None = None,
) -> EffectResult:
    config = _load_config()
    profile = config["profiles"].get(quality_profile, config["profiles"]["clean"])
    params = config["parameters"]
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    rng_seed = seed if seed is not None else f"{path.name}:{quality_profile}"
    rng = random.Random(rng_seed)
    selected_effects = _selected_effects(profile, rng)
    seed_digest = hashlib.sha256(str(rng_seed).encode("utf-8")).hexdigest()
    geometry_tier = (
        _phone_geometry_tier(seed_digest)
        if quality_profile == "phone_photo"
        else "none"
    )
    degradation_tier = _degradation_tier(seed_digest)
    heavy_variant = degradation_tier == "heavy"

    ctx = _PipelineContext(
        image=image,
        rng=rng,
        selected_effects=selected_effects,
        profile=profile,
        params=params,
        seed_digest=seed_digest,
        heavy_variant=heavy_variant,
        plan=plan,
        quality_profile=quality_profile,
        rng_seed=rng_seed,
        geometry_tier=geometry_tier,
        degradation_tier=degradation_tier,
    )

    _apply_noise_effects_part1(ctx)
    _apply_noise_effects_part2(ctx)
    _apply_stains_and_stamps(ctx)
    _apply_signatures_and_marks(ctx)
    _apply_distortion_effects(ctx)
    _apply_toning_effects(ctx)
    _apply_rotation_transform(ctx)
    _apply_perspective_transform(ctx)

    return _apply_final_adjustments_and_save(ctx, path, degradation_tier, geometry_tier)
