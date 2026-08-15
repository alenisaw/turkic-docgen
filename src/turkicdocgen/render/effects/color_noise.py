from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFilter, ImageOps

if TYPE_CHECKING:
    import random


def _degradation_tier(seed_digest: str) -> str:
    bucket = int(seed_digest[24:32], 16) % 100
    light_threshold = 50
    medium_threshold = 85
    if bucket < light_threshold:
        return "light"
    if bucket < medium_threshold:
        return "medium"
    return "heavy"


def _paper_texture(image: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGB", image.size, (249, 249, 248))
    noise_size = (256, max(1, int(256 * image.height / image.width)))
    noise = Image.new("L", noise_size)
    noise.putdata([rng.randint(226, 255) for _ in range(noise_size[0] * noise_size[1])])
    noise = noise.resize(image.size, Image.Resampling.BILINEAR)
    texture = ImageOps.colorize(noise, (235, 235, 232), (255, 255, 254))
    return Image.blend(Image.blend(image, overlay, 0.05), texture, 0.06)


def _apply_irregular_stains(
    image: Image.Image, rng: random.Random, count: int, alpha_range: list[int]
) -> tuple[Image.Image, list[dict[str, object]]]:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    artifacts: list[dict[str, object]] = []
    colors = ((126, 112, 82), (112, 103, 84), (145, 122, 78), (102, 96, 86))
    for index in range(count):
        mask = Image.new("L", image.size, 0)
        mdraw = ImageDraw.Draw(mask)
        cx = rng.randrange(80, image.width - 80)
        cy = rng.randrange(80, image.height - 80)
        rx = rng.randint(35, 145)
        ry = rng.randint(28, 130)
        for _ in range(rng.randint(5, 10)):
            ox = rng.randint(-rx // 3, rx // 3)
            oy = rng.randint(-ry // 3, ry // 3)
            local_rx = rng.randint(max(12, rx // 3), rx)
            local_ry = rng.randint(max(10, ry // 3), ry)
            mdraw.ellipse(
                (
                    cx + ox - local_rx,
                    cy + oy - local_ry,
                    cx + ox + local_rx,
                    cy + oy + local_ry,
                ),
                fill=rng.randint(int(alpha_range[0]), int(alpha_range[1])),
            )
        blur = rng.uniform(10, 28)
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
        color = rng.choice(colors)
        colored = Image.new("RGBA", image.size, (*color, 0))
        colored.putalpha(mask)
        overlay = Image.alpha_composite(overlay, colored)
        bbox = mask.getbbox() or (cx, cy, cx + 1, cy + 1)
        x1, y1, x2, y2 = bbox
        artifacts.append(
            {
                "id": f"stain_{index:02d}",
                "color_rgb": list(color),
                "polygon": [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                "bbox": list(bbox),
                "blur_radius": blur,
            }
        )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert(
        "RGB"
    ), artifacts


def _apply_paper_aging(
    image: Image.Image, seed_digest: str, variant_override: str | None = None
) -> tuple[Image.Image, dict[str, object]]:
    bucket = int(seed_digest[40:48], 16) % 100
    neutral_faded_threshold = 35
    light_warm_threshold = 70
    archive_gray_threshold = 90
    if variant_override == "neutral_faded" or (
        variant_override is None and bucket < neutral_faded_threshold
    ):
        name, overlay, alpha = "neutral_faded", (235, 235, 231), 0.08
    elif variant_override == "light_warm" or (
        variant_override is None and bucket < light_warm_threshold
    ):
        name, overlay, alpha = "light_warm", (225, 213, 183), 0.10
    elif variant_override == "archive_gray" or (
        variant_override is None and bucket < archive_gray_threshold
    ):
        name, overlay, alpha = "archive_gray", (208, 207, 198), 0.12
    else:
        name, overlay, alpha = "strong_yellow", (218, 190, 118), 0.16
    aged = Image.blend(image, Image.new("RGB", image.size, overlay), alpha)
    return aged, {
        "variant": name,
        "overlay_rgb": list(overlay),
        "blend": alpha,
        "strong_yellow": name == "strong_yellow",
    }


def _draw_toner_dropout(
    image: Image.Image, draw: ImageDraw.ImageDraw, rng: random.Random, count: int
) -> None:
    for _ in range(count):
        x = rng.randrange(60, image.width - 180)
        y = rng.randrange(80, image.height - 120)
        width = rng.randint(30, 120)
        height = rng.randint(4, 16)
        draw.rectangle((x, y, x + width, y + height), fill=(250, 249, 245, 70))


def _draw_roller_streaks(
    image: Image.Image, draw: ImageDraw.ImageDraw, rng: random.Random, count: int
) -> None:
    for _ in range(count):
        x = rng.randrange(80, image.width - 80)
        alpha = rng.randint(10, 28)
        draw.line(
            (x, 60, x + rng.randint(-22, 22), image.height - 60),
            fill=(0, 0, 0, alpha),
            width=rng.randint(1, 3),
        )


def _apply_scanline_jitter(
    image: Image.Image, rng: random.Random, amount: int
) -> Image.Image:
    if amount <= 0:
        return image
    rows = []
    for y in range(image.height):
        row = image.crop((0, y, image.width, y + 1))
        shift = rng.randint(-amount, amount) if y % rng.randint(18, 37) == 0 else 0
        if shift:
            shifted = Image.new("RGB", (image.width, 1), (250, 249, 245))
            shifted.paste(row, (shift, 0))
            row = shifted
        rows.append(row)
    out = Image.new("RGB", image.size)
    for y, row in enumerate(rows):
        out.paste(row, (0, y))
    return out
