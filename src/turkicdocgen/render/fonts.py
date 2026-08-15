from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from turkicdocgen.languages import canonical_language_mix, required_chars

OPTIONAL_SYSTEM_FONTS = {
    "arial.ttf",
    "arialbd.ttf",
    "times.ttf",
    "timesbd.ttf",
    "timesi.ttf",
    "timesbi.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "calibrii.ttf",
    "calibriz.ttf",
    "cambria.ttc",
    "cambriab.ttf",
    "georgia.ttf",
    "georgiab.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "verdana.ttf",
    "verdanab.ttf",
    "trebuc.ttf",
    "trebucbd.ttf",
    "cour.ttf",
    "courbd.ttf",
}

FONT_CATEGORY_TOKENS = {
    "serif": ("serif", "times", "ptserif", "freeserif", "cambria", "georgia"),
    "mono_or_table_safe": ("mono", "cour"),
    "sans": (
        "sans",
        "arial",
        "calibri",
        "roboto",
        "freesans",
        "segoe",
        "tahoma",
        "verdana",
        "trebuc",
    ),
}


@dataclass(frozen=True)
class FontChoice:
    family: str
    path: str
    source: str
    category: str = "sans"
    coverage_language: str = "kk"
    coverage_ok: bool = True


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    custom_dir = os.environ.get("TURKICDOCGEN_FONT_DIR")
    if custom_dir:
        dirs.append(Path(custom_dir))
    dirs = [
        *dirs,
        Path(".cache/turkicdocgen/fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
    ]
    if sys.platform == "win32":
        dirs.extend([Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"])
    return [path for path in dirs if str(path) and path.exists()]


def _font_category(path: Path) -> str:
    lowered = path.name.lower()
    for category, tokens in FONT_CATEGORY_TOKENS.items():
        if any(token in lowered for token in tokens):
            return category
    return "sans"


@lru_cache(maxsize=1)
def discover_font_paths() -> tuple[Path, ...]:
    names = {
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSerif.ttf",
        "DejaVuSerif-Bold.ttf",
        "LiberationSerif-Regular.ttf",
        "LiberationSerif-Bold.ttf",
        "LiberationSans-Regular.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationMono-Regular.ttf",
        "NotoSans-Regular.ttf",
        "NotoSans-Bold.ttf",
        "NotoSerif-Regular.ttf",
        "NotoSerif-Bold.ttf",
        "NotoSansMono-Regular.ttf",
        "FreeSerif.ttf",
        "FreeSans.ttf",
        "PTSans-Regular.ttf",
        "PTSerif-Regular.ttf",
        "Roboto-Regular.ttf",
        *OPTIONAL_SYSTEM_FONTS,
    }
    safe_tokens = (
        "dejavu",
        "liberation",
        "notosans",
        "notoserif",
        "notosansmono",
        "freeserif",
        "freesans",
        "ptserif",
        "ptsans",
        "roboto",
        "arial",
        "times",
        "calibri",
        "cambria",
        "georgia",
        "segoe",
        "tahoma",
        "verdana",
        "trebuc",
        "cour",
    )
    found: list[Path] = []
    for directory in _candidate_dirs():
        for suffix in ("*.ttf", "*.otf"):
            for path in directory.rglob(suffix):
                lowered = path.name.lower()
                if path.name in names or any(token in lowered for token in safe_tokens):
                    found.append(path)
    return tuple(sorted(dict.fromkeys(found), key=lambda item: item.as_posix().lower()))


def _supports(path: Path, chars: str) -> bool:
    try:
        font = ImageFont.truetype(str(path), size=24)
    except OSError:
        return False
    try:
        missing = font.getmask("\U0010ffff")
        missing_signature = (missing.size, missing.getbbox(), bytes(missing))
    except (ValueError, OSError, AttributeError):
        missing_signature = None
    for char in chars:
        if char.isspace():
            continue
        try:
            glyph = font.getmask(char)
            if glyph.getbbox() is None:
                return False
            glyph_signature = (glyph.size, glyph.getbbox(), bytes(glyph))
            if missing_signature is not None and glyph_signature == missing_signature:
                return False
        except (ValueError, OSError, AttributeError):
            return False
    return True


@lru_cache(maxsize=16)
def valid_fonts(language_mix: str) -> tuple[FontChoice, ...]:
    language_mix = canonical_language_mix(language_mix)
    required = required_chars(language_mix)
    choices: list[FontChoice] = []
    for path in discover_font_paths():
        if _supports(path, required):
            source = (
                "custom_user"
                if ".cache/turkicdocgen/fonts" in path.as_posix()
                else "system"
            )
            choices.append(
                FontChoice(
                    path.stem,
                    str(path),
                    source,
                    _font_category(path),
                    language_mix,
                    True,
                )
            )
    if not choices:
        for fallback in ("DejaVuSans.ttf", "arial.ttf"):
            fallback_path = Path(fallback)
            if _supports(fallback_path, required):
                choices.append(
                    FontChoice(
                        fallback_path.stem,
                        fallback,
                        "pil_lookup",
                        "sans",
                        language_mix,
                        True,
                    )
                )
                break
    return tuple(choices)


def choose_font(
    language_mix: str,
    seed: int,
    *,
    bold: bool = False,
    category: str | None = None,
) -> FontChoice:
    fonts = valid_fonts(language_mix)
    if category:
        preferred = [font for font in fonts if font.category == category]
        if preferred:
            fonts = tuple(preferred)
    if not fonts:
        return FontChoice("PILDefault", "", "fallback", category or "sans", "kk", False)
    chosen = fonts[seed % len(fonts)]
    if bold and chosen.path:
        path = Path(chosen.path)
        bold_candidates = [
            path.with_name(path.stem.replace("Regular", "Bold") + path.suffix),
            path.with_name(path.name.replace(".ttf", "bd.ttf")),
        ]
        for bold_name in bold_candidates:
            if bold_name.exists():
                return FontChoice(
                    bold_name.stem,
                    str(bold_name),
                    chosen.source,
                    _font_category(bold_name),
                    chosen.coverage_language,
                    chosen.coverage_ok,
                )
    return chosen
