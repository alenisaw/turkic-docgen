from __future__ import annotations

CANONICAL_LANGUAGE_MIXES = ("kk", "ky", "ru_kk", "ru_ky")

LANGUAGE_ALIASES = {
    "kk": "kk",
    "kazakh": "kk",
    "ky": "ky",
    "kg": "ky",
    "kyrgyz": "ky",
    "ru-kk": "ru_kk",
    "ru_kk": "ru_kk",
    "ru-kz": "ru_kk",
    "ru_kz": "ru_kk",
    "ru-ky": "ru_ky",
    "ru_ky": "ru_ky",
    "ru-kg": "ru_ky",
    "ru_kg": "ru_ky",
}

CYRILLIC_BASE = "АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
LATIN_BASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
DIGITS_BASE = "0123456789"
KAZAKH_SPECIAL_CYRILLIC = (
    "\u04d8\u04d9\u0492\u0493\u049a\u049b\u04a2\u04a3"
    "\u04e8\u04e9\u04b0\u04b1\u04ae\u04af\u04ba\u04bb\u0406\u0456"
)
KYRGYZ_SPECIAL_CYRILLIC = "\u04a2\u04a3\u04e8\u04e9\u04ae\u04af"

KAZAKH_CYRILLIC_REQUIRED = (
    "ӘәҒғҚқҢңӨөҰұҮүҺһІі" + CYRILLIC_BASE + LATIN_BASE + DIGITS_BASE
)
KYRGYZ_CYRILLIC_REQUIRED = "ҢңӨөҮү" + CYRILLIC_BASE + LATIN_BASE + DIGITS_BASE

REQUIRED_CHARS_BY_LANGUAGE = {
    "kk": KAZAKH_CYRILLIC_REQUIRED,
    "ky": KYRGYZ_CYRILLIC_REQUIRED,
    "ru_kk": KAZAKH_CYRILLIC_REQUIRED,
    "ru_ky": KYRGYZ_CYRILLIC_REQUIRED,
}

FORBIDDEN_LATIN_FALLBACK_TOKENS = {
    "ARIZA",
    "OTINISH",
    "OTUNUCH",
    "QABYLDANDY",
    "KABYL",
    "PRINYATO",
    "ZAYAVLENIE",
    "OBRASCHENIE",
}


def canonical_language_mix(language_mix: str | None) -> str:
    key = (language_mix or "kk").strip().lower()
    return LANGUAGE_ALIASES.get(key, key if key in CANONICAL_LANGUAGE_MIXES else "kk")


def required_chars(language_mix: str | None) -> str:
    return REQUIRED_CHARS_BY_LANGUAGE[canonical_language_mix(language_mix)]
