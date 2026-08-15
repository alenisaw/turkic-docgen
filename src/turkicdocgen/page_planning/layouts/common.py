from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from turkicdocgen.config_validation import validate_layout_specs_config
from turkicdocgen.languages import canonical_language_mix
from turkicdocgen.page_planning.content.phrase_builder import (
    build_paragraphs,
    pool,
    read_lines,
)
from turkicdocgen.render.fonts import choose_font
from turkicdocgen.schema import TableCell, TextStyle, Zone

if TYPE_CHECKING:
    import random


@dataclass(frozen=True, slots=True)
class ZoneConfig:
    zone_id: str
    zone_type: str
    bbox: tuple[int, int, int, int]
    text: str
    language: str
    order: int
    text_style: TextStyle


@dataclass(frozen=True, slots=True)
class TableCellsConfig:
    bbox: tuple[int, int, int, int]
    rows: int
    cols: int
    language: str
    rng: random.Random
    start_order: int


RENDER_PROFILE = Path(
    str(importlib.resources.files("turkicdocgen") / "configs" / "render_profile.yaml")
)


@lru_cache(maxsize=16)
def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if path.name == "layout_specs.yaml":
        return validate_layout_specs_config(raw)
    return raw


def sample_weighted(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(weights.values())
    pick = rng.random() * total
    acc = 0.0
    for key, weight in weights.items():
        acc += weight
        if pick <= acc:
            return key
    return next(reversed(weights))


def style(kind: str, rng: random.Random, language_mix: str) -> TextStyle:
    render = load_yaml(RENDER_PROFILE)
    fonts = render["fonts"]
    ranges = {
        "title": fonts["title_px"],
        "subtitle": fonts["subtitle_px"],
        "metadata": fonts["metadata_px"],
        "body": fonts["body_px"],
        "table": fonts["table_px"],
        "note": fonts["note_px"],
    }
    lo, hi = ranges.get(kind, fonts["body_px"])
    font_category = {
        "title": "serif",
        "subtitle": "serif",
        "body": "serif",
        "metadata": "sans",
        "table": "mono_or_table_safe",
        "note": "sans",
    }.get(kind, "sans")
    font = choose_font(
        language_mix,
        rng.randrange(0, 10_000),
        bold=kind in {"title", "subtitle"},
        category=font_category,
    )
    align = (
        rng.choices(("left", "center", "right"), weights=(3, 4, 3), k=1)[0]
        if kind == "title"
        else "left"
    )
    return TextStyle(
        font_family=font.family,
        font_size_px=rng.randint(int(lo), int(hi)),
        align=align,
        line_spacing=rng.uniform(1.15, 1.32),
        bold=kind in {"title", "subtitle"},
        font_path=font.path or None,
    )


def poly(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def zone(config: ZoneConfig, **metadata: object) -> Zone:
    bbox = config.bbox
    merged_metadata: dict[str, object] = dict(metadata)
    merged_metadata.setdefault("source_bbox", list(bbox))
    return Zone(
        zone_id=config.zone_id,
        zone_type=config.zone_type,
        bbox=bbox,
        polygon=poly(bbox),
        text=config.text,
        language=config.language,
        reading_order=config.order,
        style=config.text_style,
        metadata=merged_metadata,
    )


def text(
    language: str,
    rng: random.Random,
    min_chars: int,
    max_chars: int,
    max_paragraphs: int = 12,
) -> str:
    return "\n\n".join(
        build_paragraphs(
            language,
            rng,
            min_chars=min_chars,
            max_chars=max_chars,
            max_paragraphs=max_paragraphs,
        )
    )


def label(language: str, rng: random.Random, fallback: str = "Document") -> str:
    lang = canonical_language_mix(language)
    localized = {
        "kk": {
            "Document": ["Құжат", "Мәтін", "Ақпарат"],
            "Statement": ["Өтініш", "Ресми өтініш", "Мәлімдеме"],
            "Attachments": ["Қосымша", "Қосымша құжаттар"],
            "Signature": ["Қолы", "Қол қою орны"],
            "Signed": ["Қол қойған", "Жауапты тұлға"],
            "Organization": ["Мекеме атауы", "Оқу бөлімі"],
            "Subject": ["Тақырыбы", "Мәселе"],
            "Form": ["Өтініш нысаны", "Тіркеу нысаны"],
            "Table": ["Кесте", "Тізім"],
            "Note": ["Ескертпе", "Қосымша мәлімет"],
            "Bulletin": ["Хабаршы", "Ақпараттық бюллетень"],
            "Archive note": ["Мұрағат белгісі", "Қызметтік белгі"],
        },
        "ky": {
            "Document": ["Документ", "Маалымат", "Барак"],
            "Statement": ["Арыз", "Расмий арыз", "Өтүнүч"],
            "Attachments": ["Кошумча", "Кошумча документтер"],
            "Signature": ["Колу", "Кол коюу жери"],
            "Signed": ["Кол койгон", "Жооптуу адам"],
            "Organization": ["Уюмдун аталышы", "Окуу бөлүмү"],
            "Subject": ["Темасы", "Маселе"],
            "Form": ["Арыз формасы", "Каттоо формасы"],
            "Table": ["Таблица", "Тизме"],
            "Note": ["Эскертүү", "Кошумча маалымат"],
            "Bulletin": ["Бюллетень", "Маалымат барагы"],
            "Archive note": ["Архив белгиси", "Кызматтык белги"],
        },
        "ru_kk": {
            "Document": ["Құжат / Документ", "Қызметтік құжат"],
            "Statement": ["Заявление / Өтініш", "Обращение / Өтініш"],
            "Attachments": ["Қосымша / Приложение"],
            "Signature": ["Қолы / Подпись"],
            "Signed": ["Қол қойған / Подписал"],
            "Organization": ["Мекеме / Организация"],
            "Subject": ["Тақырыбы / Тема"],
            "Form": ["Нысан / Форма"],
            "Table": ["Кесте / Таблица"],
            "Note": ["Ескертпе / Примечание"],
            "Bulletin": ["Хабаршы / Бюллетень"],
            "Archive note": ["Мұрағат / Архив"],
        },
        "ru_ky": {
            "Document": ["Документ", "Кызматтык документ"],
            "Statement": ["Заявление / Арыз", "Обращение / Өтүнүч"],
            "Attachments": ["Кошумча / Приложение"],
            "Signature": ["Колу / Подпись"],
            "Signed": ["Кол койгон / Подписал"],
            "Organization": ["Уюм / Организация"],
            "Subject": ["Темасы / Тема"],
            "Form": ["Форма / Нысан"],
            "Table": ["Таблица"],
            "Note": ["Эскертүү / Примечание"],
            "Bulletin": ["Бюллетень"],
            "Archive note": ["Архив белгиси"],
        },
    }
    candidates = localized.get(lang, {}).get(fallback) or pool(lang) or [fallback]
    return rng.choice(candidates).rstrip(".")


def form_text(rng: random.Random, rows: int, language: str = "kk") -> str:
    lang = canonical_language_mix(language)
    labels_by_language = {
        "kk": [
            "Тегі",
            "Аты",
            "Туған күні",
            "Құжат нөмірі",
            "ЖСН",
            "Мекенжайы",
            "Телефон",
            "Электрондық пошта",
            "Ұйым атауы",
            "Лауазымы",
            "Бөлім",
            "Өтініш түрі",
            "Қабылдау күні",
            "Тіркеу нөмірі",
            "Қосымша құжат",
            "Қолы",
        ],
        "ky": [
            "Фамилиясы",
            "Аты",
            "Туулган күнү",
            "Документ номери",
            "Жеке номер",
            "Дареги",
            "Телефон",
            "Электрондук почта",
            "Уюмдун аталышы",
            "Кызматы",
            "Бөлүм",
            "Арыз түрү",
            "Кабыл алынган күн",
            "Каттоо номери",
            "Кошумча документ",
            "Колу",
        ],
    }
    labels = labels_by_language["ky" if lang in {"ky", "ru_ky"} else "kk"]
    values = read_lines("form_values.txt") or ["Value"]
    pairs = []
    for _ in range(rows):
        pairs.append(f"{rng.choice(labels)}: {rng.choice(values)}")
    return "\n".join(pairs)


def table_cells(config: TableCellsConfig) -> list[TableCell]:
    bbox = config.bbox
    rows = config.rows
    cols = config.cols
    language = config.language
    rng = config.rng
    start_order = config.start_order
    x1, y1, x2, y2 = bbox
    lang = canonical_language_mix(language)
    terms = read_lines("table_terms.txt") or pool(lang) or ["мән"]
    header_labels = {
        "kk": ["№", "Атауы", "Күні", "Мәртебесі", "Жауапты бөлім", "Ескертпе"],
        "ky": ["№", "Аталышы", "Күнү", "Статусу", "Жооптуу бөлүм", "Эскертүү"],
        "ru_kk": ["№", "Атауы / Название", "Күні / Дата", "Статус"],
        "ru_ky": ["№", "Аталышы / Название", "Күнү / Дата", "Статус"],
    }
    cell_w = (x2 - x1) // cols
    cell_h = (y2 - y1) // rows
    cells = []
    order = start_order
    for row in range(rows):
        for col in range(cols):
            cx1 = x1 + col * cell_w
            cy1 = y1 + row * cell_h
            cx2 = x2 if col == cols - 1 else cx1 + cell_w
            cy2 = y2 if row == rows - 1 else cy1 + cell_h
            cell_text = rng.choice(terms)
            if row == 0:
                cell_text = header_labels.get(lang, header_labels["kk"])[
                    col % len(header_labels.get(lang, header_labels["kk"]))
                ]
            cells.append(
                TableCell(row, col, (cx1, cy1, cx2, cy2), cell_text, language, order)
            )
            order += 1
    return cells
