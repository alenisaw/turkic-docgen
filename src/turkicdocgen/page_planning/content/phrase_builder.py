from __future__ import annotations

import importlib.resources
import json
import re
import time
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import TYPE_CHECKING

from turkicdocgen.languages import canonical_language_mix

if TYPE_CHECKING:
    import random

CORPUS_DIR = importlib.resources.files("turkicdocgen") / "data" / "corpus"
SEED_RECORDS = CORPUS_DIR / "seed_records.jsonl"

MAX_RECIPIENT_LINE_LENGTH = 120
MAX_AUTHOR_LINE_LENGTH = 80
MAX_SENTENCE_GENERATION_ATTEMPTS = 100
MAX_PARAGRAPHS = 12

_LAYOUT_HEADINGS = {
    "анкета / өтініш нысаны",
    "анкета / арыз формасы",
    "өтініш",
    "ресми өтініш",
    "арыз",
    "расмий арыз",
}
_LAYOUT_FIELD_PREFIXES = (
    "тегі, аты:",
    "жсн:",
    "қала:",
    "өтініш түрі:",
    "байланыс:",
    "қолы:",
    "күні:",
    "фамилиясы, аты:",
    "жеке номер:",
    "шаар:",
    "арыз түрү:",
    "байланыш:",
    "колу:",
    "күнү:",
    "подпись:",
    "дата:",
)
_LAYOUT_RECIPIENT_SUFFIXES = ("басшысына", "жетекчисине")
_LAYOUT_AUTHOR_SUFFIXES = ("атынан", "тарабынан")
_PLACEHOLDER_RE = re.compile(r"[_]{4,}")


@dataclass(frozen=True)
class SeedCorpusRecord:
    record_id: str
    language_mix: str
    domain: str
    source_type: str
    license_note: str
    recommended_layouts: tuple[str, ...]
    text: str


def is_layout_artifact_line(line: str) -> bool:
    normalized = " ".join(line.strip().split())
    if not normalized:
        return False
    folded = normalized.casefold()
    if _PLACEHOLDER_RE.search(normalized):
        return True
    if folded in _LAYOUT_HEADINGS:
        return True
    if folded.startswith(_LAYOUT_FIELD_PREFIXES):
        return True
    if len(normalized) < MAX_RECIPIENT_LINE_LENGTH and folded.endswith(
        _LAYOUT_RECIPIENT_SUFFIXES
    ):
        return True
    return len(normalized) < MAX_AUTHOR_LINE_LENGTH and folded.endswith(
        _LAYOUT_AUTHOR_SUFFIXES
    )


def sanitize_corpus_text(text: str) -> str:
    return "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not is_layout_artifact_line(line)
    )


@cache
def read_lines(name: str) -> tuple[str, ...]:
    path = CORPUS_DIR / name
    if not path.exists():
        return ()
    return tuple(
        cleaned
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
        if (cleaned := sanitize_corpus_text(line))
    )


@lru_cache(maxsize=1)
def seed_records() -> tuple[SeedCorpusRecord, ...]:
    if not SEED_RECORDS.exists():
        return ()
    records: list[SeedCorpusRecord] = []
    for raw in SEED_RECORDS.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        records.append(
            SeedCorpusRecord(
                record_id=str(payload.get("id", "seed_record")),
                language_mix=str(payload.get("language_mix", "")),
                domain=str(payload.get("domain", "")),
                source_type=str(payload.get("source_type", "")),
                license_note=str(payload.get("license_note", "")),
                recommended_layouts=tuple(payload.get("recommended_layouts", ())),
                text=sanitize_corpus_text(str(payload.get("text", ""))),
            )
        )
    return tuple(record for record in records if record.text)


def sample_seed_record(
    language_mix: str,
    rng: random.Random,
    *,
    layout_id: str | None = None,
    domain: str | None = None,
) -> SeedCorpusRecord | None:
    requested_language = canonical_language_mix(language_mix)
    candidates = [
        record
        for record in seed_records()
        if canonical_language_mix(record.language_mix) == requested_language
    ]
    if layout_id:
        layout_matches = [
            record for record in candidates if layout_id in record.recommended_layouts
        ]
        if layout_matches:
            candidates = layout_matches
    if domain:
        domain_matches = [record for record in candidates if record.domain == domain]
        if domain_matches:
            candidates = domain_matches
    if not candidates:
        return None
    return rng.choice(candidates)


def seed_record_metadata(record: SeedCorpusRecord | None) -> dict[str, object]:
    if record is None:
        return {}
    return {
        "corpus_record_id": record.record_id,
        "language_mix": canonical_language_mix(record.language_mix),
        "domain": record.domain,
        "source_type": record.source_type,
        "license_note": record.license_note,
        "recommended_layouts": list(record.recommended_layouts),
    }


def pool(language_mix: str) -> list[str]:
    language_mix = canonical_language_mix(language_mix)
    if language_mix in {"ru_kk", "ru_ky"}:
        local_code = "kk" if language_mix == "ru_kk" else "ky"
        local = read_lines(f"{local_code}_phrases.txt")
        mixed = read_lines(f"{language_mix}_mixed_phrases.txt")
        russian = read_lines("ru_phrases.txt")
        # Keep mixed documents Turkic-first while making Russian visibly present
        # in body text instead of limiting it to labels and short fragments.
        return [*local, *local, *local, *mixed, *mixed, *mixed, *russian]
    files = {
        "kk": [
            "kk_phrases.txt",
            "kk_words.txt",
        ],
        "ky": [
            "ky_phrases.txt",
            "ky_words.txt",
        ],
    }.get(language_mix, ["kk_phrases.txt"])
    out: list[str] = []
    for file in files:
        out.extend(read_lines(file))
    return list(dict.fromkeys(out))


def build_paragraphs(
    language_mix: str,
    rng: random.Random,
    *,
    min_chars: int,
    max_chars: int,
    max_paragraphs: int = MAX_PARAGRAPHS,
) -> list[str]:
    data = pool(language_mix) or ["Құжат мәтіні тексеру үшін дайындалды."]
    target = rng.randint(min_chars, max_chars)
    paragraphs: list[str] = []
    used: set[str] = set()
    attempts = 0
    while (
        sum(len(p) for p in paragraphs) < target
        and attempts < MAX_SENTENCE_GENERATION_ATTEMPTS
    ):
        attempts += 1
        sentences: list[str] = []
        for _ in range(rng.randint(3, 7)):
            s = rng.choice(data)
            if s in used and len(used) < len(data) * 0.7:
                continue
            used.add(s)
            sentences.append(s if s.endswith((".", "!", "?")) else s + ".")
        if sentences:
            paragraphs.append(" ".join(sentences))
        if len(paragraphs) > max_paragraphs:
            break
    return paragraphs


_content_planning_time = 0.0
_planning_depth = 0


def get_content_planning_time() -> float:
    return _content_planning_time


def reset_content_planning_time() -> float:
    global _content_planning_time
    t = _content_planning_time
    _content_planning_time = 0.0
    return t


def time_content_planning(func):
    def wrapper(*args, **kwargs):
        global _content_planning_time, _planning_depth
        if _planning_depth == 0:
            t0 = time.perf_counter()
            _planning_depth += 1
            try:
                return func(*args, **kwargs)
            finally:
                _planning_depth -= 1
                _content_planning_time += time.perf_counter() - t0
        else:
            return func(*args, **kwargs)

    return wrapper


sample_seed_record = time_content_planning(sample_seed_record)
pool = time_content_planning(pool)
build_paragraphs = time_content_planning(build_paragraphs)
seed_records = time_content_planning(seed_records)
read_lines = time_content_planning(read_lines)
