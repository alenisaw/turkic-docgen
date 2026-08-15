from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path

from turkicdocgen.safety import validate_structure_limits

CORPUS_DIR = importlib.resources.files("turkicdocgen") / "data" / "corpus"
MAX_CORPUS_RECORD_CHARS = 100_000

ALLOWED_LICENSES = {
    "CC-BY-4.0",
    "MIT",
    "Apache-2.0",
    "CC0-1.0",
    "ODbL-1.0",
    "open_data",
    "curated",
    "public_domain",
}


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    id: str
    language: str
    domain: str
    text: str
    source_type: str
    source: str
    license: str
    allowed_layouts: tuple[str, ...]
    tags: tuple[str, ...]


def load_corpus_records(filename: str) -> list[CorpusRecord]:
    path = Path(str(CORPUS_DIR / filename))
    if not path.exists():
        return []

    records: list[CorpusRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            if len(line) > MAX_CORPUS_RECORD_CHARS:
                raise ValueError(
                    f"Corpus record in {filename} at line {line_num} exceeds "
                    f"{MAX_CORPUS_RECORD_CHARS} characters"
                )
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {filename} at line {line_num}: {exc}"
                ) from exc
            validate_structure_limits(
                payload,
                max_depth=12,
                max_items=1_000,
                max_string_length=MAX_CORPUS_RECORD_CHARS,
            )

            # Required keys check
            for key in (
                "id",
                "language",
                "domain",
                "text",
                "source_type",
                "source",
                "license",
            ):
                if key not in payload:
                    raise ValueError(
                        f"Missing required key '{key}' in {filename} at line {line_num}"
                    )

            # License validation
            lic = payload["license"]
            if lic not in ALLOWED_LICENSES:
                raise ValueError(
                    f"Forbidden or invalid license '{lic}' in {filename} at line {line_num}"
                )

            records.append(
                CorpusRecord(
                    id=str(payload["id"]),
                    language=str(payload["language"]),
                    domain=str(payload["domain"]),
                    text=str(payload["text"]),
                    source_type=str(payload["source_type"]),
                    source=str(payload["source"]),
                    license=str(payload["license"]),
                    allowed_layouts=tuple(payload.get("allowed_layouts", [])),
                    tags=tuple(payload.get("tags", [])),
                )
            )
    return records
