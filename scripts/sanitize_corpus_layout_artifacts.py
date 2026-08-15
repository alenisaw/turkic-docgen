from __future__ import annotations

import json
from pathlib import Path

from turkicdocgen.page_planning.content.phrase_builder import sanitize_corpus_text

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "src" / "turkicdocgen" / "data" / "corpus"
PHRASE_FILES = (
    "form_values.txt",
    "kk_phrases.txt",
    "ky_phrases.txt",
    "ru_kk_mixed_phrases.txt",
    "ru_ky_mixed_phrases.txt",
)


def sanitize_phrase_file(path: Path) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8").splitlines()
    cleaned: list[str] = []
    seen: set[str] = set()
    for line in source:
        value = sanitize_corpus_text(line)
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    return len(source), len(cleaned)


def sanitize_seed_records(path: Path) -> tuple[int, int]:
    source = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cleaned: list[dict[str, object]] = []
    for row in source:
        text = sanitize_corpus_text(str(row.get("text", "")))
        if not text:
            continue
        row["text"] = text
        cleaned.append(row)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in cleaned) + "\n",
        encoding="utf-8",
    )
    return len(source), len(cleaned)


def main() -> None:
    for name in PHRASE_FILES:
        before, after = sanitize_phrase_file(CORPUS / name)
        print(f"{name}: {before} -> {after}")
    before, after = sanitize_seed_records(CORPUS / "seed_records.jsonl")
    print(f"seed_records.jsonl: {before} -> {after}")


if __name__ == "__main__":
    main()
