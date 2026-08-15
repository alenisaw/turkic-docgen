from __future__ import annotations

from pathlib import Path


def test_text_sources_do_not_contain_common_mojibake_sequences() -> None:
    roots = [Path("src/turkicdocgen")]
    suffixes = {".py", ".html", ".js", ".css", ".yaml", ".yml", ".jsonl", ".txt"}
    forbidden = (
        "\u00c3",
        "\u00c2",
        "\u00d0",
        "\u00d1",
        "\u00e2\u20ac\u201c",
        "\u00e2\u20ac\u201d",
        "\u00e2\u201e\u2016",
    )
    offenders: list[str] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8")
            if any(sequence in text for sequence in forbidden):
                offenders.append(path.as_posix())
    assert offenders == []
