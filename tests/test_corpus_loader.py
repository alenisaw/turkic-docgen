from __future__ import annotations

from turkicdocgen.page_planning.content.corpus_loader import (
    ALLOWED_LICENSES,
    load_corpus_records,
)
from turkicdocgen.page_planning.content.document_models import _DATA


def test_corpus_loader_loads_valid_records() -> None:
    records = load_corpus_records("organizations_kk.jsonl")
    assert len(records) > 0
    for record in records:
        assert record.language == "kk"
        assert record.license in ALLOWED_LICENSES
        assert record.id is not None
        assert record.text is not None


def test_corpus_loader_returns_empty_for_missing_file() -> None:
    records = load_corpus_records("nonexistent_file.jsonl")
    assert records == []


def test_typed_corpus_is_merged_into_document_context_pool() -> None:
    records = load_corpus_records("organizations_kk.jsonl")
    assert {record.text for record in records}.issubset(_DATA["kk"]["organizations"])
