# Corpus seed pack

This folder contains a starter seed corpus for OCR generation.

Important: this is not a certified official dictionary dump. Treat it as seed text, phrase banks, labels, and templates for the generator. If official/open dictionary or corpus sources are added later, document their licence and provenance in `corpus_sources.yaml`.

The expanded synthetic seed pack has been merged into the normal corpus files:

- `kk_phrases.txt`
- `ky_phrases.txt`
- `ru_kk_mixed_phrases.txt`
- `ru_ky_mixed_phrases.txt`
- `table_terms.txt`
- `form_values.txt`

Per-record metadata and body-only seed text are preserved in
`seed_records.jsonl`. Layout wrappers, signature placeholders, and structural
date/signature lines are intentionally excluded from paragraph pools. Stamp
phrases are stored in `stamp_phrases.jsonl`.

Approximate seed entries: 7600 text rows plus 1840 metadata records.

Run `python scripts/sanitize_corpus_layout_artifacts.py` after importing or
regenerating seed material.

Use these files for fallback generation when a larger approved/open corpus is not available.
