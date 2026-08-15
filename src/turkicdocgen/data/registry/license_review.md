# License review

Synthetic templates in this starter pack are authored for the project and do not include copied real documents.

Allowed:

- synthetic document text;
- synthetic names and organization names;
- manually authored generic form/table/certificate layouts;
- open-source fonts with compatible licenses.

Rejected:

- real IDs, passports, private documents or bank records;
- real official stamps or seals;
- copied incompatible corpus text used verbatim as ground truth;
- train/benchmark leakage.

## Template pattern registry

`src/turkicdocgen/data/registry/template_patterns.jsonl` contains structural references only.
Records describe field names, layout hints, and document type coverage for synthetic generation.
They must not be treated as permission to copy real document text, real identifiers, real seals, or real personal data.

## Dataset release license

Generated dataset releases are intended for open access under CC BY 4.0 unless
an individual release card states otherwise. Project source code remains
Apache-2.0 under the repository `LICENSE`.
