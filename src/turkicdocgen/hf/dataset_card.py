from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import yaml

from turkicdocgen.release_config import RELEASE_CONFIGS

if TYPE_CHECKING:
    from pathlib import Path

DATASET_FAMILY = "TurkicDocGen Synthetic Cyrillic"

TIER_LARGE_THRESHOLD = 100_000
TIER_MEDIUM_THRESHOLD = 50_000
TIER_TINY_THRESHOLD = 25_000
SAMPLE_MAX_ROWS = 1_000


def _tier(rows: int) -> tuple[str, str, str]:
    if rows >= TIER_LARGE_THRESHOLD:
        return ("large", "100,000", "100K<n<1M")
    if rows >= TIER_MEDIUM_THRESHOLD:
        return ("medium", "50,000", "10K<n<100K")
    if rows >= TIER_TINY_THRESHOLD:
        return ("tiny", "25,000", "10K<n<100K")
    return ("sample", str(rows), "n<1K" if rows < SAMPLE_MAX_ROWS else "1K<n<10K")


def dataset_release_name(rows: int) -> str:
    tier, target_pages, _ = _tier(rows)
    if tier == "sample":
        return f"{DATASET_FAMILY} Sample"
    return f"{DATASET_FAMILY} {tier.title()} {target_pages}"


def write_dataset_card(
    out_path: Path,
    pretty_name: str | None = None,
    license_name: str = "cc-by-4.0",
    *,
    summary: dict[str, Any] | None = None,
    config_data_files: dict[str, dict[str, str]] | None = None,
) -> None:
    rows = int((summary or {}).get("rows", 0))
    layouts = sorted((summary or {}).get("layouts", {}).keys())
    languages = sorted((summary or {}).get("languages", {}).keys())
    effects = sorted((summary or {}).get("effects", {}).keys())
    tier, target_pages, size_category = _tier(rows)
    pretty_name = pretty_name or dataset_release_name(rows)
    config_data_files = config_data_files or {
        config_name: {
            "train": f"indexes/{config_name}.parquet",
        }
        for config_name in RELEASE_CONFIGS
    }
    front_matter = {
        "license": license_name,
        "language": ["kk", "ky", "ru"],
        "task_categories": [
            "image-to-text",
            "document-question-answering",
        ],
        "tags": [
            "ocr",
            "document-ocr",
            "synthetic-data",
            "kazakh",
            "kyrgyz",
            "cyrillic",
            "turkic",
        ],
        "pretty_name": pretty_name,
        "size_categories": [size_category],
        "configs": [
            {
                "config_name": config_name,
                "data_files": [
                    {"split": split, "path": path}
                    for split, path in split_paths.items()
                ],
            }
            for config_name, split_paths in config_data_files.items()
        ],
    }
    metadata = yaml.safe_dump(
        json.loads(json.dumps(front_matter, ensure_ascii=False)),
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    content = f"""---
{metadata}
---

# {DATASET_FAMILY}

## Dataset Summary

{DATASET_FAMILY} is a multi-task dataset containing high-fidelity synthetic document pages for Cyrillic-based OCR, document layout analysis, and visual document understanding (VDU) tasks. It features three nested configurations designed for progressive training scale (tiny, medium, and large).

This local export contains:
* **Exported rows**: {rows}
* **Default tier**: {tier} (target size: {target_pages})
* **Languages**: {", ".join(languages) if languages else "none"}

## Dataset Family and Configs

- Family: {DATASET_FAMILY}
- Image format: JPEG (`.jpg`) by default; PNG/WebP are accepted for compatible local imports
- Public configs: tiny, medium, large

`tiny`, `medium`, and `large` are deterministic nested views of the same master generation. They are exposed as Hugging Face dataset configurations, not as separate datasets.

## Document Layout Coverage

Observed layouts in this export: {", ".join(layouts) if layouts else "none"}

The generator features 29 distinct page layouts grouped into five major categories:
* **Administrative & Official**: Applications, archival notifications, official letters, memos, meeting minutes, and official statements.
* **Forms & Registries**: Structured forms, registry extracts, and questionnaires.
* **Books & Prose**: Single and two-column book pages, dictionary entries, glossaries, alphabetic indexes, and academic abstracts.
* **Educational & Specialized**: Course syllabi, lecture notes, exam sheets, and worksheets.
* **Structured Tables & Transactional**: Invoices, receipts, catalog entries, simple/schedule/attendance tables, and wide schedule sheets.

## Visual Degradation Effects

Observed effect profiles in this export: {", ".join(effects) if effects else "none"}

To simulate real-world document variety and capture natural visual shifts, the dataset incorporates procedurally generated degradation effects:
* **Scan & Print Degradations**: Scanner sensor noise, paper feed bands, printer streaks, toner speckles, and page bleed-through (print-through).
* **Physical Damage & Aging**: Repeated photocopy erosion/thresholding, edge-curl shadows, paper grain texture, aging/yellowing, water stains, and tea/coffee blotches.
* **Camera & Spatial Distortions**: Scanline jitter, perspective rotations, phone camera projections, and lens defocus blur.
* **Authenticity Elements**: Procedural blue/black pen handwriting for signatures, and round/rectangular official ink stamps.

## Files

- `README.md`
- `CITATION.cff`
- `family_index.json`
- `dataset_info.json`
- `checksums.sha256`
- `provenance.json`
- `indexes/samples.parquet`
- `indexes/<configuration>/<split>.parquet`
- `data/train/` packed TAR shards
- `data/validation/` packed TAR shards
- `data/test/` packed TAR shards

## Intended Use

The dataset is intended for training and evaluating text detection, optical character recognition (OCR), document layout analysis, and visual document understanding (VDU) models (such as LayoutLM, Donut, Pix2Struct, or ColPali).

## Quality Control

Release validation checks image and manifest alignment, non-empty OCR labels, minimum box dimensions, detection/recognition key consistency, QA status, and file hashes.

## Limitations

* **Synthetic Text Generation**: The text contents on document pages are procedurally generated from corpus sources. While the grammatical structures are natural, the semantic binding between entities (e.g. name-address alignment on forms, or logical continuity of statements) is synthetic and randomized. The dataset is optimized for visual/geometric text extraction and layout recognition rather than semantic natural language reasoning or fact checking.
* **Real-world Generalization**: The data is synthetic. Models trained on this dataset should be validated on real scanned or photographed documents to guarantee real-world generalization.

## Acknowledgements

The author would like to thank the Research and Innovation Center "CyberTech" at Astana IT University for their support and resources during the creation of this dataset.

## Citation

If you use this dataset or associated recognizers, please cite:

```bibtex
@inproceedings{{issayev2026turkicocr,
  title={{TurkicOCR-SVTRv2-B: Lightweight Line-Grounded Recognizer for Kazakh and Kyrgyz Optical Character Recognition}},
  author={{Issayev, Alen and Zhalgas, Aidana}},
  booktitle={{Analysis of Images, Social Networks and Texts (AIST 2026)}},
  series={{Lecture Notes in Computer Science (LNCS)}},
  publisher={{Springer}},
  year={{2026}},
  doi={{10.1007/978-3-031-XXXXX-X_XX}}
}}

@misc{{issayev_2026_turkicocr_cyrillic,
  author       = {{Issayev, Alen}},
  title        = {{TurkicOCR-Cyrillic}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  doi          = {{10.57967/hf/9255}},
  url          = {{https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic}},
  note         = {{Synthetic Cyrillic OCR and document-understanding dataset}}
}}
```
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
