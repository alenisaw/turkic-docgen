---
pretty_name: "TurkicOCR Synthetic Cyrillic"
language: [kk, ky, ru]
license: cc-by-4.0
task_categories: [image-to-text, document-question-answering]
tags: [ocr, document-ocr, synthetic-data, kazakh, kyrgyz, cyrillic, turkic]
size_categories: ["100K<n<1M"]
configs:
  - config_name: tiny
    data_files:
      - split: train
        path: indexes/tiny/train.parquet
      - split: validation
        path: indexes/tiny/validation.parquet
      - split: test
        path: indexes/tiny/test.parquet
  - config_name: medium
    data_files:
      - split: train
        path: indexes/medium/train.parquet
      - split: validation
        path: indexes/medium/validation.parquet
      - split: test
        path: indexes/medium/test.parquet
  - config_name: large
    data_files:
      - split: train
        path: indexes/large/train.parquet
      - split: validation
        path: indexes/large/validation.parquet
      - split: test
        path: indexes/large/test.parquet
---

# TurkicOCR Synthetic Cyrillic

100,000 synthetic document pages for Cyrillic OCR, layout analysis, and visual document understanding (VDU) in Kazakh and Kyrgyz. Three nested configs for progressive training scale.

```python
from datasets import load_dataset
ds = load_dataset("alenisaw/turkicocr-cyrillic", name="large")
```

## Configs

| Config | Total | Train | Validation | Test |
|--------|------:|------:|-----------:|-----:|
| `tiny` | 25,000 | 22,500 | 1,250 | 1,250 |
| `medium` | 50,000 | 45,000 | 2,500 | 2,500 |
| `large` | 100,000 | 90,000 | 5,000 | 5,000 |

`tiny` ⊂ `medium` ⊂ `large` — deterministic nested views of the same generation. Images are stored as JPEG inside packed TAR shards; parquet indexes reference each page by `page_id` and `tar_path`.

## Document Layouts

29 layouts across 5 categories:

| Category | Layouts |
|---|---|
| **Administrative & Official** | Official letters, memos, meeting minutes, official statements, archival notifications, certificates |
| **Forms & Registries** | Application forms, simple forms, registry extracts |
| **Books & Prose** | Single/two-column book pages, dictionary entries, glossaries, indexes, academic abstracts, bulletins, historical newspapers |
| **Educational & Specialized** | Syllabi, lecture notes, exam sheets, exam registers, worksheets |
| **Tables & Transactional** | Invoices, receipts, catalog entries, attendance/schedule/simple/wide-schedule tables, inventory sheets |

## Degradation Profiles

7 procedurally generated visual effect profiles:

| Profile | Simulates |
|---|---|
| `clean` | No degradation |
| `low_dpi_scan` | Low-resolution scan artifacts |
| `office_scan` | Office scanner noise and banding |
| `official_stamped` | Round/rectangular ink stamps and handwritten signatures |
| `old_paper` | Aging, yellowing, water stains, blotches |
| `phone_photo` | Perspective distortion, lens blur, camera projection |
| `photocopy` | Repeated photocopy erosion and thresholding |

## Intended Use

For training and evaluating OCR, document layout analysis, and VDU models (LayoutLM, Donut, Pix2Struct, ColPali). The dataset is synthetic — validate on real-world documents before deployment.

## Limitations

- **Synthetic content**: Text is procedurally generated from corpus sources. Semantic coherence between entities (e.g. name–address binding on forms) is not guaranteed. Optimized for visual/geometric recognition, not semantic NLP tasks.
- **Domain gap**: Real-world generalization should be verified on actual scanned or photographed documents.

## Acknowledgements

The author would like to thank the Research and Innovation Center "CyberTech" at Astana IT University for their support and resources during the creation of this dataset.

## Citation

If you use this dataset or associated recognizers, please cite:

```bibtex
@inproceedings{issayev2026turkicocr,
  title={TurkicOCR-SVTRv2-B: Lightweight Line-Grounded Recognizer for Kazakh and Kyrgyz Optical Character Recognition},
  author={Issayev, Alen and Zhalgas, Aidana},
  booktitle={Analysis of Images, Social Networks and Texts (AIST 2026)},
  series={Lecture Notes in Computer Science (LNCS)},
  publisher={Springer},
  year={2026},
  doi={10.1007/978-3-031-XXXXX-X_XX}
}

@misc{issayev_2026_turkicocr_cyrillic,
  author       = {Issayev, Alen},
  title        = {TurkicOCR-Cyrillic},
  year         = {2026},
  publisher    = {Hugging Face},
  doi          = {10.57967/hf/9255},
  url          = {https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic},
  note         = {Synthetic Cyrillic OCR and document-understanding dataset}
}
```
