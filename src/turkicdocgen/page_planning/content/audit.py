from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any


def calculate_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def run_diversity_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    layouts: dict[str, int] = {}
    languages: dict[str, int] = {}

    # Field collections
    fields_data: dict[str, list[str]] = {
        "person_name": [],
        "organization": [],
        "department": [],
        "address": [],
        "subject": [],
        "document_number": [],
        "date": [],
        "long_note": [],
    }

    title_distribution: dict[str, dict[str, int]] = {}
    field_distribution: dict[str, dict[str, int]] = {}
    corpus_record_reuse: dict[str, int] = {}
    table_rows: list[str] = []
    paragraphs: list[str] = []
    normalized_paragraphs: list[str] = []
    normalized_titles: dict[str, Counter[str]] = {}
    orientations: Counter[str] = Counter()
    qa_flags: Counter[str] = Counter()

    for row in rows:
        layout_id = row.get("layout_id", "unknown")
        lang_mix = row.get("language_mix", "unknown")
        orientations[str(row.get("orientation", "portrait"))] += 1
        qa_flags.update(str(flag) for flag in row.get("qa_flags", []))

        layouts[layout_id] = layouts.get(layout_id, 0) + 1
        languages[lang_mix] = languages.get(lang_mix, 0) + 1

        zones = row.get("zones", [])
        for zone in zones:
            z_type = zone.get("zone_type", "")
            text = zone.get("text", "").strip()
            meta = zone.get("metadata", {})
            role = meta.get("role", zone.get("role", ""))

            # Title distribution
            if z_type == "title" and text:
                title_distribution.setdefault(layout_id, {})
                title_distribution[layout_id][text] = (
                    title_distribution[layout_id].get(text, 0) + 1
                )
                normalized_titles.setdefault(layout_id, Counter())[
                    normalize_text(text)
                ] += 1

            # Corpus reuse
            corpus_id = meta.get("corpus_record_id")
            if corpus_id:
                corpus_record_reuse[str(corpus_id)] = (
                    corpus_record_reuse.get(str(corpus_id), 0) + 1
                )

            # Paragraphs for duplicate checks
            if z_type in ("body", "paragraph") and text:
                for p in text.splitlines():
                    p_clean = p.strip()
                    if len(p_clean) > 80:
                        paragraphs.append(p_clean)
                        normalized_paragraphs.append(normalize_text(p_clean))

            # Role mapping
            if role == "sender_block":
                fields_data["organization"].append(text)
            elif role == "recipient_block":
                fields_data["person_name"].append(text)
            elif role == "date":
                fields_data["date"].append(text)
            elif role == "ref_number":
                fields_data["document_number"].append(text)
            elif role == "signature_zone":
                # Name is usually the last line
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if lines:
                    fields_data["person_name"].append(lines[-1])

            # Forms rendered_fields
            rendered_fields = meta.get("rendered_fields", [])
            for field in rendered_fields:
                f_key = field.get("field_key", "")
                f_val = field.get("value_text", "").strip()
                if not f_val:
                    continue

                # Distribution count
                dist_key = f"{layout_id}:{lang_mix}"
                field_distribution.setdefault(dist_key, {})
                field_distribution[dist_key][f_key] = (
                    field_distribution[dist_key].get(f_key, 0) + 1
                )

                # Map field values to fields_data
                if "status" in f_key:
                    continue
                if "name" in f_key or f_key == "applicant":
                    fields_data["person_name"].append(f_val)
                elif "organization" in f_key or "company" in f_key:
                    fields_data["organization"].append(f_val)
                elif "department" in f_key or f_key == "office":
                    fields_data["department"].append(f_val)
                elif "address" in f_key:
                    fields_data["address"].append(f_val)
                elif (
                    "subject" in f_key
                    or "topic" in f_key
                    or f_key in {"request_type", "request_summary"}
                ):
                    fields_data["subject"].append(f_val)
                elif "number" in f_key or "id" in f_key:
                    fields_data["document_number"].append(f_val)
                elif "date" in f_key:
                    fields_data["date"].append(f_val)
                elif "note" in f_key or len(f_val) > 40:
                    fields_data["long_note"].append(f_val)

            # Tables row uniqueness
            cells = zone.get("cells", [])
            if cells:
                rows_cells: dict[int, list[str]] = {}
                for cell in cells:
                    r = cell.get("row", 0)
                    t = cell.get("text", "").strip()
                    # skip header
                    if r > 0:
                        rows_cells.setdefault(r, []).append(t)
                for _, c_texts in rows_cells.items():
                    # structural table rows
                    row_str = " | ".join(c_texts)
                    if row_str:
                        table_rows.append(row_str)

    # Compute field summaries
    fields_summary: dict[str, dict[str, Any]] = {}
    for f_name, f_values in fields_data.items():
        cnt = len(f_values)
        unique_vals = list(set(f_values))
        uniq = len(unique_vals)
        ratio = round(uniq / cnt, 4) if cnt > 0 else 0.0
        entropy = calculate_entropy(f_values)

        counts = Counter(f_values)
        top = [[val, freq] for val, freq in counts.most_common(5)]

        fields_summary[f_name] = {
            "count": cnt,
            "unique": uniq,
            "unique_ratio": ratio,
            "top_values": top,
            "entropy": entropy,
        }

    p_counts = Counter(paragraphs)
    exact_duplicate_clusters = [
        {"text": val, "count": freq} for val, freq in p_counts.most_common() if freq > 1
    ]
    normalized_counts = Counter(normalized_paragraphs)
    normalized_duplicate_clusters = [
        {"normalized_text": val, "count": freq}
        for val, freq in normalized_counts.most_common()
        if val and freq > 1
    ]

    # Table row duplicates
    tr_counts = Counter(table_rows)
    table_row_duplicates = {val: freq for val, freq in tr_counts.items() if freq > 1}
    title_repeat_violations: list[dict[str, Any]] = []
    dataset_size = len(rows)
    for layout_id, counts in normalized_titles.items():
        page_count = layouts.get(layout_id, 0)
        if page_count <= 0 or dataset_size <= 0:
            continue
        for title, count in counts.most_common():
            dataset_ratio = count / dataset_size
            if count > 3 or dataset_ratio > 0.05:
                title_repeat_violations.append(
                    {
                        "layout_id": layout_id,
                        "normalized_title": title,
                        "count": count,
                        "layout_pages": page_count,
                        "dataset_ratio": round(dataset_ratio, 4),
                    }
                )

    return {
        "fields": fields_summary,
        "layouts": layouts,
        "languages": languages,
        "orientations": dict(orientations),
        "qa_flags": dict(qa_flags),
        "title_distribution": title_distribution,
        "title_repeat_violations": title_repeat_violations,
        "field_distribution": field_distribution,
        "corpus_record_reuse": corpus_record_reuse,
        "near_duplicate_clusters": exact_duplicate_clusters,
        "exact_duplicate_clusters": exact_duplicate_clusters,
        "normalized_duplicate_clusters": normalized_duplicate_clusters,
        "table_row_duplicates": table_row_duplicates,
    }


def _iter_atomic_texts(row: dict[str, Any]):
    for zone in row.get("zones", []):
        cells = zone.get("cells", [])
        lines = zone.get("lines", [])
        rendered_fields = zone.get("metadata", {}).get("rendered_fields", [])
        if cells:
            yield from (str(cell.get("text", "")) for cell in cells if cell.get("text"))
        elif rendered_fields:
            for field in rendered_fields:
                for key in ("label_text", "value_text"):
                    if field.get(key):
                        yield str(field[key])
        elif lines:
            yield from (str(line.get("text", "")) for line in lines if line.get("text"))
        elif zone.get("text"):
            yield str(zone["text"])


def _visual_audit_target(total_pages: int) -> int:
    if total_pages >= 100_000:
        return 500
    if total_pages >= 25_000:
        return 250
    if total_pages >= 10_000:
        return 100
    return total_pages


def _visual_audit_attributes(row: dict[str, Any]) -> dict[str, Any]:
    font_families = sorted(
        {
            str(zone.get("style", {}).get("font_family"))
            for zone in row.get("zones", [])
            if zone.get("style", {}).get("font_family")
        }
    )
    font_sizes = [
        int(zone.get("style", {}).get("font_size_px", 0))
        for zone in row.get("zones", [])
        if zone.get("style", {}).get("font_size_px")
    ]
    effect_chain = row.get("effect_chain") or row.get("effect_metadata", {}).get(
        "effect_chain", []
    )
    qa_flags = [str(flag) for flag in row.get("qa_flags", [])]
    if not qa_flags:
        qa_flags = [
            str(issue.get("code", issue))
            for issue in row.get("qa_issues", [])
            if isinstance(issue, dict)
        ]
    column_count = int(
        row.get("column_count")
        or row.get("metadata_groups", {}).get("layout", {}).get("column_count")
        or (2 if row.get("layout_id") == "book_page_two_columns" else 1)
    )
    return {
        "layout_id": row.get("layout_id", "unknown"),
        "layout_variant": row.get("layout_variant", "unknown"),
        "quality_profile": row.get("quality_profile", "unknown"),
        "effect_chain": list(effect_chain),
        "language_mix": row.get("language_mix", "unknown"),
        "orientation": row.get("orientation", "portrait"),
        "font_families": font_families,
        "font_size_bucket": (
            f"{min(font_sizes)}-{max(font_sizes)}" if font_sizes else "unknown"
        ),
        "text_density": row.get("layout_density", "unknown"),
        "column_count": column_count,
        "qa_risk": sorted(qa_flags),
    }


def generate_visual_audit_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    target = _visual_audit_target(len(rows))
    strata: dict[tuple[Any, ...], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    variant_entries: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = (
        defaultdict(list)
    )
    layout_ids = set()
    for row in rows:
        page_id = str(row["page_id"])
        attributes = _visual_audit_attributes(row)
        stratum = (
            attributes["layout_id"],
            attributes["layout_variant"],
            attributes["quality_profile"],
            json.dumps(attributes["effect_chain"], ensure_ascii=False, sort_keys=True),
            attributes["language_mix"],
            attributes["orientation"],
            tuple(attributes["font_families"]),
            attributes["font_size_bucket"],
            attributes["text_density"],
            attributes["column_count"],
            tuple(str(item) for item in attributes["qa_risk"]),
        )
        strata[stratum].append((page_id, attributes))
        variant_entries[(attributes["layout_id"], attributes["layout_variant"])].append(
            (page_id, attributes)
        )
        layout_ids.add(attributes["layout_id"])

    for entries in strata.values():
        entries.sort(
            key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest()
        )
    for entries in variant_entries.values():
        entries.sort(
            key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest()
        )

    selected = []
    selected_ids = set()

    def add(page_id: str, attributes: dict[str, Any]) -> bool:
        if page_id in selected_ids:
            return False
        selected_ids.add(page_id)
        selected.append(
            {
                "page_id": page_id,
                **attributes,
                "sample_path": f"/sample/{page_id}",
                "review_status": "pending",
                "reviewer_note": "",
            }
        )
        return True

    if len(rows) >= 100_000:
        target = max(
            target,
            20 * len(layout_ids),
            5 * len(variant_entries),
        )
        for variant in sorted(variant_entries):
            added = 0
            for page_id, attributes in variant_entries[variant]:
                if add(page_id, attributes):
                    added += 1
                if added >= 5:
                    break
        selected_layout_counts = Counter(sample["layout_id"] for sample in selected)
        for layout_id in sorted(layout_ids):
            if selected_layout_counts[layout_id] >= 20:
                continue
            layout_candidates = sorted(
                (
                    item
                    for key, entries in strata.items()
                    if key[0] == layout_id
                    for item in entries
                ),
                key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
            )
            for page_id, attributes in layout_candidates:
                if add(page_id, attributes):
                    selected_layout_counts[layout_id] += 1
                if selected_layout_counts[layout_id] >= 20:
                    break

    layout_strata: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for key in sorted(strata, key=repr):
        layout_strata[str(key[0])].append(key)
    layout_cursors = Counter()
    while len(selected) < target and any(strata.values()):
        consumed_entry = False
        for layout_id in sorted(layout_strata):
            keys = layout_strata[layout_id]
            for _ in range(len(keys)):
                cursor = layout_cursors[layout_id] % len(keys)
                layout_cursors[layout_id] += 1
                key = keys[cursor]
                if not strata[key]:
                    continue
                page_id, attributes = strata[key].pop(0)
                consumed_entry = True
                add(page_id, attributes)
                break
            if len(selected) >= target:
                break
        if not consumed_entry:
            break

    report = {
        "target_count": target,
        "selected_count": len(selected),
        "selection_method": "deterministic_composite_strata_round_robin",
        "reviewed_count": 0,
        "samples": selected,
    }
    report_path = out_dir / "reports" / "visual_audit_manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def generate_iteration3_reports(
    out_dir: Path,
    rows: list[dict[str, Any]],
    profile_name: str | None = None,
) -> None:
    from turkicdocgen.languages import (
        KAZAKH_SPECIAL_CYRILLIC,
        KYRGYZ_SPECIAL_CYRILLIC,
    )

    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    char_counts_by_lang_case = defaultdict(
        lambda: {"upper": Counter(), "lower": Counter(), "other": Counter()}
    )
    punctuation_counts = Counter()
    digit_counts = Counter()
    bigrams_by_lang = defaultdict(Counter)
    trigrams_by_lang = defaultdict(Counter)
    words_by_lang = defaultdict(list)
    line_patterns = Counter()

    for row in rows:
        lang = row.get("language_mix", "unknown")
        for text in _iter_atomic_texts(row):
            if not text:
                continue
            text = unicodedata.normalize("NFC", text)
            for line in text.splitlines():
                pattern = re.sub(r"\d", "0", line)
                pattern = re.sub(r"[A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі]", "A", pattern)
                line_patterns[pattern] += 1

            for i, char in enumerate(text):
                if char.isalpha():
                    if char.isupper():
                        char_counts_by_lang_case[lang]["upper"][char] += 1
                    else:
                        char_counts_by_lang_case[lang]["lower"][char] += 1
                elif char.isdigit():
                    digit_counts[char] += 1
                    char_counts_by_lang_case[lang]["other"][char] += 1
                elif unicodedata.category(char).startswith("P"):
                    punctuation_counts[char] += 1
                    char_counts_by_lang_case[lang]["other"][char] += 1
                else:
                    char_counts_by_lang_case[lang]["other"][char] += 1

                if i < len(text) - 1:
                    bigram = text[i : i + 2]
                    if all(c.isalpha() for c in bigram):
                        bigrams_by_lang[lang][bigram] += 1
                if i < len(text) - 2:
                    trigram = text[i : i + 3]
                    if all(c.isalpha() for c in trigram):
                        trigrams_by_lang[lang][trigram] += 1

            for word in text.split():
                clean_word = re.sub(r"[^\w]", "", word).lower()
                if clean_word:
                    words_by_lang[lang].append(clean_word)

    char_coverage = {
        "character_counts_by_language_and_case": {
            lang: {
                "upper": dict(cases["upper"].most_common(100)),
                "lower": dict(cases["lower"].most_common(100)),
                "other": dict(cases["other"].most_common(100)),
            }
            for lang, cases in char_counts_by_lang_case.items()
        },
        "punctuation_coverage": dict(punctuation_counts),
        "digit_coverage": dict(digit_counts),
        "top_character_bigrams": {
            lang: dict(bigrams.most_common(50))
            for lang, bigrams in bigrams_by_lang.items()
        },
        "top_character_trigrams": {
            lang: dict(trigrams.most_common(50))
            for lang, trigrams in trigrams_by_lang.items()
        },
    }
    special_counts = Counter()
    for cases in char_counts_by_lang_case.values():
        for case_counts in cases.values():
            special_counts.update(case_counts)
    required_specials = sorted(set(KAZAKH_SPECIAL_CYRILLIC + KYRGYZ_SPECIAL_CYRILLIC))
    special_coverage = {char: special_counts.get(char, 0) for char in required_specials}
    violations = []
    if profile_name == "large_100k" or len(rows) >= 100_000:
        for char, count in special_coverage.items():
            if count < 2_000:
                violations.append(
                    f"Required Cyrillic character {char!r} has {count} occurrences; "
                    "minimum is 2000"
                )
    char_coverage["required_special_character_counts"] = special_coverage
    char_coverage["gates_passed"] = not violations
    char_coverage["violations"] = violations

    (out_dir / "reports" / "character_coverage_report.json").write_text(
        json.dumps(char_coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base_audit = run_diversity_audit(rows)

    word_cardinality = {}
    for lang, w_list in words_by_lang.items():
        cnt = len(w_list)
        uniq = len(set(w_list))
        word_cardinality[lang] = {
            "total_words": cnt,
            "unique_words": uniq,
            "cardinality_ratio": round(uniq / cnt, 4) if cnt > 0 else 0.0,
        }

    diversity_report = {
        "content_cardinality": {
            "word_cardinality_by_language": word_cardinality,
            "line_pattern_cardinality": {
                "total_patterns": sum(line_patterns.values()),
                "unique_patterns": len(line_patterns),
                "top_patterns": [
                    {"pattern": pat, "count": freq}
                    for pat, freq in line_patterns.most_common(20)
                ],
            },
            "repetition": {
                "exact_paragraph_duplicates": len(
                    base_audit.get("exact_duplicate_clusters", [])
                ),
                "normalized_paragraph_duplicates": len(
                    base_audit.get("normalized_duplicate_clusters", [])
                ),
                "table_row_duplicates": len(base_audit.get("table_row_duplicates", {})),
            },
            "corpus_record_reuse_distribution": base_audit.get(
                "corpus_record_reuse", {}
            ),
        },
        "fields": base_audit.get("fields", {}),
        "layouts": base_audit.get("layouts", {}),
        "languages": base_audit.get("languages", {}),
        "orientations": base_audit.get("orientations", {}),
        "title_distribution": base_audit.get("title_distribution", {}),
        "title_repeat_violations": base_audit.get("title_repeat_violations", []),
    }

    (out_dir / "reports" / "diversity_report.json").write_text(
        json.dumps(diversity_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if violations:
        raise ValueError(
            "Hard Cyrillic coverage gate violations encountered:\n"
            + "\n".join(violations)
        )


def generate_iteration5_reports(
    out_dir: Path, rows: list[dict[str, Any]], profile_name: str
) -> None:
    import json
    import math
    from collections import Counter, defaultdict

    from turkicdocgen.page_planning.layouts.registry import LAYOUT_FAMILIES
    from turkicdocgen.page_planning.layouts.variants import LAYOUT_VARIANTS
    from turkicdocgen.page_planning.planner import quality_distribution_for_layout
    from turkicdocgen.profiles import load_profiles

    N = len(rows)
    if N == 0:
        return

    profile_cfg = load_profiles()
    lang_targets = profile_cfg.get("languages", {})
    quality_targets = profile_cfg.get("quality", {})
    layout_targets = profile_cfg.get("layouts", {})

    # Compute family targets
    family_targets = defaultdict(float)
    for layout_id, share in layout_targets.items():
        family = LAYOUT_FAMILIES.get(layout_id, "other")
        family_targets[family] += share

    # Compute orientation targets
    # (landscape thresholds from layout_policy.py)
    landscape_thresholds = {
        "schedule_table_page": 0.25,
        "registry_extract_page": 0.20,
        "invoice_like_page": 0.15,
        "catalog_entry_page": 0.15,
        "simple_table_page": 0.10,
        "exam_register_page": 0.25,
        "inventory_sheet_page": 0.20,
        "attendance_sheet_page": 0.20,
        "wide_schedule_page": 1.0,
    }
    landscape_share = 0.0
    for layout_id, share in layout_targets.items():
        landscape_share += share * landscape_thresholds.get(layout_id, 0.0)
    portrait_share = 1.0 - landscape_share
    orientation_targets = {"landscape": landscape_share, "portrait": portrait_share}

    # Marginal Actual Counts
    lang_counts = Counter(row.get("language_mix") for row in rows)
    quality_counts = Counter(row.get("quality_profile") for row in rows)
    layout_counts = Counter(row.get("layout_id") for row in rows)
    family_counts = Counter(
        LAYOUT_FAMILIES.get(row.get("layout_id"), "other") for row in rows
    )
    orientation_counts = Counter(row.get("orientation", "portrait") for row in rows)

    # Shard extraction
    def get_shard_id(row: dict[str, Any]) -> str:
        if row.get("shard_id"):
            return str(row["shard_id"])
        img_path = row.get("image_path") or ""
        parts = img_path.replace("\\", "/").split("/")
        for p in parts:
            if p.startswith("shard-"):
                return p
        return "shard-00000"

    shards = defaultdict(list)
    for row in rows:
        shards[get_shard_id(row)].append(row)

    # Entropy helper
    def calculate_entropy_dict(counts: dict[Any, int]) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    # Shard-level drift helper
    def calculate_drift(key_extractor, global_counts: dict[Any, int]) -> float:
        if len(shards) <= 1:
            return 0.0
        max_drift = 0.0
        all_keys = set(global_counts.keys())
        for _shard_id, shard_rows in shards.items():
            shard_counts = Counter(key_extractor(r) for r in shard_rows)
            shard_total = len(shard_rows)
            for k in all_keys | set(shard_counts.keys()):
                global_p = global_counts.get(k, 0) / N
                shard_p = shard_counts.get(k, 0) / shard_total
                max_drift = max(max_drift, abs(shard_p - global_p))
        return round(max_drift, 4)

    # Rank helpers
    def get_under_over_represented(
        deviation_dict: dict[Any, float],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sorted_devs = sorted(deviation_dict.items(), key=lambda x: x[1])
        under = [
            {"item": str(k), "deviation": round(v, 4)} for k, v in sorted_devs if v < 0
        ]
        over = [
            {"item": str(k), "deviation": round(v, 4)}
            for k, v in reversed(sorted_devs)
            if v > 0
        ]
        return under, over

    # Audit Marginal category helper
    def audit_marginal(
        targets: dict[str, float], counts: dict[str, int], key_extractor
    ) -> dict[str, Any]:
        report = {}
        deviation_dict = {}
        for k, target_share in targets.items():
            expected = target_share * N
            actual = counts.get(k, 0)
            abs_dev = actual - expected
            rel_dev = abs_dev / expected if expected > 0 else 0.0
            deviation_dict[k] = abs_dev
            report[k] = {
                "target_share": round(target_share, 4),
                "expected_count": round(expected, 2),
                "actual_count": actual,
                "absolute_deviation": round(abs_dev, 2),
                "relative_deviation": round(rel_dev, 4),
            }
        entropy = calculate_entropy_dict(counts)
        under, over = get_under_over_represented(deviation_dict)
        return {
            "buckets": report,
            "entropy": entropy,
            "effective_cardinality": round(2**entropy, 4),
            "shard_level_drift": calculate_drift(key_extractor, counts),
            "underrepresented_ranking": under,
            "overrepresented_ranking": over,
        }

    # Run marginal audits
    marginal_reports = {
        "language": audit_marginal(
            lang_targets, lang_counts, lambda r: r.get("language_mix")
        ),
        "quality_profile": audit_marginal(
            quality_targets, quality_counts, lambda r: r.get("quality_profile")
        ),
        "layout_id": audit_marginal(
            layout_targets, layout_counts, lambda r: r.get("layout_id")
        ),
        "layout_family": audit_marginal(
            family_targets,
            family_counts,
            lambda r: LAYOUT_FAMILIES.get(r.get("layout_id"), "other"),
        ),
        "orientation": audit_marginal(
            orientation_targets,
            orientation_counts,
            lambda r: r.get("orientation", "portrait"),
        ),
    }

    # Joint distributions counts
    # quality-chain helper
    def get_quality_chain(row: dict[str, Any]) -> str:
        chain = " - ".join(
            eff.get("effect", "")
            for eff in row.get("effect_chain", [])
            if isinstance(eff, dict)
        )
        return chain if chain else "none"

    actual_joint = Counter()
    actual_family_joint = Counter()
    quality_chain_counts = Counter()
    quality_counts_for_chains = Counter()
    for row in rows:
        layout = row.get("layout_id", "unknown")
        family = LAYOUT_FAMILIES.get(layout, "other")
        quality = row.get("quality_profile", "unknown")
        chain = get_quality_chain(row)
        lang = row.get("language_mix", "unknown")
        actual_joint[(layout, quality, chain, lang)] += 1
        actual_family_joint[(family, quality, chain, lang)] += 1
        quality_chain_counts[(quality, chain)] += 1
        quality_counts_for_chains[quality] += 1

    chain_share_by_quality = {
        (quality, chain): count / quality_counts_for_chains[quality]
        for (quality, chain), count in quality_chain_counts.items()
        if quality_counts_for_chains[quality]
    }
    quality_by_layout = {
        layout: quality_distribution_for_layout(profile_cfg, layout)
        for layout in layout_targets
    }

    # Build joint lists with expected/actual
    joint_report = []
    joint_devs = {}
    for layout, p_layout in sorted(layout_targets.items()):
        for lang, p_lang in sorted(lang_targets.items()):
            for quality, chain in sorted(quality_chain_counts):
                actual = actual_joint.get((layout, quality, chain, lang), 0)
                expected = (
                    N
                    * p_layout
                    * p_lang
                    * quality_by_layout[layout].get(quality, 0.0)
                    * chain_share_by_quality[(quality, chain)]
                )
                abs_dev = actual - expected
                rel_dev = abs_dev / expected if expected > 0 else 0.0
                key_str = f"{layout} x {quality} x {chain} x {lang}"
                joint_devs[key_str] = abs_dev
                joint_report.append(
                    {
                        "bucket": key_str,
                        "layout_id": layout,
                        "quality_profile": quality,
                        "effect_chain": chain,
                        "language_mix": lang,
                        "expected_count": round(expected, 2),
                        "actual_count": actual,
                        "absolute_deviation": round(abs_dev, 2),
                        "relative_deviation": round(rel_dev, 4),
                        "allowed_relative_deviation": round(
                            max(0.15, 3.0 / math.sqrt(expected))
                            if expected > 0
                            else 0.15,
                            4,
                        ),
                    }
                )

    # Family-level joint report
    family_joint_report = []
    family_joint_devs = {}
    for family in sorted(family_targets):
        for lang, p_lang in sorted(lang_targets.items()):
            for quality, chain in sorted(quality_chain_counts):
                actual = actual_family_joint.get((family, quality, chain, lang), 0)
                expected_quality_layout = sum(
                    p_layout * quality_by_layout[layout].get(quality, 0.0)
                    for layout, p_layout in layout_targets.items()
                    if LAYOUT_FAMILIES.get(layout, "other") == family
                )
                expected = (
                    N
                    * p_lang
                    * expected_quality_layout
                    * chain_share_by_quality[(quality, chain)]
                )
                abs_dev = actual - expected
                rel_dev = abs_dev / expected if expected > 0 else 0.0
                key_str = f"{family} x {quality} x {chain} x {lang}"
                family_joint_devs[key_str] = abs_dev
                family_joint_report.append(
                    {
                        "bucket": key_str,
                        "layout_family": family,
                        "quality_profile": quality,
                        "effect_chain": chain,
                        "language_mix": lang,
                        "expected_count": round(expected, 2),
                        "actual_count": actual,
                        "absolute_deviation": round(abs_dev, 2),
                        "relative_deviation": round(rel_dev, 4),
                        "allowed_relative_deviation": round(
                            max(0.15, 3.0 / math.sqrt(expected))
                            if expected > 0
                            else 0.15,
                            4,
                        ),
                    }
                )

    # Entropy for joint distributions
    joint_entropy = calculate_entropy_dict(actual_joint)
    family_joint_entropy = calculate_entropy_dict(actual_family_joint)
    under_joint, over_joint = get_under_over_represented(joint_devs)
    under_fam_joint, over_fam_joint = get_under_over_represented(family_joint_devs)

    # Auditing the 10 Coverage Dimensions
    font_families = Counter()
    font_sizes = []
    densities = Counter()
    orientations_list = Counter()
    columns_list = Counter()
    tables_list = []
    form_fields = Counter()
    zones_list = []
    paragraphs_list = []
    reading_order_depths = []

    for row in rows:
        zones = row.get("zones", [])
        zones_list.append(len(zones))
        max_ro = 0
        p_count = 0
        t_count = 0
        for zone in zones:
            z_type = zone.get("zone_type", "")
            meta = zone.get("metadata", {})
            role = meta.get("role", zone.get("role", ""))

            # Fonts
            style = zone.get("style", {})
            font_fam = style.get("font_family")
            if font_fam:
                font_families[font_fam] += 1
            font_size = style.get("font_size_px")
            if font_size:
                font_sizes.append(font_size)

            # Table count
            if z_type == "table" or role == "typed_table":
                t_count += 1

            # Paragraphs
            if z_type in ("body", "paragraph"):
                text = zone.get("text", "")
                p_count += len([p for p in text.split("\n\n") if p.strip()])

            # Reading order
            max_ro = max(max_ro, zone.get("reading_order", 0))

        tables_list.append(t_count)
        paragraphs_list.append(p_count)
        reading_order_depths.append(max_ro)

        # Layout Variant properties
        layout = row.get("layout_id", "unknown")
        family = LAYOUT_FAMILIES.get(layout, "other")
        variant_id = row.get("layout_variant")
        props = LAYOUT_VARIANTS.get(family, {}).get(variant_id, {})
        columns_list[props.get("columns", 1)] += 1
        form_fields[props.get("fields_count", 0)] += 1
        densities[row.get("layout_density", "standard")] += 1
        orientations_list[row.get("orientation", "portrait")] += 1

    # Font size buckets
    size_buckets = {
        "< 14": sum(1 for s in font_sizes if s < 14),
        "14-18": sum(1 for s in font_sizes if 14 <= s < 18),
        "18-22": sum(1 for s in font_sizes if 18 <= s < 22),
        "22-26": sum(1 for s in font_sizes if 22 <= s < 26),
        "26-30": sum(1 for s in font_sizes if 26 <= s < 30),
        "30-36": sum(1 for s in font_sizes if 30 <= s < 36),
        "36-48": sum(1 for s in font_sizes if 36 <= s < 48),
        "> 48": sum(1 for s in font_sizes if s >= 48),
    }

    coverage_dimensions = {
        "font_family": dict(font_families),
        "font_size_bucket": size_buckets,
        "text_density_bucket": dict(densities),
        "page_orientation": dict(orientations_list),
        "column_count": dict(columns_list),
        "table_count": dict(Counter(tables_list)),
        "form_field_count": dict(form_fields),
        "zone_count": {
            "min": min(zones_list, default=0),
            "max": max(zones_list, default=0),
            "mean": round(sum(zones_list) / len(zones_list), 2) if zones_list else 0.0,
        },
        "paragraph_count": {
            "min": min(paragraphs_list, default=0),
            "max": max(paragraphs_list, default=0),
            "mean": round(sum(paragraphs_list) / len(paragraphs_list), 2)
            if paragraphs_list
            else 0.0,
        },
        "reading_order_depth": {
            "min": min(reading_order_depths, default=0),
            "max": max(reading_order_depths, default=0),
            "mean": round(sum(reading_order_depths) / len(reading_order_depths), 2)
            if reading_order_depths
            else 0.0,
        },
    }

    # GATES CHECKING
    violations = []
    unknown_layouts = sorted(set(layout_counts) - set(layout_targets))
    unknown_languages = sorted(set(lang_counts) - set(lang_targets))
    unknown_qualities = sorted(set(quality_counts) - set(quality_targets))
    if unknown_layouts:
        violations.append(f"Unknown layouts observed: {unknown_layouts}")
    if unknown_languages:
        violations.append(f"Unknown language mixes observed: {unknown_languages}")
    if unknown_qualities:
        violations.append(f"Unknown quality profiles observed: {unknown_qualities}")

    # Gate 1: joint expected >= 25 must not be absent
    for j in joint_report:
        if j["expected_count"] >= 25 and j["actual_count"] == 0:
            violations.append(
                f"Gate 1 Violation: Joint bucket '{j['bucket']}' has expected count {j['expected_count']} but is absent."
            )

    # Gate 2: keep a 15% floor while allowing three standard deviations for
    # sparse joint buckets, which avoids false failures across many comparisons.
    for j in joint_report:
        if j["expected_count"] >= 100:
            allowed_deviation = j["allowed_relative_deviation"]
            if abs(j["relative_deviation"]) > allowed_deviation:
                violations.append(
                    f"Gate 2 Violation: Joint bucket '{j['bucket']}' has expected count {j['expected_count']} "
                    f"but relative deviation is {j['relative_deviation']:.2%} "
                    f"(statistical limit is +/-{allowed_deviation:.2%})."
                )

    # Gate 3: marginal language, layout family, orientation, and quality-profile shares must remain within +/-2 percentage points
    def check_gate3(name, target_shares, actual_counts):
        for val, target in target_shares.items():
            actual_share = actual_counts.get(val, 0) / N
            diff = actual_share - target
            if abs(diff) > 0.02:
                violations.append(
                    f"Gate 3 Violation: Marginal share of '{name}:{val}' is {actual_share:.2%} "
                    f"(target {target:.2%}, deviation {diff:+.2%}, limit is +/-2 percentage points)."
                )

    check_gate3("language", lang_targets, lang_counts)
    check_gate3("layout_family", family_targets, family_counts)
    check_gate3("orientation", orientation_targets, orientation_counts)
    check_gate3("quality_profile", quality_targets, quality_counts)

    # Gate 4: every major layout family meets its variant minimum (>= 12 unique variant IDs generated if family count > 50)
    for fam in family_targets.keys():
        unique_vars = {
            row.get("layout_variant")
            for row in rows
            if LAYOUT_FAMILIES.get(row.get("layout_id")) == fam
            and row.get("layout_variant")
        }
        required_min = 12
        if len(unique_vars) < required_min and family_counts.get(fam, 0) >= 50:
            violations.append(
                f"Gate 4 Violation: Layout family '{fam}' has only {len(unique_vars)} unique variants "
                f"generated (minimum required is {required_min})."
            )

    # Gate 5: no font family, orientation, or column-count category declared public has zero coverage
    for val in orientation_targets.keys():
        if orientation_counts.get(val, 0) == 0:
            violations.append(
                f"Gate 5 Violation: Page orientation '{val}' has zero coverage."
            )
    if not font_families:
        violations.append("Gate 5 Violation: No font families have coverage.")
    for col in (1, 2):
        if columns_list.get(col, 0) == 0:
            violations.append(
                f"Gate 5 Violation: Column count '{col}' has zero coverage."
            )

    # Write distribution_report.json
    distribution_report = {
        "profile": profile_name,
        "total_samples": N,
        "gates_passed": len(violations) == 0,
        "violations": violations,
        "marginal_distributions": marginal_reports,
        "joint_distribution_layout_quality_effect_lang": {
            "entropy": joint_entropy,
            "effective_cardinality": round(2**joint_entropy, 4),
            "underrepresented_ranking": under_joint[:20],
            "overrepresented_ranking": over_joint[:20],
            "buckets": joint_report,
        },
        "joint_distribution_family_quality_effect_lang": {
            "entropy": family_joint_entropy,
            "effective_cardinality": round(2**family_joint_entropy, 4),
            "underrepresented_ranking": under_fam_joint[:20],
            "overrepresented_ranking": over_fam_joint[:20],
            "buckets": family_joint_report,
        },
        "coverage_dimensions": coverage_dimensions,
    }

    report_path = out_dir / "reports" / "distribution_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(distribution_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Generate human-readable summary
    summary_lines = [
        "# Multidimensional Distribution Diversity Report Summary",
        f"Profile: {profile_name}",
        f"Total Samples: {N}",
        f"Status: {'PASSED' if not violations else 'FAILED'}",
        "",
    ]
    if violations:
        summary_lines.append("## Violations:")
        for v in violations:
            summary_lines.append(f"- {v}")
        summary_lines.append("")

    summary_lines.append("## Marginal Distributions:")
    for name, rep in marginal_reports.items():
        summary_lines.append(
            f"### {name.replace('_', ' ').title()} Entropy: {rep['entropy']} (Effective Cardinality: {rep['effective_cardinality']:.2f})"
        )
        summary_lines.append("| Category | Target % | Actual % | Abs Dev | Rel Dev |")
        summary_lines.append("|---|---|---|---|---|")
        for val, info in rep["buckets"].items():
            actual_pct = (info["actual_count"] / N) * 100
            target_pct = info["target_share"] * 100
            summary_lines.append(
                f"| {val} | {target_pct:.2f}% | {actual_pct:.2f}% | {info['absolute_deviation']:+.2f} | {info['relative_deviation']:.2%} |"
            )
        summary_lines.append("")

    summary_path = out_dir / "reports" / "distribution_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    # Handle error blocking
    is_public = (
        profile_cfg.get("profiles", {}).get(profile_name, {}).get("public", False)
    )
    if violations and (is_public or N >= 10000):
        raise ValueError(
            f"Hard distribution gate violations encountered for profile '{profile_name}':\n"
            + "\n".join(violations)
        )


def generate_iteration6_reports(
    out_dir: Path,
    rows: list[dict[str, Any]],
    profile_name: str | None = None,
) -> None:
    from turkicdocgen.page_planning.layouts.registry import LAYOUT_FAMILIES
    from turkicdocgen.qa import QA_CONFIG

    font_sizes = Counter()
    total_elements = 0
    wrapped_elements = 0
    not_wrapped_elements = 0
    completed_elements = 0
    truncated_elements = 0
    total_truncations_meta = 0
    small_table_fonts = 0
    qa_issue_counts = Counter()
    density_by_family = defaultdict(Counter)

    for row in rows:
        layout_id = row.get("layout_id", "unknown")
        family = LAYOUT_FAMILIES.get(layout_id, "other")
        density = row.get("layout_density", "standard")
        density_by_family[family][density] += 1

        for flag in row.get("qa_flags", []):
            qa_issue_counts[flag] += 1

        zones = row.get("zones", [])
        for zone in zones:
            meta = zone.get("metadata", {})
            style = zone.get("style", {})
            z_type = zone.get("zone_type", "")

            def process_item(item_meta, default_fs):
                nonlocal total_elements, wrapped_elements, not_wrapped_elements
                nonlocal completed_elements, truncated_elements, total_truncations_meta

                fs = item_meta.get("font_size") or default_fs
                if fs:
                    try:
                        font_sizes[int(fs)] += 1
                    except (ValueError, TypeError):
                        pass

                wrapped = item_meta.get("wrap_state")
                if wrapped is None:
                    wrapped = item_meta.get("wrapped")

                complete = item_meta.get("completion_state")
                if complete is None:
                    complete = item_meta.get("rendered_complete")

                if wrapped is not None or complete is not None:
                    total_elements += 1
                    if wrapped is True:
                        wrapped_elements += 1
                    elif wrapped is False:
                        not_wrapped_elements += 1

                    if complete is True:
                        completed_elements += 1
                    elif complete is False:
                        truncated_elements += 1
                        total_truncations_meta += 1

            if z_type not in {"stamp", "decorative_non_text"}:
                process_item(meta, style.get("font_size_px"))

            if z_type == "table":
                for cell in zone.get("cells", []):
                    cell_meta = cell.get("metadata", {})
                    process_item(cell_meta, cell_meta.get("rendered_font_size"))
                    rendered_font = cell_meta.get("rendered_font_size")
                    if isinstance(rendered_font, int) and rendered_font < QA_CONFIG.get(
                        "min_table_font_px", 18
                    ):
                        small_table_fonts += 1

            elif z_type == "form":
                for field in meta.get("rendered_fields", []):
                    process_item(field, field.get("font_size"))

    overlap_keys = {
        "excessive_overlap",
        "form_field_overlap",
        "glyph_intersection_neighboring_cells",
        "glyph_intersection_label_value",
    }
    total_overlaps = sum(qa_issue_counts[k] for k in overlap_keys)

    truncation_keys = {
        "rendered_text_truncated",
        "table_cell_text_truncated",
        "required_zone_empty",
    }
    total_truncations = (
        sum(qa_issue_counts[k] for k in truncation_keys) or total_truncations_meta
    )
    mismatch_keys = {
        "table_cell_text_truncated",
        "rendered_text_mismatch",
        "rendered_text_outside_cell",
    }
    total_mismatches = sum(qa_issue_counts[k] for k in mismatch_keys)
    violations = []
    if total_truncations:
        violations.append(f"Rendered text truncations: {total_truncations}")
    if total_overlaps:
        violations.append(f"Rendered glyph overlaps: {total_overlaps}")
    if total_mismatches:
        violations.append(f"Rendered text mismatches: {total_mismatches}")
    if small_table_fonts:
        violations.append(f"Table cells below minimum font: {small_table_fonts}")

    report = {
        "font_size_distribution": dict(font_sizes),
        "wrap_and_fit_distribution": {
            "total_elements": total_elements,
            "wrapped_elements": wrapped_elements,
            "not_wrapped_elements": not_wrapped_elements,
            "completed_elements": completed_elements,
            "truncated_elements": truncated_elements,
        },
        "truncation_and_overlap_counts": {
            "total_truncations": total_truncations,
            "total_overlaps": total_overlaps,
            "total_mismatches": total_mismatches,
            "small_table_fonts": small_table_fonts,
            "qa_issue_counts": dict(qa_issue_counts),
        },
        "density_distribution_by_family": {
            fam: dict(densities) for fam, densities in density_by_family.items()
        },
        "gates_passed": not violations,
        "violations": violations,
    }

    report_path = out_dir / "reports" / "quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    hard_gate_profiles = {"tiny_25k", "medium_50k", "large_100k"}
    if violations and (profile_name in hard_gate_profiles or len(rows) >= 25_000):
        raise ValueError(
            "Hard rendering quality gate violations encountered:\n"
            + "\n".join(violations)
        )


def generate_iteration8_reports(
    out_dir: Path, rows: list[dict[str, Any]], profile_name: str | None = None
) -> None:
    import json
    from collections import defaultdict

    from turkicdocgen.dedup import (
        cluster_hashes_hamming_lsh,
        compute_binarized_page_mask_dhash,
        compute_file_sha256,
        compute_full_page_dhash,
        compute_layout_skeleton_dhash,
        compute_layout_structure_hash,
        generate_contact_sheet,
        get_exact_meaningful_text,
        get_exact_meaningful_text_hash,
        get_normalized_meaningful_text,
        get_normalized_meaningful_text_hash,
        group_near_duplicates_minhash_lsh,
    )
    from turkicdocgen.page_planning.layouts.registry import LAYOUT_FAMILIES

    full_page_hamming_threshold = 14
    page_mask_hamming_threshold = 8

    # 1. Exact meaningful text duplicates
    text_groups = defaultdict(list)
    normalized_text_groups = defaultdict(list)
    normalized_texts = []
    normalized_text_ids = []
    pid_to_row = {}
    for row in rows:
        pid = row["page_id"]
        pid_to_row[pid] = row
        txt = get_exact_meaningful_text(row)
        if txt.strip():
            h_text = get_exact_meaningful_text_hash(row)
            text_groups[h_text].append(pid)
        normalized_text = get_normalized_meaningful_text(row)
        if normalized_text:
            normalized_text_groups[get_normalized_meaningful_text_hash(row)].append(pid)
            normalized_texts.append(normalized_text)
            normalized_text_ids.append(pid)

    exact_meaningful_text_duplicates = [g for g in text_groups.values() if len(g) > 1]
    normalized_meaningful_text_duplicates = [
        group for group in normalized_text_groups.values() if len(group) > 1
    ]
    near_meaningful_text_duplicates = group_near_duplicates_minhash_lsh(
        normalized_texts,
        normalized_text_ids,
        threshold=0.85,
    )

    def stable_clusters(clusters: list[list[str]]) -> list[list[str]]:
        stable = [sorted(set(cluster)) for cluster in clusters if len(set(cluster)) > 1]
        return sorted(stable, key=lambda cluster: (-len(cluster), cluster[0]))

    exact_meaningful_text_duplicates = stable_clusters(exact_meaningful_text_duplicates)
    normalized_meaningful_text_duplicates = stable_clusters(
        normalized_meaningful_text_duplicates
    )
    near_meaningful_text_duplicates = stable_clusters(near_meaningful_text_duplicates)

    # 2. Exact & Near full page duplicates
    exact_file_hashes = defaultdict(list)
    full_page_hashes = {}
    image_errors = []
    shard_file_hashes = {}
    for manifest_path in sorted(
        (out_dir / "shards").glob("shard-*/shard_manifest.json")
    ):
        try:
            shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for relative_path, file_info in shard_manifest.get("files", {}).items():
            if relative_path.startswith("images/") and isinstance(file_info, dict):
                shard_file_hashes[Path(relative_path).name] = file_info.get("sha256")

    pending_dhashes = []
    for row in rows:
        pid = row["page_id"]
        img_rel = row.get("image") or row.get("image_path") or f"images/{pid}.png"
        img_path = out_dir / img_rel
        if not img_path.is_file():
            image_errors.append({"page_id": pid, "error": "image_missing"})
            continue
        try:
            file_hash = shard_file_hashes.get(img_path.name)
            if not file_hash:
                file_hash = compute_file_sha256(img_path)
            exact_file_hashes[file_hash].append(pid)
            stored_dhash = row.get("metadata", {}).get("full_page_dhash_32") or row.get(
                "effect_metadata", {}
            ).get("full_page_dhash_32")
            if stored_dhash:
                full_page_hashes[pid] = int(str(stored_dhash), 16)
            else:
                pending_dhashes.append((pid, img_path))
        except (OSError, ValueError) as exc:
            image_errors.append(
                {"page_id": pid, "error": type(exc).__name__, "message": str(exc)}
            )

    if pending_dhashes:
        worker_count = min(32, os.cpu_count() or 1, len(pending_dhashes))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            hashes = executor.map(
                compute_full_page_dhash,
                (path for _, path in pending_dhashes),
                repeat(32),
                chunksize=16,
            )
            for (pid, _), page_hash in zip(pending_dhashes, hashes, strict=True):
                full_page_hashes[pid] = page_hash

    exact_full_page_duplicates = stable_clusters(
        [group for group in exact_file_hashes.values() if len(group) > 1]
    )
    near_full_page_duplicates = []

    if full_page_hashes:
        pids_list = list(full_page_hashes.keys())
        hashes_list = [full_page_hashes[pid] for pid in pids_list]
        near_full_page_duplicates = stable_clusters(
            cluster_hashes_hamming_lsh(
                hashes_list,
                pids_list,
                threshold=full_page_hamming_threshold,
                hash_bits=1024,
            )
        )

    # 3. Structural layout and binarized page-mask clusters
    skeleton_hashes = {}
    structure_hashes = {}
    page_mask_hashes = {}
    structure_errors = []
    for row in rows:
        pid = row["page_id"]
        try:
            skeleton_hashes[pid] = compute_layout_skeleton_dhash(row, hash_size=32)
            structure_hashes[pid] = compute_layout_structure_hash(row)
            page_mask_hashes[pid] = compute_binarized_page_mask_dhash(row, hash_size=32)
        except (OSError, TypeError, ValueError) as exc:
            structure_errors.append(
                {"page_id": pid, "error": type(exc).__name__, "message": str(exc)}
            )

    structural_layout_clusters = []
    if skeleton_hashes:
        pids_list = list(skeleton_hashes.keys())
        hashes_list = [skeleton_hashes[pid] for pid in pids_list]
        structural_layout_clusters = stable_clusters(
            cluster_hashes_hamming_lsh(
                hashes_list, pids_list, threshold=0, hash_bits=1024
            )
        )

    exact_structure_clusters = []
    if structure_hashes:
        groups = defaultdict(list)
        for page_id, structure_hash in structure_hashes.items():
            groups[structure_hash].append(page_id)
        exact_structure_clusters = stable_clusters(
            [members for members in groups.values() if len(members) > 1]
        )

    page_mask_clusters = []
    if page_mask_hashes:
        mask_blocks = defaultdict(list)
        for row in rows:
            page_id = row["page_id"]
            if page_id not in page_mask_hashes:
                continue
            block = (
                row.get("layout_id", "unknown"),
                row.get("orientation", "portrait"),
            )
            mask_blocks[block].append(page_id)
        mask_clusters = []
        for block in sorted(mask_blocks):
            pids_list = mask_blocks[block]
            hashes_list = [page_mask_hashes[pid] for pid in pids_list]
            mask_clusters.extend(
                cluster_hashes_hamming_lsh(
                    hashes_list,
                    pids_list,
                    threshold=page_mask_hamming_threshold,
                    hash_bits=1024,
                )
            )
        page_mask_clusters = stable_clusters(mask_clusters)

    # Compute layout family concentrations
    family_pages = defaultdict(list)
    for row in rows:
        pid = row["page_id"]
        layout_id = row.get("layout_id", "unknown")
        family = LAYOUT_FAMILIES.get(layout_id, "other")
        family_pages[family].append(pid)

    FAMILY_CONCENTRATION_LIMITS = {
        "book": 0.05,
        "official": 0.05,
        "form": 0.08,
        "table": 0.08,
        "specialized": 0.05,
        "reference": 0.05,
        "structured": 0.05,
        "other": 0.05,
    }

    structural_layout_clusters_concentration = {}
    violations = []

    for family, pids in family_pages.items():
        n_fam = len(pids)
        limit = FAMILY_CONCENTRATION_LIMITS.get(family, 0.05)
        c_max = 0
        if n_fam > 0 and exact_structure_clusters:
            pids_set = set(pids)
            for cluster in exact_structure_clusters:
                intersect_size = len(pids_set.intersection(cluster))
                if intersect_size > c_max:
                    c_max = intersect_size

        concentration = c_max / n_fam if n_fam > 0 else 0.0
        passed = True
        # Fixed percentage limits are not statistically meaningful for tiny
        # family samples: 5% of 21 pages would require every hash to be unique.
        if n_fam >= 100 and concentration > limit:
            passed = False
            violations.append(
                f"Duplicate Gate Violation: Layout family '{family}' structural duplicate concentration "
                f"is {concentration:.2%} which exceeds the limit of {limit:.2%}"
            )

        structural_layout_clusters_concentration[family] = {
            "concentration": round(concentration, 4),
            "limit": limit,
            "passed": passed,
            "largest_cluster_size": c_max,
            "total_pages": n_fam,
        }

    audited_images = len(full_page_hashes)
    near_page_ids = {
        page_id for cluster in near_full_page_duplicates for page_id in cluster
    }
    near_duplicate_excess = sum(
        len(cluster) - 1 for cluster in near_full_page_duplicates
    )
    near_duplicate_rate = (
        near_duplicate_excess / audited_images if audited_images else 0.0
    )
    largest_near_cluster = max(
        (len(cluster) for cluster in near_full_page_duplicates), default=0
    )
    if exact_meaningful_text_duplicates:
        violations.append(
            "Exact meaningful-content duplicate count must be zero; "
            f"found {len(exact_meaningful_text_duplicates)} clusters"
        )
    if exact_full_page_duplicates:
        violations.append(
            "Exact full-page duplicate count must be zero; "
            f"found {len(exact_full_page_duplicates)} clusters"
        )
    if near_duplicate_rate > 0.005:
        violations.append(
            f"Full-page near-duplicate rate is {near_duplicate_rate:.2%}; limit is 0.50%"
        )
    if largest_near_cluster > 10:
        violations.append(
            f"Largest full-page near-duplicate cluster has {largest_near_cluster} pages; "
            "limit is 10"
        )
    if image_errors or audited_images != len(rows):
        violations.append(
            f"Full-page audit coverage is incomplete: {audited_images}/{len(rows)} images"
        )
    if structure_errors or len(skeleton_hashes) != len(rows):
        violations.append(
            "Structural audit coverage is incomplete: "
            f"{len(skeleton_hashes)}/{len(rows)} pages"
        )

    is_public = False
    if profile_name:
        from turkicdocgen.profiles import load_profiles

        try:
            profile_cfg = load_profiles()
            is_public = (
                profile_cfg.get("profiles", {})
                .get(profile_name, {})
                .get("public", False)
            )
        except (KeyError, TypeError, ValueError):
            is_public = False

    report_data = {
        "exact_meaningful_text_duplicates": exact_meaningful_text_duplicates,
        "normalized_meaningful_text_duplicates": normalized_meaningful_text_duplicates,
        "near_meaningful_text_duplicates": near_meaningful_text_duplicates,
        "exact_full_page_duplicates": exact_full_page_duplicates,
        "near_full_page_duplicates": near_full_page_duplicates,
        "structural_layout_clusters": structural_layout_clusters,
        "exact_structure_clusters": exact_structure_clusters,
        "page_mask_clusters": page_mask_clusters,
        "structural_layout_clusters_concentration": structural_layout_clusters_concentration,
        "metrics": {
            "total_pages": len(rows),
            "audited_images": audited_images,
            "near_full_page_count": near_duplicate_excess,
            "near_full_page_involved_pages": len(near_page_ids),
            "near_full_page_rate": round(near_duplicate_rate, 6),
            "largest_near_full_page_cluster": largest_near_cluster,
            "full_page_hamming_threshold": full_page_hamming_threshold,
            "page_mask_hamming_threshold": page_mask_hamming_threshold,
        },
        "audit_errors": {
            "images": image_errors,
            "structures": structure_errors,
        },
        "gates_passed": not violations,
        "violations": violations,
    }

    report_path = out_dir / "reports" / "duplicate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cluster_manifest = out_dir / "reports" / "duplicate_clusters.jsonl"
    with cluster_manifest.open("w", encoding="utf-8") as handle:
        cluster_sets = {
            "exact_meaningful_text": exact_meaningful_text_duplicates,
            "normalized_meaningful_text": normalized_meaningful_text_duplicates,
            "near_meaningful_text": near_meaningful_text_duplicates,
            "exact_full_page": exact_full_page_duplicates,
            "near_full_page": near_full_page_duplicates,
            "layout_skeleton": structural_layout_clusters,
            "exact_layout_structure": exact_structure_clusters,
            "page_mask": page_mask_clusters,
        }
        for mode, clusters in cluster_sets.items():
            for index, members in enumerate(clusters):
                handle.write(
                    json.dumps(
                        {
                            "cluster_id": f"{mode}_{index:06d}",
                            "mode": mode,
                            "members": members,
                            "size": len(members),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # 4. Generate contact sheets for largest clusters
    # Exact/Near meaningful text clusters
    for i, cluster in enumerate(exact_meaningful_text_duplicates[:3]):
        paths = []
        for pid in cluster:
            row = pid_to_row[pid]
            img_rel = row.get("image") or row.get("image_path") or f"images/{pid}.png"
            paths.append(out_dir / img_rel)
        contact_path = out_dir / "reports" / f"contact_sheet_text_cluster_{i + 1}.png"
        generate_contact_sheet(paths, contact_path)

    # Near full page clusters
    for i, cluster in enumerate(near_full_page_duplicates[:3]):
        paths = []
        for pid in cluster:
            row = pid_to_row[pid]
            img_rel = row.get("image") or row.get("image_path") or f"images/{pid}.png"
            paths.append(out_dir / img_rel)
        contact_path = out_dir / "reports" / f"contact_sheet_visual_cluster_{i + 1}.png"
        generate_contact_sheet(paths, contact_path)

    if violations and (is_public or len(rows) >= 10_000):
        raise ValueError(
            "Hard deduplication gate violations encountered:\n" + "\n".join(violations)
        )
