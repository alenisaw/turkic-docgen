from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from turkicdocgen.languages import KAZAKH_SPECIAL_CYRILLIC, KYRGYZ_SPECIAL_CYRILLIC
from turkicdocgen.page_planning.layouts.registry import LAYOUT_FAMILIES
from turkicdocgen.release_config import (
    LARGE_CONFIG,
    MEDIUM_CONFIG,
    RELEASE_CONFIGS,
    TINY_CONFIG,
)

RARE_CHARACTERS = set(KAZAKH_SPECIAL_CYRILLIC + KYRGYZ_SPECIAL_CYRILLIC)
SPLIT_ORDER = ("train", "val", "test")
DUPLICATE_CLUSTER_FIELDS = (
    "exact_meaningful_text_duplicates",
    "normalized_meaningful_text_duplicates",
    "near_meaningful_text_duplicates",
    "exact_full_page_duplicates",
    "near_full_page_duplicates",
    "structural_layout_clusters",
    "page_mask_clusters",
)


def proportional_split_targets(
    split_sizes: dict[str, int], total_target: int
) -> dict[str, int]:
    """Allocate an exact subset target proportionally with deterministic remainders."""
    total_size = sum(split_sizes.values())
    if total_target < 0 or total_target > total_size:
        raise ValueError("Subset target must be between zero and the master size")
    if total_size == 0:
        return dict.fromkeys(SPLIT_ORDER, 0)
    quotas = {
        split: total_target * split_sizes.get(split, 0) / total_size
        for split in SPLIT_ORDER
    }
    targets = {split: int(quotas[split]) for split in SPLIT_ORDER}
    remaining = total_target - sum(targets.values())
    remainder_order = sorted(
        SPLIT_ORDER,
        key=lambda split: (
            -(quotas[split] - targets[split]),
            SPLIT_ORDER.index(split),
        ),
    )
    for split in remainder_order[:remaining]:
        targets[split] += 1
    return targets


class DSU:
    def __init__(self, elements: list[str]):
        self.parent = {x: x for x in elements}
        self.size = {x: 1 for x in elements}

    def find(self, x: str) -> str:
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        for node in path:
            self.parent[node] = x
        return x

    def union(self, x: str, y: str) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.size[root_x] < self.size[root_y]:
                root_x, root_y = root_y, root_x
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]


def has_rare_characters(sample: dict[str, Any]) -> bool:
    """Checks if a sample contains any rare Kazakh/Kyrgyz Cyrillic characters in its text fields."""
    # Check text fields in zones
    for zone in sample.get("zones", []):
        text = zone.get("text", "")
        if any(c in RARE_CHARACTERS for c in text):
            return True
        # Check lines
        for line in zone.get("lines", []):
            if any(c in RARE_CHARACTERS for c in line.get("text", "")):
                return True
        # Check cells
        for cell in zone.get("cells", []):
            if any(c in RARE_CHARACTERS for c in cell.get("text", "")):
                return True
    return False


def load_duplicate_clusters(out_dir: Path) -> list[list[str]]:
    """Loads visual/text duplicate clusters from reports/duplicate_report.json if available."""
    report_path = out_dir / "reports" / "duplicate_report.json"
    if not report_path.exists():
        return []
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return [
            list(cluster)
            for field in DUPLICATE_CLUSTER_FIELDS
            for cluster in data.get(field, [])
        ]
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        raise ValueError(f"Invalid duplicate report: {report_path}") from exc


def get_duplicate_clusters(
    rows: list[dict[str, Any]], out_dir: Path | None = None
) -> list[list[str]]:
    """Retrieves duplicate clusters from duplicate_report.json or calculates them on-the-fly."""
    if (
        out_dir is not None
        and (out_dir / "reports" / "duplicate_report.json").is_file()
    ):
        return load_duplicate_clusters(out_dir)

    # On-the-fly fallback text duplicates
    text_groups = defaultdict(list)
    for row in rows:
        pid = row.get("page_id")
        if not pid:
            continue
        text_content = ""
        for zone in row.get("zones", []):
            text_content += zone.get("text", "")
        if text_content.strip():
            h_text = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
            text_groups[h_text].append(pid)

    return [g for g in text_groups.values() if len(g) > 1]


def build_components(
    rows: list[dict[str, Any]], duplicate_clusters: list[list[str]]
) -> list[list[str]]:
    """Builds connected components of samples sharing lineage, corpus record, grammar source, or duplicate cluster."""
    pids = [r["page_id"] for r in rows if "page_id" in r]
    dsu = DSU(pids)

    # Map duplicate clusters
    pid_to_cluster_indices = defaultdict(list)
    for idx, cluster in enumerate(duplicate_clusters):
        for pid in cluster:
            pid_to_cluster_indices[pid].append(idx)

    key_to_pids = defaultdict(list)

    for r in rows:
        pid = r["page_id"]
        linkage_keys = []

        # 1. Content Lineage ID
        lineage_id = r.get("content_lineage_id")
        if not lineage_id and "metadata_groups" in r:
            lineage_id = (
                r["metadata_groups"].get("layout", {}).get("content_lineage_id")
            )
        if lineage_id:
            linkage_keys.append(f"lineage:{lineage_id}")

        # 2. Corpus Record ID
        for cid in r.get("content_record_ids", []):
            linkage_keys.append(f"corpus:{cid}")
        for record in r.get("corpus_metadata", []):
            if "corpus_record_id" in record:
                linkage_keys.append(f"corpus:{record['corpus_record_id']}")
        for zone in r.get("zones", []):
            cid = zone.get("metadata", {}).get("corpus_record_id")
            if cid:
                linkage_keys.append(f"corpus:{cid}")

        # 3. Grammar Source
        if r.get("grammar_source"):
            linkage_keys.append(f"grammar:{r.get('grammar_source')}")
        if "metadata_groups" in r and r["metadata_groups"].get("grammar_source"):
            linkage_keys.append(f"grammar:{r['metadata_groups'].get('grammar_source')}")
        for zone in r.get("zones", []):
            g_src = zone.get("metadata", {}).get("grammar_source")
            if g_src:
                linkage_keys.append(f"grammar:{g_src}")

        # 4. Duplicate Clusters
        for cluster_index in pid_to_cluster_indices.get(pid, []):
            linkage_keys.append(f"dup_cluster:{cluster_index}")

        # Map unique linkage keys
        for key in set(linkage_keys):
            key_to_pids[key].append(pid)

    # Union all page_ids sharing the same key
    for _key, shared_pids in key_to_pids.items():
        if len(shared_pids) > 1:
            first = shared_pids[0]
            for other in shared_pids[1:]:
                dsu.union(first, other)

    # Reconstruct components
    components_map = defaultdict(list)
    for pid in pids:
        root = dsu.find(pid)
        components_map[root].append(pid)

    return list(components_map.values())


def assign_splits(components: list[list[str]], total_samples: int) -> dict[str, str]:
    """Assign intact components to exact deterministic 90/5/5 split targets."""
    normalized_components = [sorted(component) for component in components if component]
    all_page_ids = [
        page_id for component in normalized_components for page_id in component
    ]
    if len(all_page_ids) != total_samples:
        raise ValueError(
            f"Component membership has {len(all_page_ids)} pages; expected {total_samples}"
        )
    if len(set(all_page_ids)) != len(all_page_ids):
        raise ValueError("A page_id appears in more than one split component")

    def get_comp_hash(c: list[str]) -> str:
        h = hashlib.sha256()
        for pid in c:
            h.update(pid.encode("utf-8"))
        return h.hexdigest()

    targets = {
        "train": int(round(total_samples * 0.90)),
        "val": int(round(total_samples * 0.05)),
    }
    targets["test"] = total_samples - targets["train"] - targets["val"]
    assigned_counts = dict.fromkeys(SPLIT_ORDER, 0)
    page_to_split: dict[str, str] = {}
    grouped = sorted(
        (component for component in normalized_components if len(component) > 1),
        key=lambda component: (-len(component), get_comp_hash(component)),
    )
    singletons = sorted(
        (component for component in normalized_components if len(component) == 1),
        key=get_comp_hash,
    )

    for comp in grouped:
        comp_size = len(comp)
        candidates = [
            split
            for split in SPLIT_ORDER
            if assigned_counts[split] + comp_size <= targets[split]
        ]
        if not candidates:
            raise ValueError(
                "Exact 90/5/5 assignment is impossible without splitting linkage "
                f"component of size {comp_size}: {comp[:3]}"
            )
        chosen_split = min(
            candidates,
            key=lambda split: (
                assigned_counts[split] / targets[split]
                if targets[split]
                else float("inf"),
                SPLIT_ORDER.index(split),
            ),
        )

        for pid in comp:
            page_to_split[pid] = chosen_split
        assigned_counts[chosen_split] += comp_size

    singleton_index = 0
    for split in SPLIT_ORDER:
        deficit = targets[split] - assigned_counts[split]
        selected = singletons[singleton_index : singleton_index + deficit]
        if len(selected) != deficit:
            raise ValueError(
                f"Exact split assignment needs {deficit} singleton pages for {split}"
            )
        for component in selected:
            page_to_split[component[0]] = split
        assigned_counts[split] += deficit
        singleton_index += deficit

    if singleton_index != len(singletons) or assigned_counts != targets:
        raise ValueError(
            f"Exact split assignment failed: assigned={assigned_counts}, targets={targets}"
        )

    return page_to_split


def stratify_and_rank(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Performs deterministic stratified ranking within a set of samples."""
    strata = defaultdict(list)

    for s in samples:
        layout_id = s.get("layout_id", "unknown")
        layout_family = LAYOUT_FAMILIES.get(layout_id, "other")
        lang_mix = s.get("language_mix", "unknown")
        primary_lang = s.get("primary_language") or "unknown"
        quality = s.get("quality_profile", "unknown")
        has_rare = has_rare_characters(s)

        stratum_key = (layout_family, lang_mix, primary_lang, quality, has_rare)
        strata[stratum_key].append(s)

    # Sort within each stratum by page_id hash deterministically
    def get_page_hash(s: dict[str, Any]) -> str:
        page_id = s.get("page_id", "")
        return hashlib.sha256(page_id.encode("utf-8")).hexdigest()

    for stratum_key in strata:
        strata[stratum_key].sort(key=get_page_hash)

    # Merge via deterministic round-robin
    sorted_keys = sorted(
        strata.keys(),
        key=lambda k: (k[0] or "", k[1] or "", k[2] or "", k[3] or "", k[4]),
    )

    queues = {key: deque(values) for key, values in strata.items()}
    ranked = []

    while any(queues.values()):
        for key in sorted_keys:
            if queues[key]:
                ranked.append(queues[key].popleft())

    return ranked


def check_leakage(
    manifest_rows: list[dict[str, Any]],
    page_to_split: dict[str, str],
    duplicate_clusters: list[list[str]],
) -> dict[str, Any]:
    """Verifies that there is zero overlap between splits on lineage, corpus records, grammar sources, or duplicates."""
    split_lineages = defaultdict(set)
    split_corpus_records = defaultdict(set)
    split_grammar_sources = defaultdict(set)

    for row in manifest_rows:
        pid = row["page_id"]
        split = page_to_split[pid]

        # Lineage
        lineage_id = row.get("content_lineage_id")
        if not lineage_id and "metadata_groups" in row:
            lineage_id = (
                row["metadata_groups"].get("layout", {}).get("content_lineage_id")
            )
        if lineage_id:
            split_lineages[split].add(lineage_id)

        # Corpus Records
        for cid in row.get("content_record_ids", []):
            split_corpus_records[split].add(cid)
        for record in row.get("corpus_metadata", []):
            if "corpus_record_id" in record:
                split_corpus_records[split].add(record["corpus_record_id"])
        for zone in row.get("zones", []):
            cid = zone.get("metadata", {}).get("corpus_record_id")
            if cid:
                split_corpus_records[split].add(cid)

        # Grammar Sources
        if row.get("grammar_source"):
            split_grammar_sources[split].add(row.get("grammar_source"))
        if "metadata_groups" in row and row["metadata_groups"].get("grammar_source"):
            split_grammar_sources[split].add(
                row["metadata_groups"].get("grammar_source")
            )
        for zone in row.get("zones", []):
            g_src = zone.get("metadata", {}).get("grammar_source")
            if g_src:
                split_grammar_sources[split].add(g_src)

    def find_overlap(d: dict[str, set]) -> int:
        overlap = 0
        splits = ["train", "val", "test"]
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap += len(d[splits[i]] & d[splits[j]])
        return overlap

    lineage_overlap = find_overlap(split_lineages)
    corpus_overlap = find_overlap(split_corpus_records)
    grammar_overlap = find_overlap(split_grammar_sources)

    duplicate_overlap = 0
    for cluster in duplicate_clusters:
        splits_in_cluster = {
            page_to_split[pid] for pid in cluster if pid in page_to_split
        }
        if len(splits_in_cluster) > 1:
            duplicate_overlap += 1

    overlap_detected = (
        lineage_overlap > 0
        or corpus_overlap > 0
        or grammar_overlap > 0
        or duplicate_overlap > 0
    )

    return {
        "overlap_detected": overlap_detected,
        "common_lineages": lineage_overlap,
        "common_corpus_records": corpus_overlap,
        "common_grammar_sources": grammar_overlap,
        "common_duplicate_clusters": duplicate_overlap,
    }


def compare_distributions(views: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Generates distribution comparison metrics of layout families, languages, quality profiles, and rare chars."""
    comparison = {}
    for view_name, view_samples in views.items():
        split_samples = {
            "train": [s for s in view_samples if s["split"] == "train"],
            "val": [s for s in view_samples if s["split"] == "val"],
            "test": [s for s in view_samples if s["split"] == "test"],
        }

        dist = {
            "layout_families": defaultdict(dict),
            "languages": defaultdict(dict),
            "quality_profiles": defaultdict(dict),
            "rare_character_presence": defaultdict(dict),
        }

        for split in ["train", "val", "test"]:
            samples = split_samples[split]
            total = len(samples)
            if total == 0:
                continue

            layout_families = Counter()
            languages = Counter()
            quality_profiles = Counter()
            rare_presence = Counter()

            for s in samples:
                layout_id = s.get("layout_id", "unknown")
                layout_family = LAYOUT_FAMILIES.get(layout_id, "other")
                layout_families[layout_family] += 1

                languages[s.get("language_mix", "unknown")] += 1
                quality_profiles[s.get("quality_profile", "unknown")] += 1
                rare_presence[has_rare_characters(s)] += 1

            for k, v in layout_families.items():
                dist["layout_families"][k][split] = round(v / total, 4)
            for k, v in languages.items():
                dist["languages"][k][split] = round(v / total, 4)
            for k, v in quality_profiles.items():
                dist["quality_profiles"][k][split] = round(v / total, 4)
            for k, v in rare_presence.items():
                dist["rare_character_presence"][str(k)][split] = round(v / total, 4)

        # Ensure split keys exist and default to 0.0
        for feature in dist:
            for cat in dist[feature]:
                for split in ["train", "val", "test"]:
                    if split not in dist[feature][cat]:
                        dist[feature][cat][split] = 0.0

        comparison[view_name] = {
            feature: dict(categories) for feature, categories in dist.items()
        }
    return comparison


def _apply_split_info(row: dict[str, Any], info: dict[str, Any]) -> None:
    row.update(info)
    groups = row.get("metadata_groups")
    if groups is None:
        return
    if not isinstance(groups, dict):
        groups = {}
        row["metadata_groups"] = groups
    release = groups.get("release")
    if isinstance(release, dict):
        release.update(info)
    groups["split_metadata"] = dict(info)


def _rewrite_manifests_with_split_info(
    out_dir: Path,
    fallback_rows: list[dict[str, Any]],
    page_split_info: dict[str, dict[str, Any]],
) -> None:
    manifest_path = out_dir / "manifest.jsonl"
    temporary_path = manifest_path.with_suffix(".jsonl.tmp")
    split_paths = {split: out_dir / f"{split}_manifest.jsonl" for split in SPLIT_ORDER}
    split_temporary_paths = {
        split: path.with_suffix(".jsonl.tmp") for split, path in split_paths.items()
    }
    split_handles = {
        split: path.open("w", encoding="utf-8")
        for split, path in split_temporary_paths.items()
    }
    seen = set()
    try:
        if manifest_path.is_file():
            with (
                manifest_path.open("r", encoding="utf-8") as source,
                temporary_path.open("w", encoding="utf-8") as output,
            ):
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    page_id = row["page_id"]
                    info = page_split_info.get(page_id)
                    if info is None:
                        raise ValueError(f"Missing split assignment for {page_id}")
                    _apply_split_info(row, info)
                    serialized = json.dumps(row, ensure_ascii=False) + "\n"
                    output.write(serialized)
                    split_handles[info["split"]].write(serialized)
                    seen.add(page_id)
        else:
            with temporary_path.open("w", encoding="utf-8") as output:
                for row in fallback_rows:
                    page_id = row["page_id"]
                    info = page_split_info.get(page_id)
                    if info is None:
                        raise ValueError(f"Missing split assignment for {page_id}")
                    _apply_split_info(row, info)
                    serialized = json.dumps(row, ensure_ascii=False) + "\n"
                    output.write(serialized)
                    split_handles[info["split"]].write(serialized)
                    seen.add(page_id)
        missing = set(page_split_info) - seen
        if missing:
            raise ValueError(f"Manifest is missing {len(missing)} assigned pages")
        for handle in split_handles.values():
            handle.close()
        os.replace(temporary_path, manifest_path)
        for split in SPLIT_ORDER:
            os.replace(split_temporary_paths[split], split_paths[split])
    finally:
        for handle in split_handles.values():
            handle.close()
        temporary_path.unlink(missing_ok=True)
        for path in split_temporary_paths.values():
            path.unlink(missing_ok=True)


def _rewrite_metadata_with_split_info(
    metadata_path: Path,
    page_split_info: dict[str, dict[str, Any]],
) -> None:
    if not metadata_path.is_file():
        return
    temporary_path = metadata_path.with_suffix(".jsonl.tmp")
    try:
        with (
            metadata_path.open("r", encoding="utf-8") as source,
            temporary_path.open("w", encoding="utf-8") as output,
        ):
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                info = page_split_info.get(row["page_id"])
                if info is not None:
                    row.update(info)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, metadata_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_dataset_splits(out_dir: Path, manifest_rows: list[dict[str, Any]]) -> None:
    """Orchestrates split assignment, deterministic stratification, nested views generation, and reports."""
    total_samples = len(manifest_rows)
    if total_samples == 0:
        return
    if total_samples > 100_000:
        raise ValueError("The release master cannot exceed 100,000 accepted samples")

    # 1. Retrieve duplicate clusters
    dup_clusters = get_duplicate_clusters(manifest_rows, out_dir)

    # 2. Build connected components of linked samples
    components = build_components(manifest_rows, dup_clusters)

    # 3. Assign splits safely (90/5/5 ratio)
    page_to_split = assign_splits(components, total_samples)
    expected_split_sizes = {
        "train": int(round(total_samples * 0.90)),
        "val": int(round(total_samples * 0.05)),
    }
    expected_split_sizes["test"] = (
        total_samples - expected_split_sizes["train"] - expected_split_sizes["val"]
    )

    # 4. Group samples by split
    split_groups = defaultdict(list)
    pid_to_row = {r["page_id"]: r for r in manifest_rows}
    for pid, split in page_to_split.items():
        row = pid_to_row[pid]
        split_groups[split].append(row)

    # 5. Deterministic stratified ranking within each split
    split_rankings = {}
    for split in ["train", "val", "test"]:
        split_rankings[split] = stratify_and_rank(split_groups[split])
        if len(split_rankings[split]) != expected_split_sizes[split]:
            raise ValueError(
                f"Split {split} has {len(split_rankings[split])} pages; "
                f"expected {expected_split_sizes[split]}"
            )

    # 6. Assign nested ranks and derive tiny/medium views
    # Determine split-specific targets for tiny and medium
    if total_samples == 100_000:
        tiny_targets = {"train": 22500, "val": 1250, "test": 1250}
        medium_targets = {"train": 45000, "val": 2500, "test": 2500}
    else:
        split_sizes = {split: len(split_rankings[split]) for split in SPLIT_ORDER}
        tiny_targets = proportional_split_targets(
            split_sizes, max(1, round(total_samples * 0.10))
        )
        medium_targets = proportional_split_targets(
            split_sizes, max(1, round(total_samples * 0.40))
        )

    # Store split, nested rank and subset membership
    page_split_info = {}
    views = {config: [] for config in RELEASE_CONFIGS}

    for split in ["train", "val", "test"]:
        ranking = split_rankings[split]
        t_target = tiny_targets[split]
        m_target = medium_targets[split]

        for rank, row in enumerate(ranking):
            pid = row["page_id"]
            in_tiny = rank < t_target
            in_medium = rank < m_target

            subsets = [LARGE_CONFIG.name]
            if in_medium:
                subsets.insert(0, MEDIUM_CONFIG.name)
            if in_tiny:
                subsets.insert(0, TINY_CONFIG.name)

            info = {
                "split": split,
                "nested_rank": rank,
                "in_tiny": in_tiny,
                "in_medium": in_medium,
                "in_large": True,
                "subsets": subsets,
            }
            page_split_info[pid] = info

            # Add to views
            views[LARGE_CONFIG.name].append({**row, **info})
            if in_medium:
                views[MEDIUM_CONFIG.name].append({**row, **info})
            if in_tiny:
                views[TINY_CONFIG.name].append({**row, **info})

    # 7. Update manifest_rows and write out updated files
    for row in manifest_rows:
        pid = row["page_id"]
        info = page_split_info[pid]
        _apply_split_info(row, info)

    _rewrite_manifests_with_split_info(out_dir, manifest_rows, page_split_info)

    # Rewrite metadata.jsonl with same fields
    metadata_path = out_dir / "metadata.jsonl"
    _rewrite_metadata_with_split_info(metadata_path, page_split_info)

    # Regenerate metadata.parquet if pyarrow is installed
    from turkicdocgen.export import write_metadata_parquet_if_available

    write_metadata_parquet_if_available(out_dir)

    # 8. Output nested subset indexes
    for subset_name in [TINY_CONFIG.name, MEDIUM_CONFIG.name]:
        subset_samples = views[subset_name]
        subset_splits = {
            "train": [s["page_id"] for s in subset_samples if s["split"] == "train"],
            "val": [s["page_id"] for s in subset_samples if s["split"] == "val"],
            "test": [s["page_id"] for s in subset_samples if s["split"] == "test"],
        }
        all_subset_pids = [s["page_id"] for s in subset_samples]

        index_data = {
            "subset": subset_name,
            "total_count": len(all_subset_pids),
            "splits": subset_splits,
            "page_ids": all_subset_pids,
        }
        index_path = out_dir / f"{subset_name}_index.json"
        index_path.write_text(
            json.dumps(index_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 9. Generate leakage_report.json
    leak_chk = check_leakage(manifest_rows, page_to_split, dup_clusters)
    dist_comp = compare_distributions(views)

    leakage_report_data = {
        "split_sizes": {
            "train": len(split_rankings["train"]),
            "val": len(split_rankings["val"]),
            "test": len(split_rankings["test"]),
        },
        "overlap_check": leak_chk,
        "distribution_comparison": dist_comp,
    }

    leakage_report_path = out_dir / "reports" / "leakage_report.json"
    leakage_report_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_report_path.write_text(
        json.dumps(leakage_report_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if leak_chk["overlap_detected"]:
        raise ValueError("Leakage detected between train, validation, and test splits")
    if total_samples == 100_000:
        if len(views[TINY_CONFIG.name]) != TINY_CONFIG.target_rows:
            raise ValueError("tiny view does not contain exactly 25,000 samples")
        if len(views[MEDIUM_CONFIG.name]) != MEDIUM_CONFIG.target_rows:
            raise ValueError("medium view does not contain exactly 50,000 samples")
