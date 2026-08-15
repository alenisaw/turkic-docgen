from __future__ import annotations

import binascii
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) of an image using PIL and numpy.

    Grayscale conversion, resizing to (hash_size + 1, hash_size),
    comparing adjacent horizontal pixels, and packing the resulting bits
    into a 64-bit integer.
    """
    gray = image.convert("L")
    resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.int32)
    # Compare adjacent columns
    diff = pixels[:, :hash_size] > pixels[:, 1:]
    flat = diff.flatten()
    hash_val = 0
    for i, bit in enumerate(flat):
        if bit:
            hash_val |= 1 << i
    return hash_val


def compute_full_page_dhash(image_path: str | Path, hash_size: int = 8) -> int:
    """Computes the dHash of a fully rendered page by loading and resizing the image from disk."""
    with Image.open(image_path) as img:
        return compute_dhash(img, hash_size)


def compute_file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file digest without loading the complete rendered page into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _drawing_dimensions(
    plan_or_dict: Any,
    output_size: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    if isinstance(plan_or_dict, dict):
        width = plan_or_dict.get("width")
        height = plan_or_dict.get("height")
        orientation = plan_or_dict.get("orientation", "portrait")
    else:
        width = getattr(plan_or_dict, "width", None)
        height = getattr(plan_or_dict, "height", None)
        orientation = getattr(plan_or_dict, "orientation", "portrait")
        if not orientation and hasattr(plan_or_dict, "metadata"):
            orientation = plan_or_dict.metadata.get("orientation", "portrait")
    if width is None or height is None:
        width, height = (2339, 1654) if orientation == "landscape" else (1654, 2339)
    draw_width, draw_height = output_size or (width, height)
    return width, height, draw_width, draw_height


def _scaled_bbox(
    bbox: Any,
    source_width: int,
    source_height: int,
    draw_width: int,
    draw_height: int,
) -> tuple[int, int, int, int] | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    return (
        round(x1 * draw_width / source_width),
        round(y1 * draw_height / source_height),
        round(x2 * draw_width / source_width),
        round(y2 * draw_height / source_height),
    )


def get_skeleton_image(
    plan_or_dict: Any,
    output_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Generates the layout skeleton image (zones, table cell grid lines, form row outlines)."""
    if isinstance(plan_or_dict, dict):
        zones = plan_or_dict.get("zones", [])
    else:
        zones = getattr(plan_or_dict, "zones", [])
    width, height, draw_width, draw_height = _drawing_dimensions(
        plan_or_dict, output_size
    )
    img = Image.new("L", (draw_width, draw_height), 255)
    draw = ImageDraw.Draw(img)
    boundary_width = max(1, round(min(draw_width, draw_height) / 220))
    detail_width = max(1, boundary_width // 2)

    for zone in zones:
        if isinstance(zone, dict):
            bbox = zone.get("bbox")
            cells = zone.get("cells", [])
            metadata = zone.get("metadata") or {}
            zone_type = zone.get("zone_type", "")
        else:
            bbox = getattr(zone, "bbox", None)
            cells = getattr(zone, "cells", [])
            metadata = getattr(zone, "metadata", {}) or {}
            zone_type = getattr(zone, "zone_type", "")

        scaled_bbox = _scaled_bbox(bbox, width, height, draw_width, draw_height)
        if scaled_bbox:
            if zone_type == "decorative_non_text":
                draw.rectangle(scaled_bbox, fill=0)
            else:
                draw.rectangle(
                    scaled_bbox,
                    outline=0,
                    fill=None,
                    width=boundary_width,
                )

        if cells:
            for cell in cells:
                if isinstance(cell, dict):
                    c_bbox = cell.get("bbox")
                else:
                    c_bbox = getattr(cell, "bbox", None)
                scaled_cell_bbox = _scaled_bbox(
                    c_bbox, width, height, draw_width, draw_height
                )
                if scaled_cell_bbox:
                    draw.rectangle(
                        scaled_cell_bbox,
                        outline=0,
                        fill=None,
                        width=detail_width,
                    )

        rendered_fields = metadata.get("rendered_fields", [])
        if rendered_fields:
            for field in rendered_fields:
                row_bbox = field.get("row_bbox")
                scaled_row_bbox = _scaled_bbox(
                    row_bbox, width, height, draw_width, draw_height
                )
                if scaled_row_bbox:
                    draw.rectangle(
                        scaled_row_bbox,
                        outline=0,
                        fill=None,
                        width=detail_width,
                    )

    return img


def compute_layout_skeleton_dhash(plan_or_dict: Any, hash_size: int = 8) -> int:
    """Computes the dHash of the layout skeleton representation of a page plan."""
    img = get_skeleton_image(plan_or_dict, (4 * (hash_size + 1), 4 * hash_size))
    return compute_dhash(img, hash_size)


def compute_layout_structure_hash(plan_or_dict: Any) -> int:
    zones = (
        plan_or_dict.get("zones", [])
        if isinstance(plan_or_dict, dict)
        else getattr(plan_or_dict, "zones", [])
    )
    structure = []
    for zone in zones:
        if isinstance(zone, dict):
            zone_type = zone.get("zone_type", "")
            bbox = zone.get("bbox")
            cells = zone.get("cells", [])
            metadata = zone.get("metadata") or {}
        else:
            zone_type = getattr(zone, "zone_type", "")
            bbox = getattr(zone, "bbox", None)
            cells = getattr(zone, "cells", [])
            metadata = getattr(zone, "metadata", {}) or {}
        cell_boxes = [
            tuple(cell.get("bbox") or ())
            if isinstance(cell, dict)
            else tuple(getattr(cell, "bbox", ()))
            for cell in cells
        ]
        row_boxes = [
            tuple(field.get("row_bbox") or ())
            for field in metadata.get("rendered_fields", [])
            if isinstance(field, dict)
        ]
        structure.append(
            (
                str(zone_type),
                tuple(bbox or ()),
                tuple(cell_boxes),
                tuple(row_boxes),
            )
        )
    payload = json.dumps(structure, separators=(",", ":"), ensure_ascii=True)
    return int(hashlib.sha256(payload.encode("ascii")).hexdigest(), 16)


def get_binarized_page_mask_image(
    plan_or_dict: Any,
    output_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Generates the binarized page mask image (solid black rectangles for text lines, cell bboxes,

    and form label/value bboxes).
    """
    if isinstance(plan_or_dict, dict):
        zones = plan_or_dict.get("zones", [])
    else:
        zones = getattr(plan_or_dict, "zones", [])
    width, height, draw_width, draw_height = _drawing_dimensions(
        plan_or_dict, output_size
    )
    img = Image.new("L", (draw_width, draw_height), 255)
    draw = ImageDraw.Draw(img)

    for zone in zones:
        if isinstance(zone, dict):
            lines = zone.get("lines", [])
            cells = zone.get("cells", [])
            metadata = zone.get("metadata") or {}
        else:
            lines = getattr(zone, "lines", [])
            cells = getattr(zone, "cells", [])
            metadata = getattr(zone, "metadata", {}) or {}

        # Draw solid black rectangles for lines
        if lines:
            for line in lines:
                if isinstance(line, dict):
                    l_bbox = line.get("bbox")
                else:
                    l_bbox = getattr(line, "bbox", None)
                scaled_line_bbox = _scaled_bbox(
                    l_bbox, width, height, draw_width, draw_height
                )
                if scaled_line_bbox:
                    draw.rectangle(scaled_line_bbox, fill=0)

        # Draw solid black rectangles for cells
        if cells:
            for cell in cells:
                if isinstance(cell, dict):
                    c_bbox = cell.get("bbox")
                else:
                    c_bbox = getattr(cell, "bbox", None)
                scaled_cell_bbox = _scaled_bbox(
                    c_bbox, width, height, draw_width, draw_height
                )
                if scaled_cell_bbox:
                    draw.rectangle(scaled_cell_bbox, fill=0)

        # Draw solid black rectangles for form labels and values
        rendered_fields = metadata.get("rendered_fields", [])
        if rendered_fields:
            for field in rendered_fields:
                lbl_bbox = field.get("label_bbox")
                scaled_label_bbox = _scaled_bbox(
                    lbl_bbox, width, height, draw_width, draw_height
                )
                if scaled_label_bbox:
                    draw.rectangle(scaled_label_bbox, fill=0)
                val_bbox = field.get("value_bbox")
                scaled_value_bbox = _scaled_bbox(
                    val_bbox, width, height, draw_width, draw_height
                )
                if scaled_value_bbox:
                    draw.rectangle(scaled_value_bbox, fill=0)

    return img


def compute_binarized_page_mask_dhash(plan_or_dict: Any, hash_size: int = 8) -> int:
    """Compute a fixed-width occupancy hash of the binarized page mask.

    Unlike dHash, occupancy preserves the sparse geometry itself instead of
    mostly encoding identical white-to-white transitions.
    """
    img = get_binarized_page_mask_image(
        plan_or_dict, (4 * hash_size, 4 * hash_size)
    ).resize((hash_size, hash_size), Image.Resampling.BOX)
    occupied = np.asarray(img, dtype=np.uint8) < 224
    value = 0
    for bit in occupied.ravel():
        value = (value << 1) | int(bit)
    return value


def extract_meaningful_text_segments(plan_or_dict: Any) -> list[str]:
    """Extracts non-boilerplate text segments (excludes headers, footers, page numbers, form labels,

    table headers, stamps, and decorative zones).
    """
    segments = []
    if isinstance(plan_or_dict, dict):
        zones = plan_or_dict.get("zones", [])
    else:
        zones = getattr(plan_or_dict, "zones", [])

    for zone in zones:
        if isinstance(zone, dict):
            zone_type = zone.get("zone_type", "")
            role = zone.get("role", "")
            if not role:
                role = zone.get("metadata", {}).get("role", "")
            cells = zone.get("cells", [])
            lines = zone.get("lines", [])
            metadata = zone.get("metadata") or {}
            text = zone.get("text", "")
        else:
            zone_type = getattr(zone, "zone_type", "")
            role = getattr(zone, "role", "")
            if not role and hasattr(zone, "metadata"):
                role = zone.metadata.get("role", "")
            cells = getattr(zone, "cells", [])
            lines = getattr(zone, "lines", [])
            metadata = getattr(zone, "metadata", {}) or {}
            text = getattr(zone, "text", "")

        role_lower = str(role).lower()
        type_lower = str(zone_type).lower()

        # Filter out boilerplate zones
        if any(
            token in role_lower or token in type_lower
            for token in ("header", "footer", "page_number", "stamp", "decorative")
        ):
            continue

        if cells:
            for cell in cells:
                if isinstance(cell, dict):
                    row_idx = cell.get("row", 0)
                    c_text = cell.get("text", "")
                else:
                    row_idx = getattr(cell, "row", 0)
                    c_text = getattr(cell, "text", "")
                # Exclude table headers (row 0)
                if row_idx > 0 and c_text.strip():
                    segments.append(c_text.strip())
        elif metadata.get("rendered_fields"):
            for field in metadata["rendered_fields"]:
                val = field.get("value_text", "")
                if val.strip():
                    segments.append(val.strip())
        else:
            if lines:
                for line in lines:
                    if isinstance(line, dict):
                        l_text = line.get("text", "")
                    else:
                        l_text = getattr(line, "text", "")
                    if l_text.strip():
                        segments.append(l_text.strip())
            elif text.strip():
                for line in text.splitlines():
                    if line.strip():
                        segments.append(line.strip())

    return segments


def normalize_text(value: str) -> str:
    """Identical to audit.py's normalize_text: lowercases, casefolds, replaces punctuation,

    and collapses spaces.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def get_exact_meaningful_text(plan_or_dict: Any) -> str:
    """Concatenates all non-boilerplate text segments."""
    segments = extract_meaningful_text_segments(plan_or_dict)
    return "\n".join(segments)


def get_exact_meaningful_text_hash(plan_or_dict: Any) -> str:
    """Computes SHA256 of the exact meaningful text."""
    text = get_exact_meaningful_text(plan_or_dict)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_normalized_meaningful_text(plan_or_dict: Any) -> str:
    """Concatenates all normalized non-boilerplate text segments."""
    segments = extract_meaningful_text_segments(plan_or_dict)
    normalized = [normalize_text(s) for s in segments if s.strip()]
    return " ".join([s for s in normalized if s])


def get_normalized_meaningful_text_hash(plan_or_dict: Any) -> str:
    """Computes SHA256 of the normalized meaningful text."""
    text = get_normalized_meaningful_text(plan_or_dict)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_shingles(text: str, k: int = 5) -> set[str]:
    """Generates character k-shingles from text."""
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def get_word_shingles(text: str, k: int = 3) -> set[str]:
    words = normalize_text(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + k]) for index in range(len(words) - k + 1)}


def get_candidate_shingles(text: str, k: int = 4) -> set[str]:
    return get_word_shingles(text, k)


def get_similarity_shingles(text: str, k: int = 4) -> set[str]:
    words = [re.sub(r"\d+", "<num>", word) for word in normalize_text(text).split()]
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + k]) for index in range(len(words) - k + 1)}


def generate_minhash_params(num_perm: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Generates random coefficients A and B for MinHash permutations."""
    rng = np.random.default_rng(42)
    # Largest prime below 2**32 keeps A*x+B inside uint64.
    prime = 4294967291
    A = rng.integers(1, prime - 1, size=num_perm, dtype=np.uint64)
    B = rng.integers(0, prime - 1, size=num_perm, dtype=np.uint64)
    return A, B


def compute_minhash_signature(
    shingles: set[str], A: np.ndarray, B: np.ndarray, num_perm: int = 128
) -> np.ndarray:
    """Vectorized calculation of MinHash signature for a set of shingles."""
    if not shingles:
        return np.full(num_perm, 4294967291, dtype=np.uint64)

    # Convert shingles to CRC32 hashes
    shingle_hashes = np.array(
        [binascii.crc32(s.encode("utf-8")) & 0xFFFFFFFF for s in shingles],
        dtype=np.uint64,
    )
    shingle_hashes = shingle_hashes.reshape(-1, 1)

    prime = 4294967291
    # h(x) = (A * x + B) % prime
    mapped = (shingle_hashes * A + B) % prime
    return np.min(mapped, axis=0)


def compute_one_permutation_minhash(
    shingles: set[str], num_perm: int = 64
) -> np.ndarray:
    if num_perm <= 0 or num_perm & (num_perm - 1):
        raise ValueError("num_perm must be a positive power of two")
    max_value = np.iinfo(np.uint32).max
    signature = np.full(num_perm, max_value, dtype=np.uint32)
    mask = num_perm - 1
    for shingle in shingles:
        encoded = shingle.encode("utf-8")
        bucket_hash = binascii.crc32(encoded) & 0xFFFFFFFF
        value_hash = binascii.crc32(encoded, 0x9E3779B9) & 0xFFFFFFFF
        bucket = bucket_hash & mask
        if value_hash < signature[bucket]:
            signature[bucket] = value_hash
    populated = np.flatnonzero(signature != max_value)
    if not len(populated):
        return signature
    for bucket in np.flatnonzero(signature == max_value):
        distances = (populated - bucket) % num_perm
        nearest_index = int(np.argmin(distances))
        source_bucket = int(populated[nearest_index])
        distance = int(distances[nearest_index])
        signature[bucket] = np.uint32(
            (int(signature[source_bucket]) + distance * 0x9E3779B1) & 0xFFFFFFFF
        )
    return signature


def group_near_duplicates_minhash_lsh(
    texts: list[str],
    ids: list[str],
    threshold: float = 0.85,
    num_perm: int = 64,
    b: int = 16,
    r: int = 4,
) -> list[list[str]]:
    """Clusters near-duplicate texts (Jaccard similarity >= threshold) using sharded MinHash LSH

    and a Disjoint Set Union (DSU) connected components structure.
    """
    if b <= 0 or r <= 0 or b * r != num_perm:
        raise ValueError("MinHash LSH requires positive b and r with b * r == num_perm")
    if len(texts) != len(ids):
        raise ValueError("texts and ids must have the same length")

    signatures = np.empty((len(texts), num_perm), dtype=np.uint32)
    for index, text in enumerate(texts):
        signatures[index] = compute_one_permutation_minhash(
            get_candidate_shingles(text), num_perm
        )

    # DSU grouping
    parent = list(range(len(ids)))

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    normalized_shingles: dict[int, set[str]] = {}

    def exact_shingles(document_index: int) -> set[str]:
        shingles = normalized_shingles.get(document_index)
        if shingles is None:
            shingles = get_similarity_shingles(texts[document_index], 4)
            normalized_shingles[document_index] = shingles
        return shingles

    # A pair can collide in several bands. Process it only in the first matching
    # band, avoiding an unbounded global candidate-pair set.
    for band_idx in range(b):
        band_start = band_idx * r
        band_end = band_start + r
        band_values = signatures[:, band_start:band_end]
        order = np.lexsort(band_values.T[::-1])
        group_start = 0
        while group_start < len(order):
            group_end = group_start + 1
            first_doc = order[group_start]
            while group_end < len(order) and np.array_equal(
                band_values[first_doc], band_values[order[group_end]]
            ):
                group_end += 1
            bucket_docs = order[group_start:group_end]
            for left_pos, d1 in enumerate(bucket_docs):
                for d2 in bucket_docs[left_pos + 1 :]:
                    matched_earlier = any(
                        np.array_equal(
                            signatures[
                                d1, candidate_band * r : (candidate_band + 1) * r
                            ],
                            signatures[
                                d2, candidate_band * r : (candidate_band + 1) * r
                            ],
                        )
                        for candidate_band in range(band_idx)
                    )
                    if matched_earlier:
                        continue
                    shingles1 = exact_shingles(int(d1))
                    shingles2 = exact_shingles(int(d2))
                    if not shingles1 and not shingles2:
                        similarity = 1.0
                    elif not shingles1 or not shingles2:
                        similarity = 0.0
                    else:
                        similarity = len(shingles1 & shingles2) / len(
                            shingles1 | shingles2
                        )
                    if similarity >= threshold:
                        union(d1, d2)
            group_start = group_end

    components = defaultdict(list)
    for idx, page_id in enumerate(ids):
        root = find(idx)
        components[root].append(page_id)

    return [cluster for cluster in components.values() if len(cluster) > 1]


def compute_hamming_distance(hash1: int, hash2: int) -> int:
    """Compute exact Hamming distance between two fixed-width integer hashes."""
    return (hash1 ^ hash2).bit_count()


def cluster_hashes_hamming_lsh(
    hashes: list[int],
    ids: list[str],
    threshold: int = 3,
    hash_bits: int = 64,
) -> list[list[str]]:
    """Cluster hashes exactly using interleaved pigeonhole blocking.

    Hashes within radius ``threshold`` must share one of ``threshold + 1``
    exact bit partitions. Interleaving positions across the full hash avoids
    the massive low-entropy buckets produced by contiguous image regions.
    """
    if len(hashes) != len(ids):
        raise ValueError("hashes and ids must have the same length")
    if threshold < 0 or hash_bits <= threshold:
        raise ValueError("hash_bits must be greater than a non-negative threshold")
    if any(
        hash_value < 0 or hash_value.bit_length() > hash_bits for hash_value in hashes
    ):
        raise ValueError("hash values must be non-negative and fit within hash_bits")

    parent = list(range(len(ids)))

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    if threshold == 0:
        representatives: dict[int, int] = {}
        for index, hash_value in enumerate(hashes):
            representative = representatives.setdefault(hash_value, index)
            if representative != index:
                union(representative, index)
    else:
        partition_count = threshold + 1
        partition_width = math.ceil(hash_bits / partition_count)
        use_uint64 = partition_width <= 64
        signatures = (
            np.zeros((len(hashes), partition_count), dtype=np.uint64)
            if use_uint64
            else [[0] * partition_count for _ in hashes]
        )
        for hash_index, hash_value in enumerate(hashes):
            remaining = hash_value
            while remaining:
                least_significant = remaining & -remaining
                bit_index = least_significant.bit_length() - 1
                partition = bit_index % partition_count
                output_bit = bit_index // partition_count
                if use_uint64:
                    signatures[hash_index, partition] |= np.uint64(1 << output_bit)
                else:
                    signatures[hash_index][partition] |= 1 << output_bit
                remaining ^= least_significant

        for partition in range(partition_count):
            if use_uint64:
                values = signatures[:, partition]
                order = np.argsort(values, kind="stable")
            else:
                values = [signature[partition] for signature in signatures]
                order = sorted(range(len(hashes)), key=values.__getitem__)
            group_start = 0
            while group_start < len(order):
                group_end = group_start + 1
                first_index = int(order[group_start])
                while (
                    group_end < len(order)
                    and values[int(order[group_end])] == values[first_index]
                ):
                    group_end += 1
                candidates = order[group_start:group_end]
                for left_position, left_raw in enumerate(candidates):
                    left = int(left_raw)
                    for right_raw in candidates[left_position + 1 :]:
                        right = int(right_raw)
                        if partition:
                            matched_earlier = (
                                np.any(
                                    signatures[left, :partition]
                                    == signatures[right, :partition]
                                )
                                if use_uint64
                                else any(
                                    signatures[left][earlier]
                                    == signatures[right][earlier]
                                    for earlier in range(partition)
                                )
                            )
                            if matched_earlier:
                                continue
                        if (
                            compute_hamming_distance(hashes[left], hashes[right])
                            <= threshold
                        ):
                            union(left, right)
                group_start = group_end

    components = defaultdict(list)
    for idx, page_id in enumerate(ids):
        root = find(idx)
        components[root].append(page_id)

    return [cluster for cluster in components.values() if len(cluster) > 1]


def generate_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    thumb_size: tuple[int, int] = (120, 150),
    cols: int = 5,
    max_images: int = 25,
) -> None:
    """Generates a contact sheet grid image from list of page image paths."""
    image_paths = image_paths[:max_images]
    n = len(image_paths)
    if n == 0:
        return

    rows = math.ceil(n / cols)
    tw, th = thumb_size

    grid_img = Image.new("RGB", (cols * tw, rows * th), (255, 255, 255))

    for idx, img_path in enumerate(image_paths):
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as img:
                thumb = img.convert("RGB").resize((tw, th), Image.Resampling.BILINEAR)
                c = idx % cols
                r = idx // cols
                grid_img.paste(thumb, (c * tw, r * th))
        except Exception:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(output_path)
