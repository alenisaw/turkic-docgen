from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import pytest
from PIL import Image

from turkicdocgen.dedup import (
    cluster_hashes_hamming_lsh,
    compute_binarized_page_mask_dhash,
    compute_dhash,
    compute_file_sha256,
    compute_full_page_dhash,
    compute_hamming_distance,
    compute_layout_skeleton_dhash,
    compute_minhash_signature,
    extract_meaningful_text_segments,
    generate_contact_sheet,
    generate_minhash_params,
    get_exact_meaningful_text,
    get_exact_meaningful_text_hash,
    get_normalized_meaningful_text,
    get_normalized_meaningful_text_hash,
    group_near_duplicates_minhash_lsh,
)
from turkicdocgen.page_planning.content.audit import generate_iteration8_reports


def test_dhash() -> None:
    # Test dHash with simple solid image
    img1 = Image.new("RGB", (100, 100), (255, 255, 255))
    h1 = compute_dhash(img1)
    # Since all pixels are identical, h1 should be 0 or 0xFFFFFFFFFFFFFFFF depending on float roundoff/comparison
    # Let's verify it returns an integer
    assert isinstance(h1, int)

    # Test dHash with a gradient
    img2 = Image.new("L", (100, 100))
    pixels = img2.load()
    for y in range(100):
        for x in range(100):
            pixels[x, y] = x * 2  # Gradient
    h2 = compute_dhash(img2)
    assert isinstance(h2, int)


def test_perceptual_page_representations() -> None:
    # Mock PagePlan as a dict
    mock_plan = {
        "width": 800,
        "height": 1000,
        "zones": [
            {
                "bbox": [50, 50, 750, 150],
                "zone_type": "title",
                "role": "title",
                "lines": [{"bbox": [50, 50, 750, 150], "text": "Сынақ Тақырыбы"}],
            },
            {
                "bbox": [50, 200, 750, 500],
                "zone_type": "table",
                "cells": [
                    {
                        "row": 0,
                        "col": 0,
                        "bbox": [50, 200, 400, 250],
                        "text": "Кесте Басы",
                    },
                    {
                        "row": 1,
                        "col": 0,
                        "bbox": [50, 250, 400, 300],
                        "text": "Ақпарат",
                    },
                ],
            },
            {
                "bbox": [50, 550, 750, 800],
                "zone_type": "form",
                "metadata": {
                    "rendered_fields": [
                        {
                            "label_bbox": [50, 560, 200, 600],
                            "value_bbox": [210, 560, 750, 600],
                            "row_bbox": [50, 550, 750, 610],
                            "label_text": "Аты:",
                            "value_text": "Әлібек",
                        }
                    ]
                },
            },
        ],
    }

    h_skeleton = compute_layout_skeleton_dhash(mock_plan)
    h_mask = compute_binarized_page_mask_dhash(mock_plan)

    assert isinstance(h_skeleton, int)
    assert isinstance(h_mask, int)


def test_extract_meaningful_text() -> None:
    mock_plan = {
        "zones": [
            {
                "zone_type": "header",
                "role": "header",
                "text": "Бойлерплейт тақырыбы",
            },
            {
                "zone_type": "title",
                "text": "Маңызды тақырып",
            },
            {
                "zone_type": "table",
                "cells": [
                    {"row": 0, "col": 0, "text": "Кесте тақырыбы"},
                    {"row": 1, "col": 0, "text": "Мәлімет жолы"},
                ],
            },
            {
                "zone_type": "form",
                "metadata": {
                    "rendered_fields": [
                        {
                            "label_text": "Аты:",
                            "value_text": "Асан",
                        }
                    ]
                },
            },
            {
                "zone_type": "footer",
                "role": "page_number",
                "text": "12-бет",
            },
        ]
    }

    segments = extract_meaningful_text_segments(mock_plan)
    # Should exclude header (row 0) and boilerplate zones (header/footer/page_number)
    # Should keep: "Маңызды тақырып", "Мәлімет жолы" (table row 1), "Асан" (form value)
    assert "Бойлерплейт тақырыбы" not in segments
    assert "Кесте тақырыбы" not in segments
    assert "12-бет" not in segments
    assert "Маңызды тақырып" in segments
    assert "Мәлімет жолы" in segments
    assert "Асан" in segments

    exact_text = get_exact_meaningful_text(mock_plan)
    assert (
        "Маңызды тақырып\nМәлімет жолы\nАсан" in exact_text
        or exact_text == "Маңызды тақырып\nМәлімет жолы\nАсан"
    )

    exact_hash = get_exact_meaningful_text_hash(mock_plan)
    assert len(exact_hash) == 64

    norm_text = get_normalized_meaningful_text(mock_plan)
    assert "маңызды тақырып мәлімет жолы асан" in norm_text

    norm_hash = get_normalized_meaningful_text_hash(mock_plan)
    assert len(norm_hash) == 64


def test_group_near_duplicates_minhash_lsh() -> None:
    texts = [
        "Қазақ тіліндегі бірінші құжат мәтіні тексеру үшін",
        "Қазақ тіліндегі бірінші құжат мәтіні тексеру үшін.",
        "Мүлдем басқа мәтін мұнда жазылған",
    ]
    ids = ["doc1", "doc2", "doc3"]
    clusters = group_near_duplicates_minhash_lsh(texts, ids, threshold=0.8)
    assert len(clusters) == 1
    assert "doc1" in clusters[0]
    assert "doc2" in clusters[0]
    assert "doc3" not in clusters[0]


def test_minhash_signature_is_deterministic_and_unsigned() -> None:
    coefficients_a, coefficients_b = generate_minhash_params()
    first = compute_minhash_signature(
        {"қазақ", "кыргыз", "түркі"}, coefficients_a, coefficients_b
    )
    second = compute_minhash_signature(
        {"түркі", "қазақ", "кыргыз"}, coefficients_a, coefficients_b
    )
    assert first.dtype.name == "uint64"
    assert first.tolist() == second.tolist()


def test_minhash_lsh_does_not_truncate_large_candidate_bucket() -> None:
    texts = [
        f"Бірдей ұзақ құжат мәтіні және бірегей нөмір {index % 2}"
        for index in range(130)
    ]
    ids = [f"doc-{index:03d}" for index in range(len(texts))]
    clusters = group_near_duplicates_minhash_lsh(texts, ids, threshold=0.8)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ids


def test_minhash_lsh_detects_reordered_content_blocks() -> None:
    first = " ".join(f"token{index}" for index in range(40))
    words = first.split()
    second = " ".join(words[20:] + words[:20])

    clusters = group_near_duplicates_minhash_lsh(
        [first, second],
        ["first", "second"],
        threshold=0.8,
    )

    assert clusters == [["first", "second"]]


def test_cluster_hashes_hamming_lsh() -> None:
    # 64-bit integers
    h1 = 0b1111111111111111000000000000000000000000000000000000000000000000
    h2 = 0b1111111111111111000000000000000000000000000000000000000000000011  # Hamming distance = 2
    h3 = 0b0000000000000000111111111111111111111111111111111111111111111111  # Distant

    hashes = [h1, h2, h3]
    ids = ["id1", "id2", "id3"]
    clusters = cluster_hashes_hamming_lsh(hashes, ids, threshold=3)
    assert len(clusters) == 1
    assert "id1" in clusters[0]
    assert "id2" in clusters[0]


@pytest.mark.parametrize(("hash_bits", "threshold"), [(64, 3), (1024, 16)])
def test_hamming_clustering_matches_brute_force(hash_bits: int, threshold: int) -> None:
    rng = random.Random(90210 + hash_bits)
    hashes = [rng.getrandbits(hash_bits) for _ in range(60)]
    hashes.extend(
        [
            hashes[0],
            hashes[1] ^ ((1 << threshold) - 1),
            hashes[2] ^ 1,
            hashes[2] ^ 3,
        ]
    )
    ids = [f"id-{index}" for index in range(len(hashes))]

    parent = list(range(len(hashes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if compute_hamming_distance(hashes[left], hashes[right]) <= threshold:
                union(left, right)
    expected_groups: dict[int, list[str]] = defaultdict(list)
    for index, page_id in enumerate(ids):
        expected_groups[find(index)].append(page_id)
    expected = sorted(
        sorted(group) for group in expected_groups.values() if len(group) > 1
    )

    actual = sorted(
        sorted(group)
        for group in cluster_hashes_hamming_lsh(
            hashes, ids, threshold=threshold, hash_bits=hash_bits
        )
    )
    assert actual == expected


def test_hamming_clustering_handles_large_low_entropy_bucket() -> None:
    rng = random.Random(10101)
    hashes = [(rng.getrandbits(963) << 61) for _ in range(10_000)]
    hashes.extend([hashes[0], hashes[1] ^ 1])
    ids = [f"id-{index}" for index in range(len(hashes))]

    clusters = cluster_hashes_hamming_lsh(hashes, ids, threshold=1, hash_bits=1024)

    assert sorted(sorted(cluster) for cluster in clusters) == [
        ["id-0", "id-10000"],
        ["id-1", "id-10001"],
    ]


def test_contact_sheet_and_reports(tmp_path: Path) -> None:
    # Create temp images to test contact sheet
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path1 = img_dir / "page1.png"
    img_path2 = img_dir / "page2.png"

    Image.new("RGB", (200, 300), (255, 0, 0)).save(img_path1)
    Image.new("RGB", (200, 300), (0, 255, 0)).save(img_path2)

    # Test contact sheet generation
    contact_path = tmp_path / "reports" / "contact_sheet.png"
    generate_contact_sheet([img_path1, img_path2], contact_path)
    assert contact_path.exists()
    assert isinstance(compute_full_page_dhash(img_path1), int)
    assert compute_file_sha256(img_path1) != compute_file_sha256(img_path2)

    # Test audit reporting
    rows = [
        {
            "page_id": "page1",
            "image": "images/page1.png",
            "layout_id": "book_page_single_column",
            "zones": [
                {
                    "zone_type": "body",
                    "text": "Маңызды құжат мәтіні осында орналасқан.",
                }
            ],
        },
        {
            "page_id": "page2",
            "image": "images/page2.png",
            "layout_id": "book_page_single_column",
            "zones": [
                {
                    "zone_type": "body",
                    "text": "Маңызды құжат мәтіні осында орналасқан.",  # Duplicate text
                }
            ],
        },
    ]

    generate_iteration8_reports(tmp_path, rows, profile_name="visual_300")
    report_file = tmp_path / "reports" / "duplicate_report.json"
    assert report_file.exists()

    with open(report_file, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data["exact_meaningful_text_duplicates"]) == 1
    assert data["exact_meaningful_text_duplicates"][0] == ["page1", "page2"]
    assert data["normalized_meaningful_text_duplicates"] == [["page1", "page2"]]
    assert data["structural_layout_clusters"] == [["page1", "page2"]]
    assert data["page_mask_clusters"] == [["page1", "page2"]]
    assert data["metrics"]["audited_images"] == 2
    assert data["metrics"]["near_full_page_count"] == 1
    assert data["metrics"]["near_full_page_involved_pages"] == 2
    assert data["exact_full_page_duplicates"] == []
    assert (tmp_path / "reports" / "duplicate_clusters.jsonl").is_file()


def test_iteration8_reports_missing_images_as_incomplete_audit(tmp_path: Path) -> None:
    rows = [
        {
            "page_id": "missing",
            "image": "images/missing.png",
            "layout_id": "book_page_single_column",
            "zones": [{"zone_type": "body", "text": "Бірегей мазмұн"}],
        }
    ]

    generate_iteration8_reports(tmp_path, rows, profile_name="visual_300")
    data = json.loads(
        (tmp_path / "reports" / "duplicate_report.json").read_text(encoding="utf-8")
    )
    assert data["gates_passed"] is False
    assert data["metrics"]["audited_images"] == 0
    assert data["audit_errors"]["images"][0]["error"] == "image_missing"


def test_iteration8_reuses_exported_effect_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "cached.png"
    Image.new("RGB", (200, 300), "white").save(image_path)
    cached_hash = format(compute_full_page_dhash(image_path, hash_size=32), "0256x")

    def fail_redecode(*args, **kwargs):
        raise AssertionError("cached page hash should avoid image decoding")

    monkeypatch.setattr(
        "turkicdocgen.dedup.compute_full_page_dhash",
        fail_redecode,
    )
    generate_iteration8_reports(
        tmp_path,
        [
            {
                "page_id": "cached",
                "image": "images/cached.png",
                "layout_id": "book_page_single_column",
                "effect_metadata": {"full_page_dhash_32": cached_hash},
                "zones": [{"zone_type": "body", "text": "unique content"}],
            }
        ],
        profile_name="visual_300",
    )

    report = json.loads(
        (tmp_path / "reports" / "duplicate_report.json").read_text(encoding="utf-8")
    )
    assert report["metrics"]["audited_images"] == 1


def test_iteration8_public_profile_enforces_duplicate_gates(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = Image.new("RGB", (200, 300), "white")
    image.save(image_dir / "first.png")
    image.save(image_dir / "second.png")
    rows = [
        {
            "page_id": page_id,
            "image": f"images/{page_id}.png",
            "layout_id": "book_page_single_column",
            "zones": [{"zone_type": "body", "text": "Бірдей маңызды мазмұн"}],
        }
        for page_id in ("first", "second")
    ]

    with pytest.raises(ValueError, match="Hard deduplication gate"):
        generate_iteration8_reports(tmp_path, rows, profile_name="large_100k")
    report = json.loads(
        (tmp_path / "reports" / "duplicate_report.json").read_text(encoding="utf-8")
    )
    assert report["gates_passed"] is False
    assert report["exact_full_page_duplicates"] == [["first", "second"]]
