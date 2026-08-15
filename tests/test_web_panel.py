from __future__ import annotations

import json
import shutil
from http import HTTPStatus
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from turkicdocgen.web.app import app
from turkicdocgen.web.routers.utils import (
    _build_manifest_index,
    _has_stamp,
    _manifest_index_path,
    _normalize_manifest_row,
)


def _write_manifest(run_dir: Path, count: int, *, image: str = "") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(count):
            row = {
                "id": f"sample_{index:06d}",
                "page_id": f"sample_{index:06d}",
                "qa_ok": index % 5 != 0,
                "language_mix": "kk",
                "layout_id": "book_page_single_column",
                "effect_profile": "clean",
                "quality_profile": "visual_check",
                "orientation": "portrait",
                "image": image,
                "qa_issues": [{"code": "quality_gate"}] if index % 5 == 0 else [],
                "zones": [
                    {
                        "zone_id": "body",
                        "zone_type": "body",
                        "reading_order": 1,
                        "bbox": [10, 10, 100, 100],
                        "text": f"heavy manifest text payload {index}",
                    }
                ],
            }
            handle.write(json.dumps(row) + "\n")


def test_web_profiles_are_dataset_only() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.get("/api/profiles")
        assert resp.status_code == HTTPStatus.OK
        profiles = resp.json()["profiles"]
        assert "visual_300" in profiles
        assert "quality_gate" not in profiles


def test_web_job_creation_defaults_to_new_profile() -> None:
    root = Path(__file__).resolve().parent.parent
    test_out = root / "runs" / "test_web_run_temp"
    if test_out.exists():
        shutil.rmtree(test_out, ignore_errors=True)

    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.post(
            "/api/jobs",
            json={
                "profile": "visual_300",
                "out_dir": str(test_out),
                "count": 1,
                "seed": 42,
                "workers": 1,
            },
        )
        try:
            assert resp.status_code == HTTPStatus.ACCEPTED
            payload = resp.json()
            assert payload["profile"] == "visual_300"
            assert payload["status"] in {"pending", "running", "done"}
        finally:
            if test_out.exists():
                shutil.rmtree(test_out, ignore_errors=True)


def test_gallery_accepts_effect_filter() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        resp = client.get("/", params={"effect_filter": "clean"})
        assert resp.status_code == HTTPStatus.OK


def test_gallery_pagination_and_filters_preserve_selected_run() -> None:
    root = Path(__file__).resolve().parent.parent
    output_root = root / "outputs"
    run_id = "test_gallery_selected_run"
    run_dir = output_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    rows = [
        {
            "id": f"sample_{index:03d}",
            "qa_ok": True,
            "language_mix": "kk",
            "layout_id": "book_page_single_column",
            "effect_profile": "clean",
        }
        for index in range(30)
    ]
    (run_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                "/gallery",
                params={
                    "run_id": run_id,
                    "output_base": "outputs",
                    "status_filter": "accepted",
                },
            )
            assert resp.status_code == HTTPStatus.OK
            assert f'name="run_id" value="{run_id}"' in resp.text
            assert f"page=2&run_id={run_id}" in resp.text

            next_page = client.get(
                "/gallery",
                params={
                    "page": 2,
                    "run_id": run_id,
                    "output_base": "outputs",
                    "status_filter": "accepted",
                },
            )
            assert next_page.status_code == HTTPStatus.OK
            assert f'<span class="mono">{run_id}</span>' in next_page.text
            assert "sample_024" in next_page.text
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_gallery_first_page_does_not_build_full_index_synchronously() -> None:
    root = Path(__file__).resolve().parent.parent
    run_id = "test_gallery_no_sync_index"
    run_dir = root / "outputs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    _write_manifest(run_dir, 2_500)

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                "/gallery",
                params={"run_id": run_id, "output_base": "outputs"},
            )
        assert resp.status_code == HTTPStatus.OK
        assert "Manifest index missing" in resp.text
        assert "sample_000001" in resp.text
        assert not _manifest_index_path(run_dir).exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_gallery_page_size_is_clamped() -> None:
    root = Path(__file__).resolve().parent.parent
    run_id = "test_gallery_page_size_clamp"
    run_dir = root / "outputs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    _write_manifest(run_dir, 140)

    try:
        _build_manifest_index(run_dir)
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                "/gallery",
                params={
                    "run_id": run_id,
                    "output_base": "outputs",
                    "status_filter": "all",
                    "page_size": 250,
                },
            )
        assert resp.status_code == HTTPStatus.OK
        assert "sample_000095" in resp.text
        assert "sample_000096" not in resp.text
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_100k_manifest_opens_first_gallery_page_without_index() -> None:
    root = Path(__file__).resolve().parent.parent
    run_id = "test_gallery_100k_first_page"
    run_dir = root / "outputs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    _write_manifest(run_dir, 100_000)

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                "/gallery",
                params={"run_id": run_id, "output_base": "outputs"},
            )
        assert resp.status_code == HTTPStatus.OK
        assert "sample_000001" in resp.text
        assert "heavy manifest text payload" not in resp.text
        assert not _manifest_index_path(run_dir).exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_indexed_sample_lookup_reaches_after_100k() -> None:
    root = Path(__file__).resolve().parent.parent
    run_id = "test_gallery_after_100k_index"
    run_dir = root / "outputs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    _write_manifest(run_dir, 100_005)

    try:
        assert _build_manifest_index(run_dir) is not None
        with TestClient(app, raise_server_exceptions=True) as client:
            sample = client.get(
                f"/runs/{run_id}/sample/sample_100004",
                params={"output_base": "outputs"},
            )
            zones = client.get(
                f"/api/runs/{run_id}/samples/sample_100004/zones",
                params={"output_base": "outputs"},
            )
        assert sample.status_code == HTTPStatus.OK
        assert "sample_100004" in sample.text
        assert zones.status_code == HTTPStatus.OK
        assert zones.json()["zones"][0]["text"] == "heavy manifest text payload 100004"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_thumbnail_endpoint_returns_image_and_reuses_cache() -> None:
    root = Path(__file__).resolve().parent.parent
    run_id = "test_gallery_thumb_cache"
    run_dir = root / "outputs" / run_id
    image_dir = run_dir / "images"
    shutil.rmtree(run_dir, ignore_errors=True)
    image_dir.mkdir(parents=True)
    image_path = image_dir / "page.jpg"
    Image.new("RGB", (120, 180), "white").save(image_path)

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            first = client.get(
                "/thumb",
                params={"path": f"outputs/{run_id}/images/page.jpg", "w": 80},
            )
            second = client.get(
                "/thumb",
                params={"path": f"outputs/{run_id}/images/page.jpg", "w": 80},
            )
        assert first.status_code == HTTPStatus.OK
        assert first.headers["content-type"].startswith("image/jpeg")
        assert second.status_code == HTTPStatus.OK
        assert first.content == second.content
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_manifest_index_api_status_and_build() -> None:
    root = Path(__file__).resolve().parent.parent
    run_id = "test_index_api_build"
    run_dir = root / "outputs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    _write_manifest(run_dir, 32)

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            missing = client.get(
                f"/api/runs/{run_id}/index/status",
                params={"output_base": "outputs"},
            )
            build = client.post(
                f"/api/runs/{run_id}/index",
                params={"output_base": "outputs"},
            )
            ready = client.get(
                f"/api/runs/{run_id}/index/status",
                params={"output_base": "outputs"},
            )
        assert missing.status_code == HTTPStatus.OK
        assert missing.json()["state"] == "missing"
        assert build.status_code == HTTPStatus.ACCEPTED
        assert ready.status_code == HTTPStatus.OK
        assert ready.json()["ready"] is True
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_rejected_page_is_paginated_with_index() -> None:
    root = Path(__file__).resolve().parent.parent
    run_id = "test_rejected_paginated"
    run_dir = root / "outputs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    _write_manifest(run_dir, 140)

    try:
        _build_manifest_index(run_dir)
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(
                f"/runs/{run_id}/rejected",
                params={"output_base": "outputs", "page": 2, "page_size": 12},
            )
        assert resp.status_code == HTTPStatus.OK
        assert "2 /" in resp.text
        assert "sample_000060" in resp.text
        assert "sample_000000" not in resp.text
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_web_manifest_index_reaches_samples_after_first_10k() -> None:
    root = Path(__file__).resolve().parent.parent
    output_root = root / "outputs"
    run_id = "test_gallery_large_manifest_index"
    run_dir = output_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    rows = []
    for index in range(10_005):
        rows.append(
            {
                "id": f"sample_{index:05d}",
                "page_id": f"sample_{index:05d}",
                "qa_ok": True,
                "language_mix": "kk",
                "layout_id": "book_page_single_column",
                "effect_profile": "clean",
                "zones": [
                    {
                        "zone_id": "body",
                        "zone_type": "body",
                        "bbox": [10, 10, 100, 100],
                        "text": f"late zone payload {index}",
                    }
                ],
            }
        )
    (run_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            sample = client.get(
                f"/runs/{run_id}/sample/sample_10004",
                params={"output_base": "outputs"},
            )
            assert sample.status_code == HTTPStatus.OK
            assert "sample_10004" in sample.text

            zones = client.get(
                f"/api/runs/{run_id}/samples/sample_10004/zones",
                params={"output_base": "outputs"},
            )
            assert zones.status_code == HTTPStatus.OK
            assert zones.json()["zones"][0]["text"] == "late zone payload 10004"

            gallery = client.get(
                "/gallery",
                params={"run_id": run_id, "output_base": "outputs"},
            )
            assert gallery.status_code == HTTPStatus.OK
            assert "data-zones=" not in gallery.text
            assert "late zone payload" not in gallery.text
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_gallery_streaming_fallback_is_not_capped_at_10k(
    tmp_path: Path, monkeypatch
) -> None:
    from turkicdocgen.web.routers import utils

    run_dir = tmp_path / "fallback_run"
    run_dir.mkdir()
    with (run_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(10_005):
            row = {
                "id": f"sample_{index:05d}",
                "qa_ok": True,
                "language_mix": "kk",
                "layout_id": "book_page_single_column",
                "effect_profile": "clean",
            }
            handle.write(json.dumps(row) + "\n")

    monkeypatch.setattr(utils, "_ensure_manifest_index", lambda _run_dir: None)
    filters = utils.GalleryFilters(page=1, page_size=24)

    rows, total, _options = utils._read_indexed_gallery(run_dir, filters)

    assert total <= utils.STREAM_FALLBACK_SCAN_LIMIT
    assert rows[0]["id"] == "sample_00000"


def test_web_ui_defaults_to_russian_with_english_secondary_mode() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        ru_resp = client.get("/")
        assert ru_resp.status_code == HTTPStatus.OK
        assert "Галерея" in ru_resp.text
        en_resp = client.get("/", params={"lang": "en"})
        assert en_resp.status_code == HTTPStatus.OK
        assert "Gallery" in en_resp.text


def test_root_redirects_to_gallery_and_gallery_is_operational() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        root = client.get("/", follow_redirects=False)
        assert root.status_code == HTTPStatus.SEE_OTHER
        assert root.headers["location"].startswith("/gallery")
        gallery = client.get("/gallery")
        assert gallery.status_code == HTTPStatus.OK
        assert "<span>TurkicDocGen</span>" in gallery.text
        assert 'class="brand-logo" src="/static/logo.png"' in gallery.text
        assert (
            gallery.text.index('class="brand"')
            < gallery.text.index('class="primary-nav"')
            < gallery.text.index("data-mobile-nav")
        )
        assert 'rel="icon" type="image/png" href="/static/logo.png"' in gallery.text
        assert 'id="delete-modal" class="modal-overlay hidden"' in gallery.text
        panel_css = client.get("/static/panel.css")
        assert panel_css.status_code == HTTPStatus.OK
        assert (
            "min-width: 220px"
            not in panel_css.text.split(".brand", 1)[1].split("}", 1)[0]
        )
        assert ".primary-nav" in panel_css.text


def test_stamp_detection_ignores_placeholder_metadata() -> None:
    legacy = _normalize_manifest_row({"id": "old", "effect_metadata": {}})
    placeholder = _normalize_manifest_row(
        {
            "id": "new",
            "effect_metadata": {
                "stamp_metadata": {"stamp_id": None, "stamp_text": None}
            },
        }
    )
    stamped = _normalize_manifest_row(
        {
            "id": "stamped",
            "effect_metadata": {
                "stamp_metadata": {"stamp_id": "ky_01", "stamp_text": "КАБЫЛ АЛЫНДЫ"}
            },
        }
    )
    assert not _has_stamp(legacy)
    assert not _has_stamp(placeholder)
    assert _has_stamp(stamped)


def test_zone_inspector_supports_selection_decorations_and_image_dimensions() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        panel_js = client.get("/static/panel.js")
        assert panel_js.status_code == HTTPStatus.OK
        assert 'selected: "#2563eb"' in panel_js.text
        assert "colorWithOpacity" in panel_js.text
        assert (
            "if (!layerDecorations && isDecorationZone(zone)) return;" in panel_js.text
        )
        assert (
            'Number.parseFloat(img.getAttribute("width")) || img.naturalWidth'
            in panel_js.text
        )
        assert (
            'Number.parseFloat(img.getAttribute("height")) || img.naturalHeight'
            in panel_js.text
        )
        assert "rect.width / coordinateWidth" in panel_js.text
        assert "rect.height / coordinateHeight" in panel_js.text
        assert "if (zone.drawable === false) return;" in panel_js.text

        panel_css = client.get("/static/panel.css")
        assert panel_css.status_code == HTTPStatus.OK
        assert (
            "#zone-overlay-canvas { z-index: 1; pointer-events: auto;" in panel_css.text
        )


def test_zone_normalization_and_validation() -> None:
    from turkicdocgen.web.routers.pages import _normalize_and_validate_zones

    # Test case 1: Valid zone portrait
    zones = [
        {
            "zone_id": "zone_1",
            "bbox": [10.0, 20.0, 100.0, 200.0],
            "reading_order": 1,
            "role": "title",
        }
    ]
    norm, stats = _normalize_and_validate_zones(zones, "portrait")
    assert stats["received"] == 1
    assert stats["valid"] == 1
    assert stats["rejected"] == 0
    assert stats["drawable"] == 1
    assert norm[0]["drawable"] is True
    assert norm[0]["validation_status"] == "valid"
    assert norm[0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert len(norm[0]["polygon"]) == 4

    # Test case 2: Invalid coordinates (out of portrait bounds: 1654x2339)
    zones = [
        {
            "zone_id": "zone_1",
            "bbox": [10.0, 20.0, 2000.0, 200.0],  # 2000.0 > 1654
            "reading_order": 1,
        }
    ]
    norm, stats = _normalize_and_validate_zones(zones, "portrait")
    assert stats["valid"] == 0
    assert stats["rejected"] == 1
    assert stats["drawable"] == 0
    assert norm[0]["drawable"] is False

    # Test case 3: Bbox order incorrect
    zones = [
        {
            "zone_id": "zone_1",
            "bbox": [100.0, 20.0, 10.0, 200.0],  # x_min > x_max
            "reading_order": 1,
        }
    ]
    norm, stats = _normalize_and_validate_zones(zones, "portrait")
    assert stats["valid"] == 0
    assert stats["rejected"] == 1
    assert stats["drawable"] == 0
    assert norm[0]["bbox"] == [0.0, 0.0, 0.0, 0.0]

    # Test case 4: Non-finite coordinates
    zones = [
        {
            "zone_id": "zone_1",
            "bbox": [10.0, 20.0, float("nan"), 200.0],
            "reading_order": 1,
        }
    ]
    norm, stats = _normalize_and_validate_zones(zones, "portrait")
    assert stats["valid"] == 0
    assert stats["rejected"] == 1
    assert stats["drawable"] == 0

    # Test case 5: Duplicate zone_id
    zones = [
        {
            "zone_id": "zone_1",
            "bbox": [10.0, 20.0, 100.0, 200.0],
            "reading_order": 1,
        },
        {
            "zone_id": "zone_1",
            "bbox": [20.0, 30.0, 120.0, 220.0],
            "reading_order": 2,
        },
    ]
    norm, stats = _normalize_and_validate_zones(zones, "portrait")
    assert stats["received"] == 2
    assert stats["valid"] == 1
    assert stats["rejected"] == 1
    assert stats["drawable"] == 1
    assert norm[0]["drawable"] is True
    assert norm[1]["drawable"] is False
