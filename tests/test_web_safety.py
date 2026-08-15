from __future__ import annotations

import json
from http import HTTPStatus
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from fastapi.testclient import TestClient

from turkicdocgen.web.app import app, get_web_token


def test_web_rejects_traversal_run_id() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/runs/../manifest", params={"output_base": "outputs"})
    assert resp.status_code in {HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND}


def test_web_rejects_arbitrary_output_base() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/runs", params={"output_base": "src"})
    assert resp.status_code == HTTPStatus.FORBIDDEN


def test_web_image_route_only_serves_generated_media(tmp_path: Path) -> None:
    source_file = tmp_path / "secret.jpg"
    source_file.write_bytes(b"not really an image")
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/image", params={"path": str(source_file)})
    assert resp.status_code == HTTPStatus.FORBIDDEN


def test_remote_auth_requires_bearer_header() -> None:
    web_token = get_web_token()
    with TestClient(
        app,
        raise_server_exceptions=False,
        client=("203.0.113.10", 50000),
    ) as client:
        query_response = client.get(
            "/",
            params={"token": web_token},
            follow_redirects=False,
        )
        bearer_response = client.get(
            "/",
            headers={"Authorization": f"Bearer {web_token}"},
            follow_redirects=False,
        )

    assert query_response.status_code == HTTPStatus.UNAUTHORIZED
    assert bearer_response.status_code in {
        HTTPStatus.OK,
        HTTPStatus.SEE_OTHER,
        HTTPStatus.TEMPORARY_REDIRECT,
    }


def test_loopback_with_proxy_headers_requires_auth() -> None:
    with TestClient(
        app,
        raise_server_exceptions=False,
        client=("127.0.0.1", 50000),
    ) as client:
        normal_resp = client.get("/", follow_redirects=False)
        assert normal_resp.status_code in {HTTPStatus.OK, HTTPStatus.SEE_OTHER}

        proxy_resp = client.get(
            "/",
            headers={"X-Forwarded-For": "203.0.113.10"},
            follow_redirects=False,
        )
        assert proxy_resp.status_code == HTTPStatus.UNAUTHORIZED


def test_sample_detail_blocks_zone_path_escape_and_escapes_html(
    tmp_path: Path, monkeypatch
) -> None:
    import turkicdocgen.safety as safety
    import turkicdocgen.web.routers.utils as web_utils

    monkeypatch.setattr(safety, "ROOT", tmp_path)
    monkeypatch.setattr(web_utils, "ROOT", tmp_path)
    output_base = tmp_path / "outputs"
    run_dir = output_base / "security-run"
    run_dir.mkdir(parents=True)
    secret = tmp_path / "secret.json"
    secret.write_text(
        json.dumps({"zones": [{"text": "EXTERNAL_SECRET"}]}), encoding="utf-8"
    )
    row = {
        "id": "sample-1",
        "page_id": "sample-1",
        "qa_ok": True,
        "zone_gt_path": "../../secret.json",
        "zones": [
            {
                "zone_id": "body",
                "zone_type": "body",
                "reading_order": 1,
                "bbox": [10, 10, 100, 100],
                "text": "<script>alert('xss')</script>",
            }
        ],
    }
    (run_dir / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with TestClient(app, raise_server_exceptions=True) as client:
        response = client.get(
            "/runs/security-run/sample/sample-1",
            params={"output_base": "outputs"},
        )
    assert response.status_code == HTTPStatus.OK
    assert "EXTERNAL_SECRET" not in response.text
    assert "<script>alert('xss')</script>" not in response.text


def test_web_rejects_oversized_job_and_review_payloads() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        job_response = client.post(
            "/api/jobs",
            json={
                "profile": "visual_300",
                "out_dir": "oversized",
                "count": 250_001,
            },
        )
        review_response = client.patch(
            "/api/samples/sample/visual-status",
            json={
                "visual_qa_status": "flagged",
                "reviewer_note": "x" * 4001,
                "run_id": "run",
            },
        )
    assert job_response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert review_response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
