from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from turkicdocgen.hf.release import export_hf_release
from turkicdocgen.safety import (
    ROOT,
    assert_file_in_generated_roots,
    assert_generated_path,
    is_relative_to,
    redact_sensitive,
    safe_remove_generated_path,
)

from ..jobs import (
    cancel_job,
    create_job,
    get_all_jobs,
    get_job,
    log_event_generator,
    start_job,
)
from ..models import JobConfig
from .utils import (
    _available_profiles,
    _build_manifest_index,
    _find_indexed_manifest_row,
    _list_runs,
    _manifest_index_status,
    _output_base,
    _read_manifest,
    _run_dir,
)

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


class StartJobRequest(BaseModel):
    profile: str = Field(max_length=100)
    out_dir: str = Field(max_length=500)
    count: int = Field(default=24, ge=1, le=100_000)
    seed: int = 20260513
    languages: list[str] = Field(default_factory=list, max_length=20)
    layouts: list[str] = Field(default_factory=list, max_length=100)
    effects: list[str] = Field(default_factory=list, max_length=30)

    def normalized_job_config(self) -> JobConfig:
        return JobConfig(
            profile=self.profile,
            out_dir=self.out_dir,
            count=self.count,
            seed=self.seed,
            languages=tuple(self.languages),
            layouts=tuple(self.layouts),
            effects=tuple(self.effects),
        )


@router.post("/api/jobs")
async def api_start_job(req: StartJobRequest):
    profiles = _available_profiles()
    if req.profile not in profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid profile: {req.profile}. Must be one of {profiles}",
        )
    try:
        safe_base = _output_base()
        out_path = Path(req.out_dir)
        if out_path.parts and out_path.parts[0].lower() == safe_base.name.lower():
            out_path = Path(*out_path.parts[1:])
        if out_path.is_absolute():
            resolved_out = out_path.resolve()
            if not is_relative_to(resolved_out, safe_base):
                # Force it to be relative inside safe_base
                resolved_out = safe_base / out_path.name
        else:
            for part in out_path.parts:
                if part in {".", ".."}:
                    raise ValueError("invalid path parts")
            resolved_out = safe_base / out_path

        req.out_dir = str(
            assert_generated_path(resolved_out, purpose="start job output")
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = create_job(req.normalized_job_config())
    start_job(job)
    return JSONResponse(job.to_dict(), status_code=202)


@router.get("/api/jobs")
async def api_list_jobs():
    return [j.to_dict() for j in get_all_jobs()]


@router.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/api/jobs/{job_id}/events")
async def api_job_events(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(
        log_event_generator(job),
        media_type="text/event-stream",
    )


@router.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    cancelled = cancel_job(job)
    return {"cancelled": cancelled, "status": job.status.value}


@router.get("/api/profiles")
async def api_profiles():
    return {"profiles": _available_profiles()}


@router.get("/api/runs")
async def api_runs(output_base: str = "outputs"):
    return {"runs": _list_runs(output_base)}


class DeleteRunRequest(BaseModel):
    delete_type: str  # "soft" or "hard"
    output_base: str = "outputs"


@router.post("/api/runs/{run_id}/delete")
async def delete_run(run_id: str, req: DeleteRunRequest):
    run_dir = _run_dir(req.output_base, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    if req.delete_type == "hard":
        try:
            safe_remove_generated_path(run_dir)
            return {"ok": True, "message": f"Run {run_id} fully deleted."}
        except (OSError, ValueError) as e:
            logger.error(
                "Failed to hard delete run %s: %s", run_id, redact_sensitive(e)
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to hard delete: {redact_sensitive(e)}",
            ) from e

    elif req.delete_type == "soft":
        try:
            # Delete generated artifacts while keeping manifest metadata.
            heavy_dirs = ["images", "zones", "reports"]
            deleted_dirs = []
            for sub in heavy_dirs:
                sub_path = run_dir / sub
                if sub_path.exists():
                    safe_remove_generated_path(sub_path)
                    deleted_dirs.append(sub)

            # Also clean up any loose heavy image files in the root of run_dir.
            for item in run_dir.iterdir():
                if item.is_file() and item.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    safe_remove_generated_path(item)

            # Mark the run as soft-deleted in a small marker file
            marker = run_dir / ".soft_deleted"
            marker.touch()

            return {
                "ok": True,
                "message": f"Run {run_id} soft-deleted. Cleared folders: {', '.join(deleted_dirs)}.",
            }
        except (OSError, ValueError) as e:
            logger.error(
                "Failed to soft delete run %s: %s", run_id, redact_sensitive(e)
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to soft delete: {redact_sensitive(e)}",
            ) from e
    else:
        raise HTTPException(
            status_code=400, detail="Invalid delete type. Must be 'soft' or 'hard'."
        )


@router.post("/api/runs/{run_id}/export")
async def api_export_run(run_id: str, output_base: str = "outputs"):
    run_dir = _run_dir(output_base, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Manifest is missing; export is not possible",
        )

    out_dir = assert_generated_path(ROOT / "release" / run_id, purpose="export release")

    try:
        export_hf_release(run_dir, out_dir, hf_card=True, force=True)
        report_path = out_dir / "reports" / "release_report.json"
        return {
            "ok": True,
            "message": f"Run exported to {out_dir}",
            "report_path": str(report_path),
        }
    except (OSError, ValueError, ImportError) as e:
        logger.error("Export failed for run %s: %s", run_id, redact_sensitive(e))
        raise HTTPException(
            status_code=500, detail=f"Export failed: {redact_sensitive(e)}"
        ) from e


@router.get("/api/runs/{run_id}/index/status")
async def api_run_index_status(run_id: str, output_base: str = "outputs"):
    run_dir = _run_dir(output_base, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return _manifest_index_status(run_dir)


@router.post("/api/runs/{run_id}/index")
async def api_build_run_index(
    run_id: str,
    background_tasks: BackgroundTasks,
    output_base: str = "outputs",
):
    run_dir = _run_dir(output_base, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    background_tasks.add_task(_build_manifest_index, run_dir)
    return JSONResponse(
        {"ok": True, "status": _manifest_index_status(run_dir)},
        status_code=202,
    )


@router.get("/image")
async def serve_image(path: str):
    """Serve generated images from the output directory."""
    try:
        img_path = assert_file_in_generated_roots(
            Path(path), suffixes={".png", ".jpg", ".jpeg", ".webp"}
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path))


@router.get("/thumb")
async def serve_thumbnail(path: str, w: int = 360):
    """Serve cached thumbnails for generated images."""
    try:
        img_path = assert_file_in_generated_roots(
            Path(path), suffixes={".png", ".jpg", ".jpeg", ".webp"}
        )
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    width = max(64, min(int(w or 360), 1024))
    try:
        stat = img_path.stat()
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc
    cache_key = hashlib.sha256(
        f"{img_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{width}".encode()
    ).hexdigest()
    cache_dir = assert_generated_path(ROOT / ".web_cache" / "thumbs", purpose="thumb")
    cache_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = cache_dir / f"{cache_key}.jpg"
    if not thumb_path.exists():
        try:
            with Image.open(img_path) as image:
                image.thumbnail((width, width * 4), Image.Resampling.LANCZOS)
                image.convert("RGB").save(thumb_path, format="JPEG", quality=82)
        except OSError as exc:
            raise HTTPException(status_code=422, detail="Thumbnail failed") from exc
    return FileResponse(str(thumb_path), media_type="image/jpeg")


class VisualStatusRequest(BaseModel):
    visual_qa_status: str  # "pending_manual" | "accepted" | "rejected" | "flagged"
    reviewer_note: str = Field(default="", max_length=4000)
    output_base: str = Field(default="outputs", max_length=100)
    run_id: str = Field(default="", max_length=200)


@router.patch("/api/samples/{sample_id}/visual-status")
async def update_visual_status(sample_id: str, req: VisualStatusRequest):
    """Update visual_qa_status for a sample. Writes to visual_review.jsonl in the run dir."""
    valid_statuses = {"pending_manual", "accepted", "rejected", "flagged"}
    if req.visual_qa_status not in valid_statuses:
        raise HTTPException(
            status_code=422, detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    if not req.run_id:
        raise HTTPException(status_code=422, detail="run_id is required")

    run_dir = _run_dir(req.output_base, req.run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    entry = {
        "sample_id": sample_id,
        "visual_qa_status": req.visual_qa_status,
        "reviewer_note": req.reviewer_note,
        "reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    review_log = run_dir / "visual_review.jsonl"
    with open(review_log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {
        "ok": True,
        "sample_id": sample_id,
        "visual_qa_status": req.visual_qa_status,
    }


@router.get("/api/runs/{run_id}/manifest")
async def api_run_manifest(run_id: str, output_base: str = "outputs", limit: int = 200):
    """Return manifest rows as JSON, used for zone overlay JS."""
    run_dir = _run_dir(output_base, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    rows = _read_manifest(run_dir, limit=limit)
    return {"run_id": run_id, "rows": rows, "total": len(rows)}


@router.get("/api/runs/{run_id}/samples/{sample_id}/zones")
async def api_sample_zones(run_id: str, sample_id: str, output_base: str = "outputs"):
    """Return zones for one sample without embedding all gallery zones in HTML."""
    run_dir = _run_dir(output_base, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    found = _find_indexed_manifest_row(run_dir, sample_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    row = found[0]
    zones = row.get("zone_gt") or row.get("zones") or []
    zone_gt_path = row.get("zone_gt_path")
    if zone_gt_path:
        path = (run_dir / str(zone_gt_path)).resolve()
        if is_relative_to(path, run_dir) and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                zones = payload.get("zones", zones)
            except json.JSONDecodeError:
                pass
    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "orientation": row.get("orientation"),
        "zones": zones,
    }
