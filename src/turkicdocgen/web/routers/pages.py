from __future__ import annotations

import json
import math
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from turkicdocgen.safety import is_relative_to

from .utils import (
    GalleryFilters,
    _available_profiles,
    _find_indexed_manifest_row,
    _list_runs,
    _manifest_index_status,
    _read_indexed_gallery,
    _read_rejected_page,
    _run_detail_data,
    _run_dir,
    templates,
)

router = APIRouter()


def _normalize_and_validate_zones(
    zones: list, orientation: str | None
) -> tuple[list[dict], dict[str, int]]:
    normalized_zones = []
    received_count = len(zones)
    valid_count = 0
    rejected_count = 0
    drawable_count = 0

    seen_ids = set()

    if orientation == "landscape":
        max_w, max_h = 2339, 1654
    else:
        max_w, max_h = 1654, 2339

    for zone in zones:
        is_valid = True

        # 1. Base zone structure
        z = dict(zone)

        # 2. Unique ID validation
        zone_id = z.get("zone_id")
        if not zone_id or not isinstance(zone_id, str):
            is_valid = False
            # Ensure it has a fallback ID for rendering
            zone_id = f"invalid_id_{len(normalized_zones)}"
        elif zone_id in seen_ids:
            is_valid = False
        else:
            seen_ids.add(zone_id)

        z["zone_id"] = zone_id

        # 3. Role and zone_type normalization
        role = z.get("role") or z.get("zone_type") or "default"
        zone_type = z.get("zone_type") or z.get("role") or "default"
        z["role"] = role
        z["zone_type"] = zone_type

        # 4. Reading order validation and normalization
        reading_order = z.get("reading_order")
        if reading_order is None:
            is_valid = False
            z["reading_order"] = 0
        else:
            try:
                ro_val = float(reading_order)
                if not math.isfinite(ro_val):
                    is_valid = False
                    z["reading_order"] = 0
                else:
                    z["reading_order"] = int(ro_val) if ro_val.is_integer() else ro_val
            except (ValueError, TypeError):
                is_valid = False
                z["reading_order"] = 0

        # 5. bbox validation
        bbox = z.get("bbox") or z.get("bounding_box")
        has_valid_bbox = False
        if bbox is not None:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                # Check finite
                coords_finite = True
                coords_bounded = True
                for c in bbox:
                    if not isinstance(c, (int, float)) or not math.isfinite(c):
                        coords_finite = False

                if coords_finite:
                    # Check page-bounded
                    # bbox: [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = bbox
                    if not (
                        0 <= x_min <= max_w
                        and 0 <= x_max <= max_w
                        and 0 <= y_min <= max_h
                        and 0 <= y_max <= max_h
                    ):
                        coords_bounded = False
                    # Check bbox order
                    coords_ordered = x_min < x_max and y_min < y_max
                    if not coords_ordered:
                        is_valid = False

                    if not coords_bounded:
                        is_valid = False

                    if coords_bounded and coords_ordered:
                        has_valid_bbox = True
                else:
                    is_valid = False
            else:
                is_valid = False
        else:
            # bbox is missing
            pass

        # 6. polygon validation
        polygon = z.get("polygon")
        has_valid_polygon = False
        if polygon is not None:
            if isinstance(polygon, (list, tuple)):
                if len(polygon) >= 3:
                    poly_ok = True
                    for pt in polygon:
                        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                            poly_ok = False
                            break
                        x, y = pt
                        if not isinstance(x, (int, float)) or not math.isfinite(x):
                            poly_ok = False
                            break
                        if not isinstance(y, (int, float)) or not math.isfinite(y):
                            poly_ok = False
                            break
                        if not (0 <= x <= max_w and 0 <= y <= max_h):
                            poly_ok = False
                            break
                    if poly_ok:
                        has_valid_polygon = True
                    else:
                        is_valid = False
                else:
                    is_valid = False
            else:
                is_valid = False
        else:
            # polygon missing
            pass

        # If both bbox and polygon are missing/invalid, then it's invalid
        if not has_valid_bbox and not has_valid_polygon:
            is_valid = False

        # 7. Normalize bbox and polygon (canonical schema ensures both are populated if one is valid)
        if has_valid_bbox and not has_valid_polygon:
            x_min, y_min, x_max, y_max = bbox
            z["bbox"] = [float(x_min), float(y_min), float(x_max), float(y_max)]
            z["polygon"] = [
                [float(x_min), float(y_min)],
                [float(x_max), float(y_min)],
                [float(x_max), float(y_max)],
                [float(x_min), float(y_max)],
            ]
        elif has_valid_polygon and not has_valid_bbox:
            xs = [float(pt[0]) for pt in polygon]
            ys = [float(pt[1]) for pt in polygon]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z["bbox"] = [x_min, y_min, x_max, y_max]
            z["polygon"] = [[float(pt[0]), float(pt[1])] for pt in polygon]
        elif has_valid_bbox and has_valid_polygon:
            z["bbox"] = [float(c) for c in bbox]
            z["polygon"] = [[float(pt[0]), float(pt[1])] for pt in polygon]
        else:
            # Neither is valid, set them to default empty/safe values to prevent errors in JS
            z["bbox"] = [0.0, 0.0, 0.0, 0.0]
            z["polygon"] = []

        # 8. Lines and Cells normalization
        if "lines" not in z or not isinstance(z["lines"], list):
            z["lines"] = []
        else:
            normalized_lines = []
            for line in z["lines"]:
                if isinstance(line, dict):
                    ld = dict(line)
                    if "text" not in ld:
                        ld["text"] = ""
                    if "reading_order" not in ld:
                        ld["reading_order"] = 0
                    normalized_lines.append(ld)
            z["lines"] = normalized_lines

        if "cells" not in z or not isinstance(z["cells"], list):
            z["cells"] = []
        else:
            normalized_cells = []
            for cell in z["cells"]:
                if isinstance(cell, dict):
                    cd = dict(cell)
                    if "text" not in cd:
                        cd["text"] = ""
                    normalized_cells.append(cd)
            z["cells"] = normalized_cells

        if "text" not in z:
            z["text"] = ""

        # 9. Calculate drawable status
        has_geometry = bool(z["polygon"]) or z["bbox"] != [0.0, 0.0, 0.0, 0.0]
        is_drawable = is_valid and has_geometry
        z["validation_status"] = "valid" if is_valid else "rejected"
        z["drawable"] = is_drawable

        if is_drawable:
            drawable_count += 1

        if is_valid:
            valid_count += 1
        else:
            rejected_count += 1

        normalized_zones.append(z)

    stats = {
        "received": received_count,
        "valid": valid_count,
        "rejected": rejected_count,
        "drawable": drawable_count,
    }
    return normalized_zones, stats


@router.get("/gallery-legacy", response_class=RedirectResponse)
async def legacy_gallery(run_id: str | None = None, output_base: str = "outputs"):
    if run_id:
        return RedirectResponse(
            url=f"/gallery?run_id={run_id}&output_base={output_base}",
            status_code=303,
        )
    return RedirectResponse(url=f"/gallery?output_base={output_base}", status_code=303)


@router.get("/gallery", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    filters: Annotated[GalleryFilters, Depends(GalleryFilters)],
):
    run_id = filters.run_id
    output_base = filters.output_base
    status_filter = filters.status_filter
    lang_filter = filters.lang_filter
    profile_filter = filters.profile_filter
    effect_filter = filters.effect_filter
    stamp_filter = filters.stamp_filter
    warning_filter = filters.warning_filter
    domain_filter = filters.domain_filter
    page = filters.page
    page_size = filters.page_size

    runs = _list_runs(output_base)
    if not run_id and runs:
        run_id = runs[0]["run_id"]

    if not run_id:
        return templates.TemplateResponse(
            request=request,
            name="gallery.html",
            context={
                "request": request,
                "runs": [],
                "run_id": "",
                "output_base": output_base,
                "rows": [],
                "total": 0,
                "page": 1,
                "page_size": page_size,
                "total_pages": 1,
                "status_filter": status_filter,
                "lang_filter": lang_filter,
                "profile_filter": profile_filter,
                "effect_filter": effect_filter,
                "stamp_filter": stamp_filter,
                "warning_filter": warning_filter,
                "domain_filter": domain_filter,
                "unique_languages": [],
                "unique_profiles": [],
                "unique_effects": [],
                "unique_warnings": [],
                "unique_domains": [],
                "index_status": {"state": "missing", "message": ""},
            },
        )

    run_dir = _run_dir(output_base, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    page_rows, total, filter_opts = _read_indexed_gallery(run_dir, filters)
    index_status = _manifest_index_status(run_dir)

    return templates.TemplateResponse(
        request=request,
        name="gallery.html",
        context={
            "request": request,
            "runs": runs,
            "run_id": run_id,
            "output_base": output_base,
            "rows": page_rows,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
            "status_filter": status_filter,
            "lang_filter": lang_filter,
            "profile_filter": profile_filter,
            "effect_filter": effect_filter,
            "stamp_filter": stamp_filter,
            "warning_filter": warning_filter,
            "domain_filter": domain_filter,
            "unique_languages": filter_opts["unique_languages"],
            "unique_profiles": filter_opts["unique_profiles"],
            "unique_effects": filter_opts["unique_effects"],
            "unique_warnings": filter_opts["unique_warnings"],
            "unique_domains": filter_opts["unique_domains"],
            "index_status": index_status,
        },
    )


@router.get("/", response_class=RedirectResponse)
async def root_page(run_id: str | None = None, output_base: str = "outputs"):
    if run_id:
        return RedirectResponse(
            url=f"/gallery?run_id={run_id}&output_base={output_base}",
            status_code=303,
        )
    return RedirectResponse(url=f"/gallery?output_base={output_base}", status_code=303)


@router.get("/generate", response_class=HTMLResponse)
async def generate_page(request: Request):
    profiles = _available_profiles()
    return templates.TemplateResponse(
        request=request,
        name="generate.html",
        context={"request": request, "profiles": profiles},
    )


@router.get("/runs", response_class=RedirectResponse)
async def runs_page(run_id: str | None = None, output_base: str = "outputs"):
    if run_id:
        return RedirectResponse(
            url=f"/gallery?run_id={run_id}&output_base={output_base}",
            status_code=303,
        )
    return RedirectResponse(url=f"/gallery?output_base={output_base}", status_code=303)


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str, output_base: str = "outputs"):
    run_dir = _run_dir(output_base, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    detail = _run_detail_data(run_dir)
    return templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "request": request,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "rows": detail["rows"],
            "total": detail["total"],
            "accepted_count": detail["accepted_count"],
            "rejected_count": detail["rejected_count"],
            "bounded": detail["bounded"],
            "index_status": detail["index_status"],
            "output_base": output_base,
        },
    )


@router.get("/runs/{run_id}/rejected", response_class=HTMLResponse)
async def rejected_page(
    request: Request,
    run_id: str,
    output_base: str = "outputs",
    reason_filter: str = "",
    page: int = 1,
    page_size: int = 24,
):
    run_dir = _run_dir(output_base, run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    rejected = _read_rejected_page(
        run_dir, page=page, page_size=page_size, reason_filter=reason_filter
    )

    return templates.TemplateResponse(
        request=request,
        name="rejected.html",
        context={
            "request": request,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "output_base": output_base,
            "rejected_rows": rejected["rows"],
            "total": rejected["total"],
            "reason_types": rejected["reason_types"],
            "selected_reason": reason_filter,
            "page": rejected["page"],
            "page_size": rejected["page_size"],
            "total_pages": rejected["total_pages"],
            "bounded": rejected["bounded"],
            "index_status": rejected["index_status"],
        },
    )


@router.get("/runs/{run_id}/gallery", response_class=RedirectResponse)
async def gallery_page(request: Request, run_id: str):
    params = dict(request.query_params)
    params["run_id"] = run_id
    if "output_base" not in params:
        params["output_base"] = "outputs"
    query_str = urlencode(params)
    return RedirectResponse(
        url=f"/gallery?{query_str}",
        status_code=303,
    )


@router.get("/runs/{run_id}/sample/{sample_id}", response_class=HTMLResponse)
async def sample_detail(
    request: Request,
    run_id: str,
    sample_id: str,
    output_base: str = "outputs",
):
    run_dir = _run_dir(output_base, run_id)
    found = _find_indexed_manifest_row(run_dir, sample_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
    row, prev_sample_id, next_sample_id, idx, total_samples = found
    zone_gt = row.get("zone_gt") or row.get("zones") or []
    zone_gt_path = row.get("zone_gt_path")
    if zone_gt_path:
        path = (run_dir / str(zone_gt_path)).resolve()
        if is_relative_to(path, run_dir) and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                zone_gt = payload.get("zones", zone_gt)
            except json.JSONDecodeError:
                pass

    orientation = row.get("orientation")
    normalized_zones, zone_stats = _normalize_and_validate_zones(zone_gt, orientation)

    qa_issues = row.get("qa_issues") or []
    for z in normalized_zones:
        z["qa_issues"] = [
            issue for issue in qa_issues if issue.get("zone_id") == z.get("zone_id")
        ]

    zone_gt = normalized_zones
    row["zones"] = normalized_zones
    row["zone_gt"] = normalized_zones

    return templates.TemplateResponse(
        request=request,
        name="sample_detail.html",
        context={
            "request": request,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "sample_id": sample_id,
            "row": row,
            "zone_gt": zone_gt,
            "zone_stats": zone_stats,
            "output_base": output_base,
            "prev_sample_id": prev_sample_id,
            "next_sample_id": next_sample_id,
            "current_idx": idx,
            "total_samples": total_samples,
        },
    )
