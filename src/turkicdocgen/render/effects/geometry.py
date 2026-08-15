from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .common import _range

if TYPE_CHECKING:
    import random

POINT_DIMENSION = 2


def _solve_homography(
    source: list[tuple[float, float]],
    destination: list[tuple[float, float]],
) -> np.ndarray:
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)
    coefficients = np.linalg.solve(np.asarray(rows), np.asarray(values))
    return np.asarray(
        [
            [coefficients[0], coefficients[1], coefficients[2]],
            [coefficients[3], coefficients[4], coefficients[5]],
            [coefficients[6], coefficients[7], 1.0],
        ],
        dtype=float,
    )


def _transform_point(
    point: tuple[int, int] | tuple[float, float],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int]:
    vector = matrix @ np.asarray([point[0], point[1], 1.0])
    x = vector[0] / vector[2]
    y = vector[1] / vector[2]
    return (
        round(max(0, min(width - 1, x))),
        round(max(0, min(height - 1, y))),
    )


def _bbox_from_points(
    points: list[tuple[int, int]], width: int, height: int
) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        max(0, min(xs)),
        max(0, min(ys)),
        min(width, max(xs) + 1),
        min(height, max(ys) + 1),
    )


def _bbox_points(bbox: list[int] | tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _transform_metadata_bbox(
    metadata: dict[str, Any],
    key: str,
    matrix: np.ndarray,
    width: int,
    height: int,
) -> None:
    bbox = metadata.get(key)
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return
    polygon = [
        _transform_point(point, matrix, width, height) for point in _bbox_points(bbox)
    ]
    metadata[f"{key}_polygon"] = polygon
    metadata[key] = list(_bbox_from_points(polygon, width, height))


def _transform_annotations(
    plan: Any, matrix: np.ndarray, width: int, height: int
) -> None:
    for zone in getattr(plan, "zones", []):
        original_zone_bbox = zone.bbox
        _transform_metadata_bbox(zone.metadata, "rendered_bbox", matrix, width, height)
        rendered_fields = zone.metadata.get("rendered_fields")
        if isinstance(rendered_fields, list):
            for field in rendered_fields:
                if not isinstance(field, dict):
                    continue
                for key in (
                    "label_bbox",
                    "value_bbox",
                    "row_bbox",
                    "rendered_bbox",
                ):
                    _transform_metadata_bbox(field, key, matrix, width, height)
        label_width = zone.metadata.get("form_label_width")
        if isinstance(label_width, int | float):
            separator_x = original_zone_bbox[0] + label_width
            zone.metadata["form_separator_polygon"] = [
                _transform_point(
                    (separator_x, original_zone_bbox[1]), matrix, width, height
                ),
                _transform_point(
                    (separator_x, original_zone_bbox[3]), matrix, width, height
                ),
            ]
        zone.polygon = [
            _transform_point(point, matrix, width, height) for point in zone.polygon
        ]
        zone.bbox = _bbox_from_points(zone.polygon, width, height)
        for line in zone.lines:
            points = line.polygon or [
                (line.bbox[0], line.bbox[1]),
                (line.bbox[2], line.bbox[1]),
                (line.bbox[2], line.bbox[3]),
                (line.bbox[0], line.bbox[3]),
            ]
            line.polygon = [
                _transform_point(point, matrix, width, height) for point in points
            ]
            line.bbox = _bbox_from_points(line.polygon, width, height)
        for cell in zone.cells:
            _transform_metadata_bbox(
                cell.metadata, "rendered_bbox", matrix, width, height
            )
            points = cell.polygon or [
                (cell.bbox[0], cell.bbox[1]),
                (cell.bbox[2], cell.bbox[1]),
                (cell.bbox[2], cell.bbox[3]),
                (cell.bbox[0], cell.bbox[3]),
            ]
            cell.polygon = [
                _transform_point(point, matrix, width, height) for point in points
            ]
            cell.bbox = _bbox_from_points(cell.polygon, width, height)


def _apply_perspective(
    image: Image.Image,
    plan: Any | None,
    rng: random.Random,
    ratio_range: list[float | int],
) -> tuple[Image.Image, dict[str, object]]:
    width, height = image.size
    ratio = _range(rng, ratio_range)
    dx = width * ratio
    dy = height * ratio
    source = [
        (0.0, 0.0),
        (width - 1.0, 0.0),
        (width - 1.0, height - 1.0),
        (0.0, height - 1.0),
    ]
    destination = [
        (rng.uniform(0, dx), rng.uniform(0, dy)),
        (width - 1 - rng.uniform(0, dx), rng.uniform(0, dy)),
        (width - 1 - rng.uniform(0, dx), height - 1 - rng.uniform(0, dy)),
        (rng.uniform(0, dx), height - 1 - rng.uniform(0, dy)),
    ]
    forward = _solve_homography(source, destination)
    inverse = np.linalg.inv(forward)
    pillow_inverse = inverse / inverse[2, 2]
    coefficients = tuple(float(value) for value in pillow_inverse.flatten()[:8])
    transformed = image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(242, 241, 237),
    )
    if plan is not None:
        _transform_annotations(plan, forward, width, height)
    return transformed, {
        "corner_offset_ratio": ratio,
        "source_corners": source,
        "destination_corners": destination,
        "forward_matrix": forward.tolist(),
        "inverse_matrix": inverse.tolist(),
    }


def _phone_geometry_tier(seed_digest: str) -> str:
    bucket = int(seed_digest[16:24], 16) % 100
    mild_threshold = 55
    moderate_threshold = 80
    if bucket < mild_threshold:
        return "mild"
    if bucket < moderate_threshold:
        return "moderate"
    return "extreme"


def _apply_phone_geometry(
    image: Image.Image,
    plan: Any | None,
    rng: random.Random,
    params: dict[str, Any],
    tier: str,
) -> tuple[Image.Image, dict[str, object]]:
    width, height = image.size
    rotation_range = params[f"phone_photo_{tier}_rotation_degrees"]
    offset_range = params[f"phone_photo_{tier}_corner_offset_ratio"]
    if tier == "extreme":
        angle = rng.choice((-1.0, 1.0)) * rng.uniform(
            9.0, max(abs(float(v)) for v in rotation_range)
        )
    elif tier == "moderate":
        angle = rng.choice((-1.0, 1.0)) * rng.uniform(
            4.0, max(abs(float(v)) for v in rotation_range)
        )
    else:
        angle = _range(rng, rotation_range)
    ratio = _range(rng, offset_range)
    source = np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=float,
    )
    center = np.asarray([width / 2, height / 2], dtype=float)
    theta = math.radians(-angle)
    rotation = np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    destination = (source - center) @ rotation.T + center
    dx, dy = width * ratio, height * ratio
    keystone = np.asarray(
        [
            [rng.uniform(-0.25, 0.45) * dx, rng.uniform(-0.15, 0.55) * dy],
            [rng.uniform(-0.45, 0.25) * dx, rng.uniform(-0.15, 0.55) * dy],
            [rng.uniform(-0.45, 0.25) * dx, rng.uniform(-0.55, 0.15) * dy],
            [rng.uniform(-0.25, 0.45) * dx, rng.uniform(-0.55, 0.15) * dy],
        ]
    )
    destination += keystone

    margin_ratio = 0.025 if tier != "extreme" else 0.035
    margin_x, margin_y = width * margin_ratio, height * margin_ratio
    span = destination.max(axis=0) - destination.min(axis=0)
    fit_scale = min(
        (width - 2 * margin_x) / max(1.0, span[0]),
        (height - 2 * margin_y) / max(1.0, span[1]),
        1.0,
    )
    destination = (destination - destination.mean(axis=0)) * fit_scale + center
    forward = _solve_homography(
        [tuple(point) for point in source],
        [tuple(point) for point in destination],
    )
    inverse = np.linalg.inv(forward)
    pillow_inverse = inverse / inverse[2, 2]
    coefficients = tuple(float(value) for value in pillow_inverse.flatten()[:8])
    fill = tuple(
        getattr(plan, "metadata", {}).get("paper_base", {}).get("rgb", [247, 247, 245])
    )
    transformed = image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )
    if plan is not None:
        _transform_annotations(plan, forward, width, height)
    destination_list = [tuple(float(value) for value in point) for point in destination]
    return transformed, {
        "geometry_tier": tier,
        "angle_degrees": angle,
        "corner_offset_ratio": ratio,
        "fit_scale": fit_scale,
        "source_corners": [tuple(point) for point in source.tolist()],
        "destination_corners": destination_list,
        "forward_matrix": forward.tolist(),
        "inverse_matrix": inverse.tolist(),
        "page_fully_visible": True,
    }


def _rotate_point(
    x: int, y: int, width: int, height: int, angle_degrees: float
) -> tuple[int, int]:
    angle = math.radians(-angle_degrees)
    cx = width / 2
    cy = height / 2
    dx = x - cx
    dy = y - cy
    nx = cx + dx * math.cos(angle) - dy * math.sin(angle)
    ny = cy + dx * math.sin(angle) + dy * math.cos(angle)
    return (
        round(max(0, min(width - 1, nx))),
        round(max(0, min(height - 1, ny))),
    )


def _rotate_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int, angle_degrees: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    points = [
        _rotate_point(x1, y1, width, height, angle_degrees),
        _rotate_point(x2, y1, width, height, angle_degrees),
        _rotate_point(x2, y2, width, height, angle_degrees),
        _rotate_point(x1, y2, width, height, angle_degrees),
    ]
    return _bbox_from_points(points, width, height)


def _rotate_plan_annotations(
    plan: Any, width: int, height: int, angle_degrees: float
) -> None:
    for zone in getattr(plan, "zones", []):
        zone.polygon = [
            _rotate_point(x, y, width, height, angle_degrees) for x, y in zone.polygon
        ]
        zone.bbox = _bbox_from_points(zone.polygon, width, height)
        for line in zone.lines:
            points = line.polygon or [
                (line.bbox[0], line.bbox[1]),
                (line.bbox[2], line.bbox[1]),
                (line.bbox[2], line.bbox[3]),
                (line.bbox[0], line.bbox[3]),
            ]
            line.polygon = [
                _rotate_point(x, y, width, height, angle_degrees) for x, y in points
            ]
            line.bbox = _bbox_from_points(line.polygon, width, height)
        for cell in zone.cells:
            points = cell.polygon or [
                (cell.bbox[0], cell.bbox[1]),
                (cell.bbox[2], cell.bbox[1]),
                (cell.bbox[2], cell.bbox[3]),
                (cell.bbox[0], cell.bbox[3]),
            ]
            cell.polygon = [
                _rotate_point(x, y, width, height, angle_degrees) for x, y in points
            ]
            cell.bbox = _bbox_from_points(cell.polygon, width, height)


def _transform_artifacts(
    artifacts: list[dict[str, object]],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> None:
    for artifact in artifacts:
        points = artifact.get("polygon")
        if not isinstance(points, list) or not points:
            continue
        transformed = [
            _transform_point((float(point[0]), float(point[1])), matrix, width, height)
            for point in points
            if isinstance(point, list | tuple) and len(point) == POINT_DIMENSION
        ]
        if transformed:
            artifact["polygon"] = transformed
            artifact["bbox"] = list(_bbox_from_points(transformed, width, height))
