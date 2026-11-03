"""Complete the weak near-contact part of an incoming-rope mask.

SAM 2 is used for the visually distinctive free span.  The thin part next to
the wound surface is recovered by robustly fitting the free-span centre line
and intersecting it with the semantic top boundary of the wound surface.  The
module is deliberately stateless so it also works on a stopped machine.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CompletionResult:
    valid: bool
    reason: str = ""
    tangent_x: float = math.nan
    tangent_y: float = math.nan
    slope: float = math.nan
    intercept: float = math.nan
    rope_width_px: float = math.nan
    line_residual_px: float = math.nan
    added_pixels: int = 0


def _largest_right_component(mask: np.ndarray) -> np.ndarray:
    source = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(source, 8)
    if count <= 1:
        return source.astype(bool)
    best = None
    best_score = -math.inf
    width = source.shape[1]
    for label in range(1, count):
        x, _y, w, _h, area = stats[label]
        right = x + w - 1
        # Prefer the long component reaching the canonical right-hand source.
        score = area * (0.35 + 0.65 * right / max(1, width - 1))
        if score > best_score:
            best_score = score
            best = label
    return labels == best


def _fit_centre_line(mask: np.ndarray):
    component = _largest_right_component(mask)
    ys, xs = np.where(component)
    if len(xs) < 120 or np.ptp(xs) < 120:
        return None

    # A centre sample per image column prevents thick/oversegmented portions
    # from dominating the robust fit.
    columns = np.unique(xs)
    cx, cy = [], []
    for x in columns:
        column_y = ys[xs == x]
        if len(column_y):
            cx.append(float(x))
            cy.append(float(np.median(column_y)))
    points = np.column_stack([cx, cy]).astype(np.float32)
    if len(points) < 80:
        return None
    vx, vy, x0, y0 = (float(value) for value in cv2.fitLine(
        points, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    if abs(vx) < 1e-5:
        return None
    slope = vy / vx
    intercept = y0 - slope * x0
    residuals = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
    residual = float(np.median(residuals))
    length = max(1.0, float(np.ptp(xs)) * math.sqrt(1.0 + slope * slope))
    rope_width = float(np.clip(component.sum() / length, 5.0, 80.0))
    return component, slope, intercept, residual, rope_width


def _top_boundary(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    boundary = np.full(width, np.nan, dtype=np.float32)
    for x in range(width):
        rows = np.flatnonzero(mask[:, x])
        if len(rows):
            # Ignore isolated one-pixel hairs from the segmentation boundary.
            boundary[x] = float(np.percentile(rows, 1.0))
    valid = np.isfinite(boundary)
    if valid.sum() < 100:
        return boundary
    x = np.arange(width)
    boundary[~valid] = np.interp(x[~valid], x[valid], boundary[valid])
    # About 2.5% of a 4096-wide frame: suppress local wound-rope texture while
    # preserving the slowly varying cylindrical silhouette.
    kernel = max(21, round(width * 0.025) | 1)
    # OpenCV 5 restricts large-kernel medianBlur to 8-bit images.  A broad
    # Gaussian on the interpolated float contour is sufficient here because
    # we need the cylindrical envelope, not individual wire grooves.
    return cv2.GaussianBlur(boundary.reshape(1, -1), (kernel, 1), 0).reshape(-1)


def complete_rope_span(incoming_mask: np.ndarray, wound_mask: np.ndarray,
                       tangent_hint: tuple[float, float] | None = None):
    fit = _fit_centre_line(incoming_mask)
    if fit is None:
        return incoming_mask.astype(bool), CompletionResult(False, "rope_line_unavailable")
    component, slope, intercept, residual, rope_width = fit
    if residual > max(18.0, 0.8 * rope_width):
        return incoming_mask.astype(bool), CompletionResult(
            False, "rope_line_residual_high", slope=slope, intercept=intercept,
            rope_width_px=rope_width, line_residual_px=residual)

    height, width = incoming_mask.shape
    if tangent_hint is not None:
        tx, ty = map(float, tangent_hint)
        if not math.isfinite(ty):
            ty = float(slope * tx + intercept)
        reason = "manual_tangent"
    else:
        boundary = _top_boundary(wound_mask.astype(bool))
        xs = np.arange(width, dtype=np.float32)
        line_y = slope * xs + intercept
        valid = np.isfinite(boundary)
        # Limit the search to the horizontal interval supported jointly by the
        # wound surface and the extrapolated free span.
        component_x = np.flatnonzero(component.any(axis=0))
        wound_x = np.flatnonzero(wound_mask.any(axis=0))
        if not len(component_x) or not len(wound_x):
            return incoming_mask.astype(bool), CompletionResult(False, "surface_boundary_unavailable")
        search_left = max(int(wound_x.min()), int(component_x.min() - width * 0.18))
        search_right = min(int(wound_x.max()), int(component_x.max()))
        valid[:search_left] = False
        valid[search_right + 1:] = False
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            return incoming_mask.astype(bool), CompletionResult(False, "surface_intersection_unavailable")
        difference = line_y[candidates] - boundary[candidates]
        crossings = np.flatnonzero(difference[:-1] * difference[1:] <= 0)
        if len(crossings):
            # Canonical rope comes from the right; the rightmost sign change is
            # the first entry into the wound surface.
            index = int(crossings[-1])
            tx = float(candidates[index])
        else:
            index = int(np.argmin(np.abs(difference)))
            tx = float(candidates[index])
            if abs(float(difference[index])) > max(45.0, 2.0 * rope_width):
                return incoming_mask.astype(bool), CompletionResult(
                    False, "surface_intersection_residual_high", slope=slope,
                    intercept=intercept, rope_width_px=rope_width,
                    line_residual_px=residual)
        ty = float(slope * tx + intercept)
        reason = "semantic_surface_intersection"

    if not (0 <= tx < width and 0 <= ty < height):
        return incoming_mask.astype(bool), CompletionResult(False, "tangent_out_of_frame")

    xs = np.flatnonzero(component.any(axis=0))
    join_x = float(np.percentile(xs, 18.0))
    # The join must be on the visible free span, to the canonical right of the
    # tangent.  Extend farther when SAM 2 has left a large contact gap.
    join_x = max(join_x, tx + max(50.0, 3.0 * rope_width))
    join_x = min(join_x, float(xs.max()))
    join_y = slope * join_x + intercept
    corridor = np.zeros_like(incoming_mask, dtype=np.uint8)
    thickness = max(5, round(0.72 * rope_width))
    cv2.line(corridor, (round(tx), round(ty)), (round(join_x), round(join_y)),
             255, thickness, cv2.LINE_AA)
    completed = (incoming_mask > 0) | (corridor > 0)
    added = int(completed.sum() - (incoming_mask > 0).sum())
    return completed, CompletionResult(
        True, reason, tx, ty, slope, intercept, rope_width, residual, added)


def _overlay(image: np.ndarray, incoming: np.ndarray, completed: np.ndarray,
             wound: np.ndarray, result: CompletionResult,
             manual: tuple[float, float] | None) -> np.ndarray:
    view = image.copy()
    added = completed & ~incoming
    layer = view.copy()
    layer[incoming] = (0, 150, 255)
    layer[added] = (80, 255, 80)
    layer[wound] = (70, 150, 40)
    view = cv2.addWeighted(view, 0.68, layer, 0.32, 0)
    if result.valid:
        cv2.circle(view, (round(result.tangent_x), round(result.tangent_y)),
                   18, (255, 0, 255), 7, cv2.LINE_AA)
    if manual is not None:
        cv2.drawMarker(view, (round(manual[0]), round(manual[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 45, 7, cv2.LINE_AA)
    return view


def evaluate_task(task_dir: Path) -> dict:
    manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
    output = task_dir / "rope_completion_qa"
    output.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for task in manifest["tasks"]:
        stem = Path(task["image_name"]).stem
        label_dir = task_dir / "labels" / stem
        annotation = json.loads((label_dir / "annotation.json").read_text(encoding="utf-8"))
        image = cv2.imread(str(task["image"]), cv2.IMREAD_COLOR)
        incoming = cv2.imread(str(label_dir / "incoming_rope_free_span.png"), 0) > 127
        wound = cv2.imread(str(label_dir / "wound_rope_surface.png"), 0) > 127
        manual = tuple(map(float, annotation["tangent_point"]))
        completed_auto, automatic = complete_rope_span(incoming, wound)
        completed_hint, hinted = complete_rope_span(incoming, wound, manual)
        error = math.hypot(automatic.tangent_x - manual[0], automatic.tangent_y - manual[1]) \
            if automatic.valid else math.nan
        rows.append({
            "image_name": task["image_name"], "time_s": task["time_s"],
            "remaining_rope_count": annotation.get("remaining_rope_count"),
            "automatic": asdict(automatic), "manual_tangent": list(manual),
            "automatic_tangent_error_px": error,
            "hinted_added_pixels": hinted.added_pixels,
        })
        tile = _overlay(image, incoming, completed_auto, wound, automatic, manual)
        scale = 900 / tile.shape[1]
        tile = cv2.resize(tile, (900, round(tile.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        title = f"{task['order']}/{len(manifest['tasks'])} t={task['time_s']:.1f}s err={error:.1f}px"
        cv2.putText(tile, title, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    valid_errors = [row["automatic_tangent_error_px"] for row in rows
                    if math.isfinite(row["automatic_tangent_error_px"])]
    report = {
        "schema_version": 1, "task_dir": str(task_dir), "anchors": rows,
        "summary": {
            "valid": len(valid_errors), "total": len(rows),
            "median_tangent_error_px": float(np.median(valid_errors)) if valid_errors else None,
            "p95_tangent_error_px": float(np.percentile(valid_errors, 95)) if valid_errors else None,
        },
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if tiles:
        blank = np.zeros_like(tiles[0])
        if len(tiles) % 2:
            tiles.append(blank)
        sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(output / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_task(args.task_dir.resolve())
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
