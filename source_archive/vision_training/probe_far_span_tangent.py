"""Offline feasibility probe: fit only the SAM 2 rope's far free-span pixels.

Uses human tangent X solely to define/evaluate a benchmark crop.  It is not a
runtime tangent detector or an independent ground-truth validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent / 'task_dual_thicklayer_v1'
GAPS = (100, 200, 300, 400, 600)


def fit_far_line(mask: np.ndarray, tangent_x: float, gap: int):
    ys, xs = np.where(mask)
    keep = (xs >= tangent_x + gap) & (xs <= tangent_x + 1800)
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 100:
        return None
    bin_size = 40
    centers = []
    for start in range(int(xs.min()), int(xs.max()) + 1, bin_size):
        values = ys[(xs >= start) & (xs < start + bin_size)]
        if len(values) < 15:
            continue
        if np.percentile(values, 90) - np.percentile(values, 10) > 55:
            continue
        centers.append((start + bin_size / 2, float(np.median(values))))
    if len(centers) < 8:
        return None
    points = np.asarray(centers, dtype=np.float64)
    # Robustly pick a line from pairs of bins; static fixtures leaking into
    # the mask must not dominate ordinary least squares.
    best = None
    for i in range(len(points)):
        for j in range(i + 4, len(points)):
            dx = points[j, 0] - points[i, 0]
            if dx < 150:
                continue
            slope = (points[j, 1] - points[i, 1]) / dx
            if abs(slope) > .8:
                continue
            intercept = points[i, 1] - slope * points[i, 0]
            residual = np.abs(points[:, 1] - slope * points[:, 0] - intercept)
            inliers = residual <= 18
            score = (int(inliers.sum()), float(np.ptp(points[inliers, 0]))
                     if inliers.any() else 0.0)
            if best is None or score > best[0]:
                best = (score, inliers)
    if best is None or best[0][0] < 8 or best[0][1] < 300:
        return None
    slope, intercept = np.polyfit(points[best[1], 0], points[best[1], 1], 1)
    return {'projected_tangent_y': float(slope * tangent_x + intercept),
            'slope': float(slope), 'inlier_bins': best[0][0],
            'span_px': best[0][1]}


def intersect_free_span_with_wound(incoming_mask: np.ndarray,
                                   wound_mask: np.ndarray):
    """Geometry-only candidate: fixed far ROI, then line/top-of-wound crossing."""
    height, width = incoming_mask.shape
    reference_x = round(width * .50)
    fit = fit_far_line(incoming_mask, reference_x, round(width * .10))
    if fit is None:
        return None
    intercept = fit['projected_tangent_y'] - fit['slope'] * reference_x
    valid_columns = wound_mask.sum(axis=0) >= 20
    top = np.argmax(wound_mask, axis=0).astype(np.float64)
    start, stop = round(width * .30), round(width * .68)
    candidates = []
    for x in range(start, stop, 4):
        neighborhood = top[max(0, x-10):min(width, x+11)]
        valid = valid_columns[max(0, x-10):min(width, x+11)]
        if valid.sum() < 12:
            continue
        edge_y = float(np.median(neighborhood[valid]))
        line_y = fit['slope'] * x + intercept
        candidates.append((abs(line_y - edge_y), x, line_y))
    if not candidates:
        return None
    residual, x, y = min(candidates)
    return {'tangent_x': float(x), 'tangent_y': float(y),
            'intersection_residual_px': float(residual),
            'free_span_inlier_bins': fit['inlier_bins'],
            'free_span_slope': fit['slope']}


def evaluate(root=ROOT):
    rows = []
    for split in ('train', 'holdout'):
        manifest = json.loads((root / split / 'manifest.json').read_text(encoding='utf-8'))
        for task in manifest['tasks']:
            folder = root / split / 'labels' / Path(task['image_name']).stem
            data = json.loads((folder / 'annotation.json').read_text(encoding='utf-8'))
            mask_path = folder / 'incoming_rope_free_span.png'
            tangent = data.get('tangent_point')
            if data.get('incoming_visibility') != 'visible' or not tangent or not mask_path.is_file():
                continue
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert('L')) >= 128
            wound_path = folder / 'wound_rope_surface.png'
            with Image.open(wound_path) as source:
                wound_mask = np.asarray(source.convert('L')) >= 128
            row = {'split': split, 'image_name': task['image_name'],
                   'human_tangent_x': tangent[0], 'human_tangent_y': tangent[1],
                   'fits': {}}
            intersection = intersect_free_span_with_wound(mask, wound_mask)
            if intersection is not None:
                intersection['absolute_x_error_px'] = abs(intersection['tangent_x'] - tangent[0])
                intersection['absolute_y_error_px'] = abs(intersection['tangent_y'] - tangent[1])
            row['geometry_only_intersection'] = intersection
            for gap in GAPS:
                fit = fit_far_line(mask, float(tangent[0]), gap)
                if fit is not None:
                    fit['absolute_y_error_px'] = abs(fit['projected_tangent_y'] - tangent[1])
                row['fits'][str(gap)] = fit
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, default=ROOT)
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    rows = evaluate(task_root)
    report = {'status': 'feasibility_probe_not_runtime_detector', 'rows': rows}
    for gap in GAPS:
        errors = [row['fits'][str(gap)]['absolute_y_error_px'] for row in rows
                  if row['fits'][str(gap)] is not None]
        report[f'gap_{gap}_summary'] = {
            'usable': len(errors), 'median_y_error_px': float(np.median(errors)) if errors else None,
            'p90_y_error_px': float(np.percentile(errors, 90)) if errors else None,
        }
    candidates = [r['geometry_only_intersection'] for r in rows
                  if r['geometry_only_intersection'] is not None]
    report['geometry_only_summary'] = {
        'usable': len(candidates),
        'median_x_error_px': (float(np.median([r['absolute_x_error_px'] for r in candidates]))
                              if candidates else None),
        'median_y_error_px': (float(np.median([r['absolute_y_error_px'] for r in candidates]))
                              if candidates else None),
        'p90_x_error_px': (float(np.percentile([r['absolute_x_error_px'] for r in candidates], 90))
                           if candidates else None),
        'p90_y_error_px': (float(np.percentile([r['absolute_y_error_px'] for r in candidates], 90))
                           if candidates else None),
    }
    output = task_root / 'far_span_tangent_probe.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
