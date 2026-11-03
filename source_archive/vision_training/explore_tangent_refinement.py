"""Inspect original-resolution rope-ridge signals around six manual tangents.

This is an offline research utility.  Manual tangent coordinates are used only
to report aligned diagnostics; the candidate detector itself must not read them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8) * 255


def fit_mask_line(mask):
    ys, xs = np.nonzero(mask)
    # Ignore ragged one-pixel mask fringes and keep runtime deterministic.
    points = np.column_stack([xs[::4], ys[::4]]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).flatten()
    if vx < 0:
        vx, vy = -vx, -vy
    return float(vx), float(vy), float(x0), float(y0)


def sample_profiles(gray, saturation, line, x_values, half_width=44):
    vx, vy, x0, y0 = line
    slope = vy / vx
    normal = np.asarray([-vy, vx], dtype=np.float32)
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float32)
    center_y = y0 + slope * (x_values - x0)
    map_x = x_values[:, None].astype(np.float32) + normal[0] * offsets[None, :]
    map_y = center_y[:, None].astype(np.float32) + normal[1] * offsets[None, :]
    gray_profiles = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    sat_profiles = cv2.remap(saturation, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return center_y, gray_profiles.astype(np.float32), sat_profiles.astype(np.float32)


def rolling_mean(values, window):
    return cv2.blur(values.reshape(1, -1).astype(np.float32), (window, 1))[0]


def detect_endpoint(x_values, gray_profiles, sat_profiles, coarse_endpoint_x):
    # Search for a bright, low-saturation ridge near the model/SAM centreline.
    center = gray_profiles.shape[1] // 2
    core = slice(center - 10, center + 11)
    side = np.r_[0:center - 22, center + 23:gray_profiles.shape[1]]
    ridge = np.percentile(gray_profiles[:, core], 80, axis=1) - np.median(
        gray_profiles[:, side], axis=1)
    neutral = np.median(sat_profiles[:, side], axis=1) - np.percentile(
        sat_profiles[:, core], 30, axis=1)
    score = rolling_mean(np.maximum(ridge, 0), 17) + .18 * rolling_mean(
        np.maximum(neutral, 0), 17)

    # The right half contains the clearly visible free span.  Its robust level
    # adapts to exposure.  Moving left, require sustained evidence rather than
    # accepting a single metallic highlight.
    right = score[int(.58 * len(score)):int(.92 * len(score))]
    baseline = float(np.percentile(right, 65)) if len(right) else 0.0
    threshold = max(3.0, .32 * baseline)
    good = score >= threshold
    support = np.convolve(good.astype(np.int16), np.ones(31, dtype=np.int16), mode='same')
    # SAM/model masks may stop after the visually perceived tangent.  Search a
    # narrow original-resolution corridor around that coarse endpoint for the
    # first sustained change in transverse rope contrast.  This does not use a
    # fixed global brightness threshold.
    smooth = rolling_mean(score, 13)
    before = np.r_[np.repeat(smooth[0], 14), smooth[:-14]]
    after = np.r_[smooth[14:], np.repeat(smooth[-1], 14)]
    change = after - before
    search = np.flatnonzero((x_values >= coarse_endpoint_x - 380) &
                            (x_values <= coarse_endpoint_x + 80))
    endpoint_index = int(search[np.argmax(change[search])]) if len(search) else 0
    return float(x_values[endpoint_index]), score, threshold, baseline, change


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', type=Path, default=ROOT / 'task_v1')
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    output = task_dir / 'tangent_refinement_exploration'
    manifest = json.loads((task_dir / 'manifest.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_row in manifest['tasks']:
        label_dir = task_dir / 'labels' / Path(task_row['image']).stem
        annotation = json.loads((label_dir / 'annotation.json').read_text(encoding='utf-8'))
        tangent = annotation['tangent_point']
        bgr = cv2.imread(task_row['image'], cv2.IMREAD_COLOR)
        mask = cv2.imread(str(label_dir / 'incoming_rope_free_span.png'), cv2.IMREAD_GRAYSCALE)
        mask = largest_component((mask >= 128).astype(np.uint8) * 255)
        line = fit_mask_line(mask)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        saturation = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        ys, xs = np.nonzero(mask)
        projection = (xs - line[2]) * line[0] + (ys - line[3]) * line[1]
        coarse_projection = float(np.percentile(projection, .5))
        coarse_endpoint_x = float(line[2] + coarse_projection * line[0])
        # The diagnostic window is selected only from the predicted mask; the
        # manual point below is used exclusively for error reporting.
        x_values = np.arange(max(0, int(coarse_endpoint_x - 500)),
                             min(gray.shape[1], int(coarse_endpoint_x + 850)),
                             dtype=np.float32)
        center_y, gray_profiles, sat_profiles = sample_profiles(
            gray, saturation, line, x_values)
        predicted_x, score, threshold, baseline, change = detect_endpoint(
            x_values, gray_profiles, sat_profiles, coarse_endpoint_x)
        slope = line[1] / line[0]
        predicted_y = line[3] + slope * (predicted_x - line[2])
        error = float(np.hypot(predicted_x - tangent[0], predicted_y - tangent[1]))
        along = float((predicted_x - tangent[0]) * line[0] +
                      (predicted_y - tangent[1]) * line[1])
        cross = float(abs(-(predicted_x - tangent[0]) * line[1] +
                          (predicted_y - tangent[1]) * line[0]))
        rows.append({
            'image_name': task_row['image_name'], 'manual_tangent': tangent,
            'predicted_tangent': [predicted_x, float(predicted_y)],
            'euclidean_error_px': error, 'along_error_px': along,
            'cross_error_px': cross, 'threshold': threshold,
            'visible_rope_baseline': baseline, 'coarse_endpoint_x': coarse_endpoint_x,
        })

        # Save a compact graph for visual diagnosis without adding matplotlib.
        graph_h, graph_w = 260, 1200
        canvas = np.full((graph_h, graph_w, 3), 255, np.uint8)
        normalized_x = np.linspace(0, graph_w - 1, len(score)).astype(int)
        maximum = max(float(np.percentile(score, 99)), threshold, 1.0)
        graph_y = (graph_h - 25 - np.clip(score / maximum, 0, 1) * (graph_h - 45)).astype(int)
        points = np.column_stack([normalized_x, graph_y]).astype(np.int32)
        cv2.polylines(canvas, [points], False, (30, 30, 30), 2)
        manual_column = int(np.interp(tangent[0], x_values, normalized_x))
        predicted_column = int(np.interp(predicted_x, x_values, normalized_x))
        threshold_y = int(graph_h - 25 - min(1, threshold / maximum) * (graph_h - 45))
        cv2.line(canvas, (0, threshold_y), (graph_w - 1, threshold_y), (180, 180, 0), 1)
        cv2.line(canvas, (manual_column, 0), (manual_column, graph_h - 1), (255, 0, 255), 2)
        cv2.line(canvas, (predicted_column, 0), (predicted_column, graph_h - 1), (0, 0, 255), 2)
        cv2.putText(canvas, f"manual magenta / predicted red / error {error:.1f}px",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(output / f'{Path(task_row["image_name"]).stem}_profile.png'), canvas)

    errors = np.asarray([row['euclidean_error_px'] for row in rows])
    report = {
        'schema_version': 1, 'status': 'exploratory_not_product_algorithm',
        'median_error_px': float(np.median(errors)),
        'p95_error_px': float(np.percentile(errors, 95)), 'anchors': rows,
    }
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
