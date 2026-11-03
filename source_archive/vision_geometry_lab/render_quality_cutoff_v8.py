"""Render confirmed target cutoffs for human QA; no labels or hardware writes."""
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np

from .centerline_quality_v5 import load_label_quality
from .centerline_quality_cutoff_v8 import cutoff_geometry
from .geometry import read_image, write_image


ROOT = Path(__file__).resolve().parents[1]


def panel(sample):
    if hashlib.sha256(Path(sample['image_path']).read_bytes()).hexdigest() != sample['image_sha256']:
        raise ValueError('Source image changed; review the source before rendering')
    label = json.loads(Path(sample['annotation_path']).read_text('utf-8-sig'))
    quality = load_label_quality(sample)
    trusted = quality['supervision_polylines']
    raw = label['polylines']
    geometry = cutoff_geometry(raw, sample['image_shape_hw'], 7)
    center = np.asarray(raw['centerline'], float)
    points = center[-4:]
    lo = np.maximum([0, 0], np.floor(points.min(axis=0)-[110, 90])).astype(int)
    hi = np.minimum(sample['image_shape_hw'][::-1],
                    np.ceil(points.max(axis=0)+[110, 90])).astype(int)
    original = read_image(sample['image_path'])
    patch = original[lo[1]:hi[1], lo[0]:hi[0]].copy()
    scale = 760 / patch.shape[1]
    view = cv2.resize(patch, (760, round(patch.shape[0]*scale)))

    def pix(values):
        return np.rint((np.asarray(values, float)-lo)*scale).astype(np.int32)

    for name in ('centerline', 'edge_flange', 'edge_surface'):
        cv2.polylines(view, [pix(raw[name])], False, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.polylines(view, [pix(trusted[name])], False, (40, 230, 80),
                      2 if name == 'centerline' else 1, cv2.LINE_AA)
    origin = np.asarray(geometry['origin_xy'])
    normal = np.asarray(geometry['unit_cross_section_direction_xy'])
    a, b = pix([origin-normal*80, origin+normal*80])
    cv2.line(view, a, b, (220, 80, 230), 2, cv2.LINE_AA)
    for index in (5, 6, 7):
        p = pix([center[index]])[0]
        color = (40, 230, 80) if index < 7 else (40, 60, 250)
        cv2.circle(view, p, 5, color, -1, cv2.LINE_AA)
        cv2.putText(view, f'C{index+1}', tuple(p+[6, 21]),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    title = sample['sample_id'].split('__')[0]
    canvas = np.full((view.shape[0]+85, 760, 3), 245, np.uint8)
    canvas[45:45+view.shape[0]] = view
    cv2.putText(canvas, f'{title}: C7 trusted | after cross-section UNKNOWN',
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .63, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Green: retained target / purple: cutoff / red C8: excluded',
                (10, canvas.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT /
                        'vision_geometry_lab/outputs/centerline_v7/review_batch01/manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT /
                        'vision_geometry_lab/outputs/centerline_v8_quality_confirmed/trusted_cutoffs.png')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep prior QA output; choose a new output path')
    manifest = json.loads(args.manifest.read_text('utf-8-sig'))
    samples = [s for s in manifest['samples'] if s['sample_id'].startswith(('C07__', 'C08__'))]
    if len(samples) != 2:
        raise ValueError('Expected the two confirmed cutoff samples')
    panels = [panel(sample) for sample in samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_image(args.output, np.vstack(panels))
    print(str(args.output))


if __name__ == '__main__':
    main()
