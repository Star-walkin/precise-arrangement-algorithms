"""Read-only label encoding audit; no training, weight updates, or hardware."""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image

from .centerline_model_v5 import (
    RASTER_ENCODING_VERSION, inspect_manifest, rasterize_manifest_record,
    manifest_native_input, _resize_native_model_input, _local_points,
    _polyline_image, _resize_plane, _file_sha256, _write_json,
)


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT/'outputs/centerline_v5/review_round1/heatmap_encoding_fix.json')
    args = parser.parse_args()
    manifest = ROOT / 'outputs/centerline_v5/pilot/manifest.json'
    inspected = inspect_manifest(manifest)
    rows = []
    for record in inspected['usable']['train'] + inspected['usable']['holdout']:
        source_hash = _file_sha256(record['annotation_path'])
        x0, y0, x1, y1 = record['roi_xyxy']
        width, height = x1-x0, y1-y0
        short = min(width, height)
        local = {key: _local_points(points, record['roi_xyxy'])
                 for key, points in record['parsed_polylines'].items()}
        old_center = _polyline_image((width, height), local['centerline'],
                                    1, max(1., short*.006))
        old_seed = _polyline_image((width, height), local['free_seed'],
                                  max(1, round(short*.006)), max(1., short*.008))
        old_center = _resize_plane(old_center, 512, Image.Resampling.BILINEAR)
        old_seed = _resize_plane(old_seed, 512, Image.Resampling.BILINEAR)
        new = rasterize_manifest_record(record, 512)
        without_quality = rasterize_manifest_record(
            dict(record, parsed_quality_exclude_polygons=[]), 512)
        center = new['center_target'].numpy()[0]
        seed = new['input'].numpy()[3]
        rgb, native_seed = manifest_native_input(record)
        infer_rgb, infer_seed = _resize_native_model_input(rgb, native_seed, [512, 512])
        row = {
            'sample_id': record['sample_id'], 'status': record['label']['status'],
            'source_image_sha256': record['image_sha256'],
            'annotation_sha256': source_hash,
            'quality_sha256': record.get('quality_sha256'),
            'quality_region_kinds': [r['kind'] for r in record.get('quality_regions', [])],
            'native_roi_hw': [height, width], 'model_input_hw': [512, 512],
            'center_peak_before': float(old_center.max()),
            'center_peak_after': float(center.max()),
            'center_pixels_ge_half_before': int((old_center >= .5).sum()),
            'center_pixels_ge_half_after': int((center >= .5).sum()),
            'seed_peak_before': float(old_seed.max()),
            'seed_peak_after': float(seed.max()),
            'seed_train_infer_max_abs_difference': float(np.abs(seed-infer_seed).max()),
            'rgb_train_infer_max_abs_difference': float(
                np.abs(new['input'].numpy()[:3].transpose(1, 2, 0)-infer_rgb).max()),
            'center_valid_pixels': int(new['center_valid_mask'].sum()),
            'body_valid_pixels': int(new['body_valid_mask'].sum()),
            'body_pixels_ge_half': int((new['body_target'].numpy() >= .5).sum()),
            'all_tensors_finite': all(np.isfinite(value.numpy()).all().item()
                                      for value in new.values() if hasattr(value, 'numpy')),
            'quality_does_not_change_model_input': bool(np.array_equal(
                new['input'].numpy(), without_quality['input'].numpy())),
            'center_valid_pixels_without_quality': int(without_quality['center_valid_mask'].sum()),
            'body_valid_pixels_without_quality': int(without_quality['body_valid_mask'].sum()),
            'quality_can_only_remove_valid_pixels': all(bool(np.all(
                new[key].numpy() <= without_quality[key].numpy()))
                for key in ('center_valid_mask', 'body_valid_mask')),
        }
        if source_hash != _file_sha256(record['annotation_path']):
            raise RuntimeError('Annotation changed during audit; rerun after saving')
        rows.append(row)
    report = dict(
        raster_encoding_version=RASTER_ENCODING_VERSION,
        manifest_sha256=inspected['manifest_sha256'], rows=rows,
        labels_modified=False, model_training=False, hardware_access=False,
        interpretation='Label encoding consistency only; not model accuracy or event timing',
    )
    output = args.output
    _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
