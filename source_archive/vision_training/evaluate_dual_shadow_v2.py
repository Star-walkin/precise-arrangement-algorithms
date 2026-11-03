"""Read-only per-frame evaluation of the two camera-specific hybrid shadows.

Anchors are independent still images. This does not validate motion direction,
occlusion forecasting, timing or actuator readiness.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from edge_count_semantics import count_kind, gap_from_human_count, parse_count


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK = ROOT / 'vision_training' / 'task_dual_thicklayer_v1'


def evaluate(task_root: Path, *, with_fallback: bool = False) -> dict:
    prototype = ROOT / 'vision_prototype'
    sys.path.insert(0, str(prototype))
    from analyze_video import identity_registration, load_config
    from semantic_shadow import SemanticShadowProcessor
    from visual_app import VisionProcessor

    processors = {}
    configs = {}
    for camera in ('left', 'right'):
        config_path = prototype / 'config' / f'{camera}_camera.json'
        configs[camera] = load_config(config_path)
        processors[camera] = SemanticShadowProcessor.from_config(
            configs[camera], config_path)
        if processors[camera] is None:
            raise RuntimeError(f'{camera} read-only shadow is not configured')
    rows = []
    for split in ('train', 'holdout'):
        manifest = json.loads((task_root / split / 'manifest.json').read_text(encoding='utf-8'))
        for task in manifest['tasks']:
            camera = task['camera']
            annotation_path = (task_root / split / 'labels' /
                               Path(task['image_name']).stem / 'annotation.json')
            annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
            image = cv2.imread(task['image'], cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(task['image'])
            # Left annotation frames are canonical mirrored images; the live
            # processor receives raw orientation and mirrors them internally.
            raw = cv2.flip(image, 1) if camera == 'left' else image
            processor = processors[camera]
            processor.reset()
            legacy = None
            if with_fallback and processor.guarded_weak_rope_fallback:
                legacy = VisionProcessor(configs[camera], raw.shape[1], raw.shape[0],
                                         identity_registration('offline_guarded_test'))
                legacy = legacy.process(raw, 0, 0., render=False)
            result = processor.process(raw, render=False, legacy_result=legacy)
            kind = count_kind(camera, annotation.get('physical_direction', 'unknown'))
            human_count = parse_count(annotation.get('edge_distance_rope_count'))
            truth_gap = gap_from_human_count(kind, human_count)
            truth_gap = truth_gap if annotation.get('incoming_visibility') == 'visible' else None
            point = annotation.get('tangent_point')
            row = {
                'split': split, 'camera': camera, 'image_name': task['image_name'],
                'visibility_truth': annotation.get('incoming_visibility'),
                'visibility_evidence': result.presence_evidence,
                'valid': result.valid, 'reason': result.reason,
                'incoming_mask_pixels': result.incoming_mask_pixels,
                'count_method': result.count_method,
                'truth_axial_gap_pitch': truth_gap,
                'estimated_axial_gap_pitch': (result.remaining_rope_count
                                               if result.valid else None),
                'absolute_gap_error': (abs(result.remaining_rope_count - truth_gap)
                                       if result.valid and truth_gap is not None else None),
                'tangent_error_px': (math.hypot(result.tangent_x - point[0],
                                                result.tangent_y - point[1])
                                     if result.valid and point is not None else None),
                'bright_dark_row_pitch_px': (result.row_pitch_candidate_px
                                             if math.isfinite(result.row_pitch_candidate_px)
                                             else None),
                'bright_dark_pitch_confidence': result.row_pitch_candidate_confidence,
                'processing_ms': result.processing_ms,
            }
            rows.append(row)

    def summarize(subset: list[dict]) -> dict:
        visible = [r for r in subset if r['visibility_truth'] == 'visible']
        absent = [r for r in subset if r['visibility_truth'] == 'out_of_view']
        errors = [r['absolute_gap_error'] for r in subset
                  if r['absolute_gap_error'] is not None]
        return {
            'frames': len(subset),
            'visible_truth': len(visible),
            'visible_valid': sum(r['valid'] for r in visible),
            'out_of_view_truth': len(absent),
            'out_of_view_rejected': sum(not r['valid'] for r in absent),
            'count_comparable': len(errors),
            'count_mae_rope_pitch': float(np.mean(errors)) if errors else None,
            'within_one_pitch': sum(error <= 1 for error in errors),
            'p95_processing_ms': float(np.percentile(
                [r['processing_ms'] for r in subset], 95)) if subset else None,
        }

    report = {
        'status': 'read_only_shadow_diagnostic_not_actuator_ready',
        'note': ('Presence thresholds were chosen with the current annotated '
                 'sessions; train/holdout results are diagnostic, not an '
                 'independent cross-site or cross-spool acceptance test.'),
        'summaries': {split: {camera: summarize([
            r for r in rows if r['split'] == split and r['camera'] == camera])
                             for camera in ('left', 'right')}
                      for split in ('train', 'holdout')},
        'rows': rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, default=DEFAULT_TASK)
    parser.add_argument('--with-fallback', action='store_true',
                        help='Evaluate the read-only guarded legacy/ridge fallback too')
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    report = evaluate(task_root, with_fallback=args.with_fallback)
    output = task_root / ('dual_shadow_v3_report.json' if args.with_fallback
                          else 'dual_shadow_v2_report.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['summaries'], ensure_ascii=False, indent=2))
    print(output)


if __name__ == '__main__':
    main()
