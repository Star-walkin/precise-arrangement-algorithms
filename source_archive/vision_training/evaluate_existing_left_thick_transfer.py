"""Read-only check of the older 4.15 mm left-camera model on new 4.44 mm anchors."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from edge_count_semantics import count_kind, human_count_from_gap, parse_count


ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT / 'task_dual_thicklayer_v1'


def main():
    import sys
    prototype = ROOT.parent / 'vision_prototype'
    sys.path.insert(0, str(prototype))
    from analyze_video import load_config
    from semantic_shadow import SemanticShadowProcessor

    config_path = prototype / 'config' / 'left_camera.json'
    processor = SemanticShadowProcessor.from_config(load_config(config_path), config_path)
    rows = []
    for split in ('train', 'holdout'):
        manifest = json.loads((TASK_ROOT / split / 'manifest.json').read_text(encoding='utf-8'))
        for task in manifest['tasks']:
            if task['camera'] != 'left':
                continue
            folder = TASK_ROOT / split / 'labels' / Path(task['image_name']).stem
            annotation = json.loads((folder / 'annotation.json').read_text(encoding='utf-8'))
            if annotation.get('incoming_visibility') != 'visible':
                continue
            canonical = cv2.imread(task['image'], cv2.IMREAD_COLOR)
            if canonical is None:
                raise RuntimeError(f'cannot open {task["image"]}')
            processor.reset()
            result = processor.process(cv2.flip(canonical, 1), render=False)
            kind = count_kind('left', annotation.get('physical_direction', 'unknown'))
            truth = parse_count(annotation.get('remaining_rope_count')
                                if kind == 'remaining' else annotation.get('departed_rope_count'))
            prediction = (human_count_from_gap(kind, result.remaining_rope_count)
                          if result.valid else None)
            point = annotation.get('tangent_point')
            row = {'split': split, 'image_name': task['image_name'],
                   'physical_direction': annotation.get('physical_direction'),
                   'count_kind': kind, 'human_count': truth,
                   'valid': result.valid, 'reason': result.reason,
                   'predicted_axial_gap_pitch': result.remaining_rope_count if result.valid else None,
                   'predicted_count_in_human_convention': prediction,
                   'count_abs_error': (abs(prediction - truth)
                                       if prediction is not None and truth is not None else None),
                   'tangent_error_px': (math.hypot(result.tangent_x-point[0],
                                                   result.tangent_y-point[1])
                                        if point is not None and math.isfinite(result.tangent_x)
                                        else None),
                   'processing_ms': result.processing_ms}
            rows.append(row)
    error = [r['count_abs_error'] for r in rows if r['count_abs_error'] is not None]
    summary = {'frames': len(rows), 'valid': sum(r['valid'] for r in rows),
               'count_comparable': len(error),
               'count_mae': float(np.mean(error)) if error else None,
               'caution': 'Old model was trained/calibrated for 4.15 mm rope; this is only a transfer diagnostic.'}
    report = {'status': 'offline_transfer_diagnostic_not_for_device_control',
              'summary': summary, 'rows': rows}
    output = TASK_ROOT / 'old_left_model_transfer.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
