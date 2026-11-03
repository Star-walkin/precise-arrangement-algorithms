"""Build a leakage-aware manifest for the first lightweight-model distillation run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Hold out one static and one later-running window.  This is only an internal
# same-session validation split, not an independent project test set.
VALIDATION_WINDOWS = {
    'anchor_02_R_static_20260922_01__t007.093__f000216',
    'anchor_06_R_run_stop_run_20260922_01__t073.503__f002248',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-dir', type=Path, default=ROOT / 'task_v1')
    parser.add_argument('--qa-name', default='propagation_qa')
    parser.add_argument('--purpose', default='same-session proof-of-concept distillation')
    parser.add_argument('--validation-window', action='append', default=[])
    args = parser.parse_args()
    task_dir = args.task_dir.resolve()
    screened = task_dir / args.qa_name / 'screened_manifest.json'
    output = task_dir / 'distillation_dataset.json'
    source = json.loads(screened.read_text(encoding='utf-8'))
    validation_windows = set(args.validation_window) or VALIDATION_WINDOWS
    records = []
    for row in source['frames']:
        if not row['candidate']:
            continue
        split = 'validation' if row['window_id'] in validation_windows else 'train'
        records.append({
            'split': split,
            'window_id': row['window_id'],
            'index': row['index'],
            'time_s': row['time_s'],
            'is_anchor': row['is_anchor'],
            'image': row['frame'],
            'mask_dir': row['mask_dir'],
        })
    payload = {
        'schema_version': 1,
        'purpose': args.purpose,
        'warning': 'validation shares capture day/camera with training; not an independent acceptance test',
        'source_manifest': str(screened.resolve()),
        'crop_xyxy_normalized': [0.20, 0.46, 1.00, 1.00],
        # The distilled model only supplies coarse semantic regions.  The
        # original-resolution geometry stage still refines the rope line and
        # tangent, so this size keeps CPU inference practical without pretending
        # that the neural mask itself is a precision measurement.
        'input_size_wh': [512, 256],
        'classes': ['wound_rope_surface', 'flange_inner_face', 'incoming_rope_free_span'],
        'records': records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    train = sum(r['split'] == 'train' for r in records)
    validation = len(records) - train
    print(f'dataset train={train} validation={validation} total={len(records)}')


if __name__ == '__main__':
    main()
