"""Screen paired-camera anchors without changing any human annotations.

This is a supervision inventory, not a claim that SAM 2 masks are pixel truth.
Capture sessions stay separate: train may fit models, holdout may only evaluate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_count_semantics import count_kind, gap_from_human_count, parse_count


ROOT = Path(__file__).resolve().parent / 'task_dual_midlayer_v1'
ROPE_QA_PREFIXES = ('incoming_rope_free_span:', 'incoming:')


def classify(task, annotation, qa_row, label_dir):
    visibility = annotation.get('incoming_visibility', 'unknown')
    if visibility not in ('visible', 'out_of_view', 'uncertain'):
        raise ValueError(f'{task["image_name"]}: unexpected visibility {visibility!r}')
    flags = qa_row.get('flags', [])
    rope_flags = [flag for flag in flags if flag.startswith(ROPE_QA_PREFIXES)]
    rope_mask = label_dir / 'incoming_rope_free_span.png'
    direction = annotation.get('physical_direction', 'unknown')
    kind = count_kind(task['camera'], direction)
    legacy_count = parse_count(annotation.get('remaining_rope_count'))
    count = (parse_count(annotation.get('edge_distance_rope_count'))
             if annotation.get('edge_distance_count_kind') == kind else None)
    if count is None:
        count = parse_count(annotation.get('departed_rope_count')
                            if kind == 'departed' else annotation.get('remaining_rope_count'))
    approaching = kind == 'remaining'
    mask_role = ('coarse_free_span_only'
                 if visibility == 'visible' and rope_mask.is_file() and not rope_flags
                 else 'exclude')
    # Even an unflagged SAM 2 mask is not reliable at the contact point.
    # Consumers must ignore a contact-neighborhood and refine the tangent
    # from the original-resolution image / separately reviewed human point.
    # SAM 2 often stops before the physical contact.  A human tangent far from
    # the SAM mask is evidence against that mask, not against the human point.
    point_role = ('provisional_human_point'
                  if visibility == 'visible' and annotation.get('tangent_point') is not None
                  and not any(flag.startswith('wound:tangent_far_') for flag in flags)
                  else 'exclude')
    return {
        'image_name': task['image_name'],
        'image': task['image'],
        'pair_id': task['pair_id'],
        'camera': task['camera'],
        'time_s': task['time_s'],
        'incoming_visibility': visibility,
        'presence_label': {'visible': 1, 'out_of_view': 0, 'uncertain': None}[visibility],
        'physical_direction': direction,
        'approaching_observed_edge': approaching,
        'remaining_count_label': count if approaching and visibility == 'visible' else None,
        'departed_count_label': count if kind == 'departed' and visibility == 'visible' else None,
        'axial_gap_pitch_label': (gap_from_human_count(kind, count)
                                  if visibility == 'visible' else None),
        'counting_convention': {
            'remaining': 'remaining_excludes_active_rope',
            'departed': 'departed_includes_active_rope',
        }.get(kind, 'direction_undetermined'),
        'human_count_recorded_but_not_applicable': legacy_count if not approaching else None,
        'rope_mask_role': mask_role,
        'rope_mask': str(rope_mask) if mask_role != 'exclude' else None,
        'tangent_point_role': point_role,
        'tangent_point': (annotation.get('tangent_point')
                          if point_role != 'exclude' else None),
        'qa_flags': flags,
    }


def screen_split(split, root=ROOT):
    task_dir = root / split
    manifest = json.loads((task_dir / 'manifest.json').read_text(encoding='utf-8'))
    qa = json.loads((task_dir / 'qa' / 'qa_report.json').read_text(encoding='utf-8'))
    by_name = {row['image_name']: row for row in qa['anchors']}
    records = []
    for task in manifest['tasks']:
        label_dir = task_dir / 'labels' / Path(task['image_name']).stem
        annotation_path = label_dir / 'annotation.json'
        if not annotation_path.is_file():
            raise FileNotFoundError(f'annotation missing: {annotation_path}')
        annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
        if not annotation.get('completed', False):
            raise ValueError(f'annotation incomplete: {annotation_path}')
        records.append(classify(task, annotation, by_name[task['image_name']], label_dir))
    return records


def summarize(records):
    return {
        'images': len(records),
        'visible': sum(r['presence_label'] == 1 for r in records),
        'out_of_view': sum(r['presence_label'] == 0 for r in records),
        'uncertain': sum(r['presence_label'] is None for r in records),
        'left_camera_visible': sum(r['camera'] == 'left' and r['presence_label'] == 1
                                   for r in records),
        'right_camera_visible': sum(r['camera'] == 'right' and r['presence_label'] == 1
                                    for r in records),
        'coarse_rope_masks': sum(r['rope_mask_role'] != 'exclude' for r in records),
        'provisional_tangent_points': sum(r['tangent_point_role'] != 'exclude'
                                          for r in records),
        'approaching_count_labels': sum(r['remaining_count_label'] is not None
                                        for r in records),
        'departed_count_labels': sum(r['departed_count_label'] is not None
                                     for r in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, default=ROOT)
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    train = screen_split('train', task_root)
    holdout = screen_split('holdout', task_root)
    payload = {
        'schema_version': 1,
        'status': 'screened_supervision_inventory_not_device_ready',
        'source_note': 'Human masks are coarse; free-span labels do not establish the exact contact pixel.',
        'split_rule': 'Different capture sessions; holdout must not train. Cross-condition generalization is unproven.',
        'train': {'summary': summarize(train), 'records': train},
        'holdout': {'summary': summarize(holdout), 'records': holdout},
    }
    output = task_root / 'quality_screen.json'
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    for split in ('train', 'holdout'):
        print(f'{split}: {payload[split]["summary"]}')
    print(output)


if __name__ == '__main__':
    main()
