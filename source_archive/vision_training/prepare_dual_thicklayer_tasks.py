"""Build small paired thick-layer anchor tasks from the reviewed overview.

Train uses sessions 1 and 2.  Session 3 is held out in full, even if its
frames show both approach and departure; this prevents same-video leakage.
Existing task manifests and human labels are never overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / 'task_dual_thicklayer_v1'
OVERVIEW = ROOT / 'overview' / 'overview_manifest.json'
SELECTIONS = {
    'train': {
        'dual_20260923_113029': (5, 25, 55),
        'dual_20260923_113305': (5, 25, 55),
    },
    'holdout': {
        'dual_20260923_113745': (5, 25, 45, 50, 55, 58),
    },
}


def build_tasks(overview, selected_sessions):
    sessions = {row['session']: row for row in overview['sessions']}
    tasks = []
    for session_name, target_times in selected_sessions.items():
        pairs = {row['requested_s']: row for row in sessions[session_name]['paired_moments']}
        for target in target_times:
            pair = pairs[target]
            for role in ('right', 'left'):
                source = pair[role]
                image = Path(source['image'])
                if not image.is_file():
                    raise FileNotFoundError(image)
                tasks.append({
                    'order': len(tasks) + 1,
                    'image': str(image.resolve()), 'image_name': image.name,
                    'clip': session_name, 'time_s': source['host_time_s'],
                    'source_frame': source['frame'], 'pair_id': pair['pair_id'],
                    'camera': role,
                    'annotation_orientation': ('canonical_horizontal_mirror'
                                               if role == 'left' else 'original'),
                    'remaining_rope_count': 'unknown',
                    'quick_review': {'incoming_rope': '未复核', 'inner_edge': '未复核',
                                     'tangent': '未复核', 'note': ''},
                })
    return tasks


def prepare():
    overview = json.loads(OVERVIEW.read_text(encoding='utf-8'))
    # A revised time selection is safe before labeling begins, but never
    # rewrite a manifest once any human annotation exists in either split.
    if any(path.is_file()
           for split in SELECTIONS
           for path in (ROOT / split / 'labels').rglob('annotation.json')):
        raise FileExistsError('human labels exist; refusing to revise task manifests')
    for split, sessions in SELECTIONS.items():
        task_dir = ROOT / split
        manifest_path = task_dir / 'manifest.json'
        tasks = build_tasks(overview, sessions)
        manifest = {
            'schema_version': 1,
            'purpose': f'dual-camera thick-layer {split} anchors; offline only',
            'split': split, 'source_sessions': list(sessions),
            'capture_conditions': {
                'spool_outer_diameter_mm': 700, 'spool_inner_width_mm': None,
                'spool_core_diameter_mm': None, 'winding_layer': 'thick',
                'rope_diameter_mm': 4.44, 'rope_diameter_tolerance_mm': 0.09,
                'twist_pitch_mm_per_rev': 35.0,
                'twist_pitch_tolerance_mm_per_rev': 1.05,
                'light_level_each': 500, 'same_physical_spool': None,
            },
            'camera_roles': {'right': 'observes_left_flange',
                             'left': 'observes_right_flange'},
            'winding_thickness': 'thick',
            'require_remaining_rope_count': False,
            'require_tangent_point': False,
            'collect_physical_direction': True,
            'collect_incoming_visibility': True,
            'annotation_note': (
                '先判断本相机活动绳是否可见；看不见与遮挡/不确定要区分。'
                '只在活动绳朝本相机拍摄的边缘接近时填写剩余根数；远离时填unknown。'
                '活动绳近接触区SAM掩膜可能不准，切点独立标注，看不清不要猜。'),
            'classes': ['wound_rope_surface', 'flange_inner_face',
                        'incoming_rope_free_span'],
            'required_point': 'tangent_point',
            'expected_reversal_window_s': ([50, 55] if split == 'holdout' else None),
            'tasks': tasks,
        }
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'labels').mkdir(exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding='utf-8')
    return {split: len(build_tasks(overview, sessions))
            for split, sessions in SELECTIONS.items()}


if __name__ == '__main__':
    print(prepare())
