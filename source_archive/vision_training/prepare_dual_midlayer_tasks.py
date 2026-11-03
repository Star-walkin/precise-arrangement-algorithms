"""Create paired SAM 2 anchor tasks without mixing train and held-out sessions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '现场视频' / '训练采集' / '双相机同步'
DESTINATION = ROOT / 'task_dual_midlayer_v1'
SELECTIONS = {
    'train': (0, (5, 15, 25, 35, 45, 55)),
    'holdout': (1, (10, 30, 50)),
}


def closest_pair(rows: list[dict], target_s: float) -> dict:
    return min(rows, key=lambda item: abs(float(item['right_host_time_s']) - target_s))


def prepare() -> dict:
    sessions = sorted(SOURCE.glob('dual_*'))
    if len(sessions) != 2:
        raise ValueError(f'expected exactly two dual sessions, found {len(sessions)}')
    summary = {'schema_version': 1, 'status': 'unreviewed_anchor_tasks',
               'training_session': sessions[0].name,
               'held_out_session': sessions[1].name,
               'condition': {'spool_outer_diameter_mm': 700,
                             'spool_inner_width_mm': None,
                             'spool_core_diameter_mm': None,
                             'light_level_each': 500,
                             'winding_layer': 'middle',
                             'rope_diameter_mm': 4.44,
                             'rope_diameter_tolerance_mm': 0.09,
                             'twist_pitch_mm_per_rev': 35.0,
                             'twist_pitch_tolerance_mm_per_rev': 1.05,
                             'same_spool': True,
                             'between_recordings_gap_s_approx': 20,
                             'movement_and_occlusion_events': 'unknown'}}
    for split, (session_index, target_times) in SELECTIONS.items():
        session = sessions[session_index]
        with (session / 'paired_frames.csv').open(encoding='utf-8-sig', newline='') as handle:
            pairs = list(csv.DictReader(handle))
        task_dir = DESTINATION / split
        frame_dir = task_dir / 'frames'
        frame_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'labels').mkdir(exist_ok=True)
        caps = {role: cv2.VideoCapture(str(next(session.glob(role + '_*.avi'))))
                for role in ('right', 'left')}
        tasks = []
        thumbnails = []
        try:
            for requested_time in target_times:
                pair = closest_pair(pairs, requested_time)
                pair_id = int(pair['pair_id'])
                for role in ('right', 'left'):
                    frame_index = int(pair[role + '_video_frame'])
                    caps[role].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = caps[role].read()
                    if not ok:
                        raise RuntimeError(f'{session.name} {role} frame {frame_index} unreadable')
                    if role == 'left':
                        frame = cv2.flip(frame, 1)
                    name = f'{session.name}__pair{pair_id:04d}__{role}.jpg'
                    image_path = frame_dir / name
                    if not cv2.imwrite(str(image_path), frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f'cannot save {image_path}')
                    tasks.append({
                        'order': len(tasks) + 1, 'image': str(image_path.resolve()),
                        'image_name': name, 'clip': session.name,
                        'time_s': float(pair[role + '_host_time_s']),
                        'source_frame': frame_index, 'pair_id': pair_id,
                        'camera': role,
                        'annotation_orientation': ('canonical_horizontal_mirror'
                                                   if role == 'left' else 'original'),
                        'remaining_rope_count': 'unknown',
                        'quick_review': {'incoming_rope': '未复核',
                                         'inner_edge': '未复核',
                                         'tangent': '未复核', 'note': ''},
                    })
                    thumb = cv2.resize(frame, (512, 375), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(thumb, (0, 0), (512, 32), (0, 0, 0), -1)
                    cv2.putText(thumb, f'{split} {role} t={requested_time}s',
                                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .62,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    thumbnails.append(thumb)
        finally:
            for cap in caps.values():
                cap.release()
        columns = 2
        sheet = np.vstack([np.hstack(thumbnails[i:i + columns])
                           for i in range(0, len(thumbnails), columns)])
        contact_sheet = task_dir / 'contact_sheet.jpg'
        cv2.imwrite(str(contact_sheet), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest = {
            'schema_version': 1,
            'purpose': f'dual camera middle layer {split}; offline only',
            'split': split,
            'source_session': session.name,
            'capture_conditions': summary['condition'],
            'camera_roles': {'right': 'observes_left_flange',
                             'left': 'observes_right_flange'},
            'winding_thickness': 'middle',
            'require_remaining_rope_count': False,
            'require_tangent_point': False,
            'collect_physical_direction': True,
            'collect_incoming_visibility': True,
            'annotation_note': '活动绳不在本相机视野内选“不在本相机视野内”；遮挡或无法确定选“遮挡／无法确定”。只有可见但SAM分割错误才标记SAM不可用；切点、根数或方向看不清可留空/unknown。',
            'classes': ['wound_rope_surface', 'flange_inner_face',
                        'incoming_rope_free_span'],
            'required_point': 'tangent_point',
            'contact_sheet': str(contact_sheet.resolve()),
            'tasks': tasks,
        }
        (task_dir / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        summary[split] = {'session': session.name, 'paired_moments': len(target_times),
                          'images': len(tasks), 'task_dir': str(task_dir.resolve())}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
