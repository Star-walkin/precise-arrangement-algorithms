"""Extract a small canonical/mirrored SAM 2 anchor set for the left camera."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
VIDEO_DIR = PROJECT / '现场视频' / '训练采集' / '左侧相机' / '厚层'
TASK_DIR = ROOT / 'task_left_v1'
FRAME_DIR = TASK_DIR / 'frames'

ANCHORS = [
    ('L_static_20260922_01.avi', 0.0, '全程静止：起始'),
    ('L_static_20260922_01.avi', 50.0, '全程静止：中段'),
    ('L_run_stop_run_20260922_01.avi', 10.0, '首次启动前'),
    ('L_run_stop_run_20260922_01.avi', 30.0, '首次运行中'),
    ('L_run_stop_run_20260922_01.avi', 55.0, '45～62秒停机段'),
    ('L_run_stop_run_20260922_01.avi', 90.0, '再次运行中'),
    ('L_run_stop_run_20260922_01.avi', 150.0, '146秒后停机段'),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    (TASK_DIR / 'labels').mkdir(parents=True, exist_ok=True)
    tasks = []; thumbnails = []
    video_hashes = {}
    for order, (filename, requested_time, note) in enumerate(ANCHORS, 1):
        path = VIDEO_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)); count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = count / fps
        time_s = min(requested_time, max(0.0, duration - 1 / fps))
        frame_index = min(count - 1, round(time_s * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read(); capture.release()
        if not ok:
            raise RuntimeError(f'cannot read {filename} at {time_s:.3f}s')
        # The production left-camera profile mirrors horizontally before all
        # analysis.  Label that same canonical orientation so both cameras can
        # share model structure and geometry conventions.
        canonical = cv2.flip(frame, 1)
        image_name = (f'{Path(filename).stem}__t{time_s:07.3f}__f{frame_index:06d}'
                      '__canonical_mirror.jpg')
        destination = FRAME_DIR / image_name
        cv2.imwrite(str(destination), canonical, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumbnail = cv2.resize(canonical, (512, 375), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumbnail, (0, 0), (512, 32), (0, 0, 0), -1)
        cv2.putText(thumbnail, f'{Path(filename).stem}  t={time_s:.1f}s', (7, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        thumbnails.append(thumbnail)
        tasks.append({
            'order': order, 'image': str(destination.resolve()),
            'image_name': image_name, 'clip': filename, 'time_s': time_s,
            'source_frame': frame_index, 'source_orientation': 'left_camera_original',
            'annotation_orientation': 'canonical_horizontal_mirror',
            'remaining_rope_count': 'unknown',
            'quick_review': {'incoming_rope': '未复核', 'inner_edge': '未复核',
                             'tangent': '未复核', 'note': note},
        })
        if filename not in video_hashes:
            video_hashes[filename] = sha256(path)
    columns = 3; sheet_rows = math.ceil(len(thumbnails) / columns)
    sheet = np.full((sheet_rows * 375, columns * 512, 3), 235, np.uint8)
    for index, thumbnail in enumerate(thumbnails):
        y = index // columns * 375; x = index % columns * 512
        sheet[y:y + 375, x:x + 512] = thumbnail
    contact_sheet = TASK_DIR / 'anchor_contact_sheet.jpg'
    cv2.imwrite(str(contact_sheet), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    manifest = {
        'schema_version': 1,
        'purpose': 'left-camera thick-layer SAM 2 anchors; offline only',
        'camera_serial': 'DA9626065',
        'camera_role': 'left_camera_observes_right_flange',
        'winding_thickness': 'thick',
        'canonical_transform': 'horizontal_mirror',
        'require_remaining_rope_count': True,
        'classes': ['wound_rope_surface', 'flange_inner_face',
                    'incoming_rope_free_span'],
        'required_point': 'tangent_point',
        'source_video_sha256': video_hashes,
        'contact_sheet': str(contact_sheet.resolve()),
        'tasks': tasks,
    }
    manifest_path = TASK_DIR / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding='utf-8')
    print(f'prepared={len(tasks)} manifest={manifest_path}')


if __name__ == '__main__':
    main()
