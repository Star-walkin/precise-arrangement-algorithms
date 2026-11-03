"""Extract paired overview frames from the three 2026-09-23 thick-layer sessions.

Read-only toward the source AVI/CSV files; output is a separate overview folder.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '现场视频' / '训练采集' / '双相机同步'
OUTPUT = ROOT / 'task_dual_thicklayer_v1' / 'overview'
SESSIONS = ('dual_20260923_113029', 'dual_20260923_113305',
            'dual_20260923_113745')
TIMES = (5, 15, 25, 35, 45, 50, 55, 58)


def closest_pair(rows, target_s):
    return min(rows, key=lambda row: abs(float(row['right_host_time_s']) - target_s))


def extract_frame(cap, frame_index, session, role):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'{session} {role} frame {frame_index} unreadable')
    return frame


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for session_name in SESSIONS:
        session = SOURCE / session_name
        metadata = json.loads((session / 'session.json').read_text(encoding='utf-8'))
        if metadata['status'] != 'complete_read_only_capture' or metadata['errors']:
            raise ValueError(f'{session_name}: incomplete capture or recording errors')
        with (session / 'paired_frames.csv').open(encoding='utf-8-sig', newline='') as handle:
            pairs = list(csv.DictReader(handle))
        caps = {role: cv2.VideoCapture(str(next(session.glob(f'{role}_*.avi'))))
                for role in ('right', 'left')}
        tiles = []
        rows = []
        try:
            for target in TIMES:
                pair = closest_pair(pairs, target)
                pair_info = {'requested_s': target, 'pair_id': int(pair['pair_id'])}
                for role in ('right', 'left'):
                    index = int(pair[f'{role}_video_frame'])
                    frame = extract_frame(caps[role], index, session_name, role)
                    if role == 'left':
                        frame = cv2.flip(frame, 1)
                    filename = f'{session_name}__t{target:02d}__{role}.jpg'
                    image_path = OUTPUT / filename
                    cv2.imwrite(str(image_path), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    pair_info[role] = {'frame': index, 'image': str(image_path.resolve()),
                                       'host_time_s': float(pair[f'{role}_host_time_s'])}
                    thumb = cv2.resize(frame, (640, 469), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(thumb, (0, 0), (640, 38), (0, 0, 0), -1)
                    cv2.putText(thumb, f'{session_name}  t={target}s  {role}',
                                (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .65,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    tiles.append(thumb)
                rows.append(pair_info)
        finally:
            for cap in caps.values():
                cap.release()
        sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
        sheet_path = OUTPUT / f'{session_name}_contact_sheet.jpg'
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])
        results.append({'session': session_name, 'sheet': str(sheet_path.resolve()),
                        'paired_moments': rows})
    manifest_path = OUTPUT / 'overview_manifest.json'
    manifest_path.write_text(json.dumps({'sessions': results}, ensure_ascii=False, indent=2),
                             encoding='utf-8')
    return manifest_path


if __name__ == '__main__':
    print(prepare())
