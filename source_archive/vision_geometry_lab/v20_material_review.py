"""Bound raw-image review only: no events, model inference, enhancement or hardware.

The uniform schedule is fixed at 0, 5, ..., 60 seconds, not selected from events.
The display crop is inherited from V18 startup seeds, not an inferred material
boundary. Source pixels remain in each camera's original (unmirrored) direction.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .v18_front_diagnostics import (
    ARRAY_SHAPES, diagnostic_roi, font, mirror_roi, sha,
    validate_measurement, validate_original_frame,
)

TRIAL = Path(__file__).resolve().parent / 'outputs' / 'v18_gradient_units'
OUTPUT = Path(__file__).resolve().parent / 'outputs' / 'v20_material_audit' / 'review'
UNIFORM_TIMES_S = tuple(range(0, 61, 5))
LABELS = {'left': '左相机 · 拍右法兰', 'right': '右相机 · 拍左法兰'}


def load_bound_session(trial, session):
    trial = Path(trial).resolve()
    protocol = trial / 'protocol.json'
    index_path = trial / session / 'index.json'
    npz_path = trial / session / 'measurement.npz'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    hashes = {str(path): sha(path) for path in (protocol, index_path, npz_path)}
    if index.get('output_status') != 'complete_observations_only_no_events':
        raise ValueError('source measurement must be complete')
    if index.get('measurement_sha256') != hashes[str(npz_path)]:
        raise ValueError('source measurement hash mismatch')
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ('times_s', 'rows_y', *ARRAY_SHAPES)}
    validate_measurement(arrays, index, hashes[str(protocol)])
    video = index['video']
    if video['camera'] != session:
        raise ValueError('source camera/session mismatch')
    video_hash = sha(video['path'])
    if video_hash != video.get('sha256'):
        raise ValueError('source video hash mismatch')
    hashes[str(Path(video['path']).resolve())] = video_hash
    canonical_roi, roi_info = diagnostic_roi(arrays, video['width'], video['height'])
    raw_roi = mirror_roi(canonical_roi, video['width'], session)
    return dict(index=index, video=video, raw_roi=raw_roi, roi_info=roi_info,
                input_hashes=hashes, protocol_sha256=hashes[str(protocol)])


def schedule_records(index, times_s=UNIFORM_TIMES_S):
    """Exact source frames only, never nearest-frame substitution."""
    records = {r['frame_index']: r for r in index['records']}
    fps = float(index['video']['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('invalid fps')
    result = []
    for time_s in times_s:
        frame = round(float(time_s) * fps)
        if abs(frame / fps - float(time_s)) > 1e-8:
            raise ValueError('requested time is not an exact source frame')
        record = records.get(frame)
        if record is None:
            raise ValueError(f'missing bound source frame {frame}')
        if abs(float(record['time_s']) - frame / fps) > 1e-8:
            raise ValueError('source record time mismatch')
        result.append(record)
    return result


def decode_bound(cap, record, video):
    expected = int(record['frame_index'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, expected)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f'source decoding failed at frame {expected}')
    actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    validate_original_frame(frame, actual, record, video)
    return frame


def open_bound_video(video):
    cap = cv2.VideoCapture(video['path'])
    if not cap.isOpened():
        raise ValueError('cannot open bound source video')
    if (abs(cap.get(cv2.CAP_PROP_FPS) - float(video['fps'])) > 1e-6
            or int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) != video['frame_count']):
        cap.release()
        raise ValueError('source video metadata mismatch')
    return cap


def raw_images(frame, raw_roi):
    x0, y0, x1, y1 = map(int, raw_roi)
    h, w = frame.shape[:2]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('raw ROI outside source')
    picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return picture, picture.crop((x0, y0, x1, y1))


def save_new_image(picture, path, output):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'will not overwrite {path}')
    picture.save(path)
    return dict(path=str(path.relative_to(output)), sha256=sha(path), size=list(picture.size))


def base_manifest():
    return dict(schema='v20_raw_material_review_1', script_sha256=sha(__file__),
                selection='fixed_uniform_source_video_times_not_event_selected',
                ground_truth_files_opened=False, enhancement=False,
                mirrored=False, algorithm_overlays=False, model_inference=False,
                event_output=False, hardware_access=False, main_ui_modified=False,
                sessions={}, input_hashes={})


def verify_unchanged(input_hashes):
    for path, digest in input_hashes.items():
        if sha(path) != digest:
            raise ValueError(f'input changed during review export: {path}')


def first_previews(trial=TRIAL, output=OUTPUT):
    """Priority full-frame previews for independent material ROI selection."""
    output = Path(output).resolve()
    targets = [output / f'source_first_{side}.png' for side in LABELS]
    targets += [output / 'source_first_manifest.json']
    if any(path.exists() for path in targets):
        raise FileExistsError('first-frame artifacts already exist; use a new directory')
    loaded = {side: load_bound_session(trial, side) for side in LABELS}
    output.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest()
    manifest['selection'] = 'first_source_frame_only_not_event_selected'
    for side, data in loaded.items():
        record = schedule_records(data['index'], (0,))[0]
        cap = open_bound_video(data['video'])
        try:
            frame = decode_bound(cap, record, data['video'])
        finally:
            cap.release()
        picture, _ = raw_images(frame, data['raw_roi'])
        preview = ImageOps.contain(picture, (1200, 1200), Image.Resampling.LANCZOS)
        artifact = save_new_image(preview, output / f'source_first_{side}.png', output)
        manifest['input_hashes'].update(data['input_hashes'])
        manifest['sessions'][side] = dict(video=data['video'], source_frame=0, time_s=0.,
            original_frame_sha256=record['frame_sha256'], hash_and_decode_index_verified=True,
            preview=artifact, resize_only=True, prospective_uniform_raw_roi=data['raw_roi'])
    verify_unchanged(manifest['input_hashes'])
    (output / 'source_first_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return output


def _fit(canvas, picture, box):
    x, y, w, h = box
    resized = ImageOps.contain(picture, (w, h), Image.Resampling.LANCZOS)
    canvas.paste(resized, (x + (w - resized.width) // 2, y + (h - resized.height) // 2))


def contact_sheet(cards, side, page, pages):
    cell_w, cell_h = 1000, 620
    canvas = Image.new('RGB', (2000, 160 + 2 * cell_h), '#151c25')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 12), f'V20 原始材料复核 | {LABELS[side]} | 第 {page}/{pages} 页', font=font(29), fill='white')
    draw.text((24, 58), '固定均匀抽样 0、5、…、60 秒；未按事件选择；不是算法检出或事件时间评分。', font=font(24), fill='#ffd388')
    draw.text((24, 96), '每格左：整帧上下文；右：固定原始 ROI。无增强、无镜像、无算法线；另存原尺寸局部可放大。', font=font(23), fill='#d0dbe5')
    for k, card in enumerate(cards):
        x, y = (k % 2) * cell_w, 160 + (k // 2) * cell_h
        draw.text((x + 15, y), f"源视频 {card['time_s']:.1f} 秒 | 原帧号 {card['source_frame']}（从 0 开始）", font=font(25), fill='white')
        _fit(canvas, card['full'], (x + 10, y + 56, 275, 495))
        _fit(canvas, card['crop'], (x + 300, y + 56, 685, 495))
        draw.text((x + 15, y + 572), '可见材料变化 ≠ 已确认换层；看不清时请选“无法判断”。', font=font(21), fill='#d0dbe5')
    return canvas


def uniform_review(trial=TRIAL, output=None):
    output = Path(output).resolve() if output else OUTPUT / ('uniform_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists():
        raise FileExistsError('uniform review output already exists; never overwrite')
    loaded = {side: load_bound_session(trial, side) for side in LABELS}
    output.mkdir(parents=True)
    manifest = base_manifest()
    manifest['fixed_times_s'] = list(UNIFORM_TIMES_S)
    manifest['coordinate_contract'] = 'Original BGR camera frame converted only to RGB for PNG; no mirror. ROI uses V18 startup seed display rule, fixed for all times.'
    for side, data in loaded.items():
        folder = output / side
        folder.mkdir()
        records = schedule_records(data['index'])
        cap = open_bound_video(data['video'])
        cards, exports = [], []
        try:
            for record in records:
                frame = decode_bound(cap, record, data['video'])
                full, crop = raw_images(frame, data['raw_roi'])
                idx, time_s = record['frame_index'], record['time_s']
                prefix = f'f{idx:06d}_t{time_s:04.1f}'
                preview = ImageOps.contain(full, (1200, 1200), Image.Resampling.LANCZOS)
                images = {
                    'raw_crop_native': save_new_image(crop, folder / (prefix + '_raw_crop.png'), output),
                    'raw_full_preview': save_new_image(preview, folder / (prefix + '_full_preview.png'), output),
                }
                cards.append(dict(source_frame=idx, time_s=time_s, full=preview, crop=crop))
                exports.append(dict(source_frame=idx, time_s=time_s,
                    original_frame_sha256=record['frame_sha256'], hash_and_decode_index_verified=True, images=images))
        finally:
            cap.release()
        page_count = (len(cards) + 3) // 4
        sheets = []
        for start in range(0, len(cards), 4):
            number = start // 4 + 1
            sheets.append(save_new_image(contact_sheet(cards[start:start + 4], side, number, page_count),
                folder / f'contact_{number:02d}.png', output))
        manifest['input_hashes'].update(data['input_hashes'])
        manifest['sessions'][side] = dict(video=data['video'], fixed_raw_crop_xyxy=data['raw_roi'],
            roi=data['roi_info'], frames=exports, contact_sheets=sheets)
    verify_unchanged(manifest['input_hashes'])
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('first', 'uniform'))
    parser.add_argument('--trial', type=Path, default=TRIAL)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = first_previews(args.trial, args.output or OUTPUT) if args.mode == 'first' else uniform_review(args.trial, args.output)
    print(json.dumps(dict(output=str(result)), ensure_ascii=False))


if __name__ == '__main__':
    main()
