"""Bound multi-scale raw image evidence; no rope identity/event inference."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from . import v20_material_review as raw
from .v18_front_diagnostics import font, sha

OUT = Path(__file__).resolve().parent / 'outputs' / 'v23_identity_review'
# Fixed from full first-frame framing, not a prediction or per-frame tracker.
# Half-open original-camera pixels; wide enough to keep free rope and reel context.
ROIS = {'left': (0, 1650, 3050, 2900), 'right': (650, 1400, 4096, 2750)}
WINDOWS = {'left': tuple(range(20, 36)), 'right': tuple(range(8, 21))}
SAVE_NATIVE = {'left': (0, 24, 27, 30), 'right': (0, 12, 15, 18)}


def schedules(side):
    return {'uniform': tuple(range(0, 61, 5)), 'development_near_edge': WINDOWS[side]}


def sheet(cards, side, schedule, number, total):
    canvas = Image.new('RGB', (2000, 150 + 2 * 650), '#151c25')
    d = ImageDraw.Draw(canvas)
    d.text((20, 10), f'V23 活动绳身份原图复核 | {raw.LABELS[side]} | {number}/{total}', font=font(28), fill='white')
    description = ('固定均匀每5秒，完整0–60秒。' if schedule == 'uniform' else
                   '已知近边开发窗口，每1秒；有意选取开发片段，不是盲测或时间预测。')
    d.text((20, 54), description, font=font(24), fill='#ffd388')
    d.text((20, 95), '每格上：完整场景；下：固定宽接触区。仅缩放，无增强/镜像/算法线；两侧为不同录制，不能逐时同步比较。', font=font(21), fill='#d0dbe5')
    for k, card in enumerate(cards):
        x, y = (k % 2) * 1000, 150 + (k // 2) * 650
        d.text((x + 15, y), f"{card['time_s']:.1f}秒 | 原帧{card['source_frame']}（从0开始）", font=font(24), fill='white')
        raw._fit(canvas, card['full'], (x + 12, y + 38, 970, 275))
        raw._fit(canvas, card['crop'], (x + 12, y + 320, 970, 305))
    return canvas


def run():
    output = OUT / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if output.exists(): raise FileExistsError(output)
    loaded = {side: raw.load_bound_session(raw.TRIAL, side) for side in raw.LABELS}
    output.mkdir(parents=True)
    manifest = dict(schema='v23_raw_identity_review_1', script_sha256=sha(__file__),
        helper_sha256=sha(raw.__file__), input_hashes={}, sessions={},
        selection='uniform_full_clip_and_explicit_development_near_edge_windows_separate',
        enhancement=False, mirror=False, model_inference=False, hardware_access=False,
        algorithm_overlays=False, ground_truth_files_opened=False, event_predictions=False,
        identity_accuracy_measured=False, cameras_are_different_recording_sessions=True)
    for side, data in loaded.items():
        folder = output / side
        folder.mkdir()
        schedule = schedules(side)
        times = sorted(set(t for sequence in schedule.values() for t in sequence))
        cap = raw.open_bound_video(data['video'])
        cards, exported = {}, []
        try:
            for record in raw.schedule_records(data['index'], times):
                frame = raw.decode_bound(cap, record, data['video'])
                full, crop = raw.raw_images(frame, ROIS[side])
                idx, time = record['frame_index'], record['time_s']
                card = dict(source_frame=idx, time_s=time,
                    full=ImageOps.contain(full,(900,900),Image.Resampling.LANCZOS),
                    crop=ImageOps.contain(crop,(1600,900),Image.Resampling.LANCZOS))
                images = {}
                if time in SAVE_NATIVE[side]:
                    prefix=f'f{idx:06d}_t{time:.1f}'
                    images['raw_wide_native']=raw.save_new_image(crop, folder/(prefix+'_raw_wide_native.png'),output)
                    images['full_preview']=raw.save_new_image(ImageOps.contain(full,(1200,1200),Image.Resampling.LANCZOS),folder/(prefix+'_full_preview.png'),output)
                cards[time] = card
                exported.append(dict(source_frame=idx,time_s=time,original_frame_sha256=record['frame_sha256'],
                    source_pixels_and_frame_index_verified=True,images=images))
        finally: cap.release()
        figures = {}
        for name, sequence in schedule.items():
            pages = (len(sequence)+3)//4
            figures[name] = [raw.save_new_image(sheet([cards[t] for t in sequence[start:start+4]],side,name,start//4+1,pages),
                folder/f'{name}_{start//4+1:02d}.png',output) for start in range(0,len(sequence),4)]
        manifest['sessions'][side]=dict(video=data['video'],fixed_raw_roi_xyxy_half_open=ROIS[side],
            roi_selection='whole first-frame spatial inspection; fixed wide free-rope plus reel context; not old side-edge-only ROI',
            schedules_s=schedule,frames=exported,contact_sheets=figures)
        manifest['input_hashes'].update(data['input_hashes'])
        print('completed',side,flush=True)
    raw.verify_unchanged(manifest['input_hashes'])
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(output)
    return output


if __name__ == '__main__':
    run()
