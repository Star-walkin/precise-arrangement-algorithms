"""Four full raw frames for intentional, post-hoc spatial bulge localization.

No detector, model, crop, enhancement, mirror, event prediction or device access.
The quoted statement and contextual reference time are recorded separately;
neither changes an existing truth file or constitutes a new truth annotation.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import v20_material_review as bound_raw
from .v18_front_diagnostics import font, sha

TIMES_S = (26.8, 27.3, 27.8, 28.3)
USER_QUOTE = '我主要还是通过肉眼判断新绳爬到旧层上，即我的肉眼判断线相对于旧的一层有明显的凸起。'
REFERENCE_CONTEXT = dict(
    source='prior_question_context_relayed_by_parent_not_part_of_verbatim_quote',
    camera='left', approximate_reference_time_s=27.3,
    interpretation='User reference concerns new rope protruding relative to the previous layer, not an algorithm-detected event.')
ROOT = Path(__file__).resolve().parent / 'outputs' / 'v21_bulge_review'


def full_raw_picture(frame):
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('native uint8 BGR source frame required')
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def contact_sheet(cards):
    if len(cards) != 4 or tuple(card['time_s'] for card in cards) != TIMES_S:
        raise ValueError('exact four-frame review schedule required')
    canvas = Image.new('RGB', (2000, 1750), '#151c25')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 12), 'V21 凸起位置复核 | 左相机原始全图 | 拍摄右侧法兰', font=font(30), fill='white')
    draw.text((24, 61), '围绕人工参考约 27.3 秒有意选择：仅请圈出新绳相对旧层凸起的位置，不是盲测或事件预测。', font=font(25), fill='#ffd388')
    draw.text((24, 105), '无裁剪、增强、镜像或算法线。每帧另存原分辨率全图，可放大；原帧号从 0 开始。', font=font(25), fill='#d0dbe5')
    for k, card in enumerate(cards):
        x, y = (k % 2) * 1000, 167 + (k // 2) * 785
        draw.text((x + 22, y), f"源视频 {card['time_s']:.1f} 秒 | 原帧 {card['source_frame']}", font=font(28), fill='white')
        bound_raw._fit(canvas, card['preview'], (x + 18, y + 53, 964, 715))
    return canvas


def export(trial=bound_raw.TRIAL, output=None):
    output = Path(output).resolve() if output else ROOT / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if output.exists():
        raise FileExistsError('fresh V21 review directory required; do not overwrite earlier artifacts')
    source = bound_raw.load_bound_session(trial, 'left')
    records = bound_raw.schedule_records(source['index'], TIMES_S)
    output.mkdir(parents=True)
    cap = bound_raw.open_bound_video(source['video'])
    cards, exports = [], []
    try:
        for record in records:
            frame = bound_raw.decode_bound(cap, record, source['video'])
            full = full_raw_picture(frame)
            preview = ImageOps.contain(full, (1200, 1200), Image.Resampling.LANCZOS)
            idx, time_s = record['frame_index'], record['time_s']
            prefix = f'f{idx:06d}_t{time_s:.1f}'
            images = dict(
                full_native=bound_raw.save_new_image(full, output / f'{prefix}_raw_full.png', output),
                preview=bound_raw.save_new_image(preview, output / f'{prefix}_preview.png', output))
            cards.append(dict(source_frame=idx, time_s=time_s, preview=preview))
            exports.append(dict(source_frame=idx, time_s=time_s,
                original_frame_sha256=record['frame_sha256'],
                hash_and_actual_decode_index_verified=True, images=images))
    finally:
        cap.release()
    sheet = bound_raw.save_new_image(contact_sheet(cards), output / 'left_four_raw_full_frames.png', output)
    bound_raw.verify_unchanged(source['input_hashes'])
    manifest = dict(
        schema='v21_posthoc_bulge_spatial_review_1',
        selection='intentional_four_times_around_prior_human_reference_not_blind_not_predicted',
        requested_times_s=list(TIMES_S), purpose='User localization of directly perceived new-rope protrusion relative to old layer.',
        user_statement=dict(verbatim_text=USER_QUOTE, quotation_source='latest_user_message_relayed_verbatim_by_parent'),
        reference_context=REFERENCE_CONTEXT,
        video=source['video'], protocol_sha256=source['protocol_sha256'],
        input_hashes=source['input_hashes'], script_sha256=sha(__file__),
        reused_reader_script_sha256=sha(bound_raw.__file__), frames=exports, contact_sheet=sheet,
        full_image_dimensions=[source['video']['width'], source['video']['height']],
        coordinate_contract='Original camera raw orientation; BGR to RGB channel conversion only for native PNG. Preview and contact sheet resized only.',
        old_side_roi_used=False, cropped=False, enhanced=False, mirrored=False,
        algorithm_overlays=False, ground_truth_files_opened=False,
        existing_ground_truth_changed=False, event_prediction=False,
        physical_layer_time_confirmed=False, model_inference=False,
        hardware_access=False, main_ui_changed=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), contact_sheet=str(output / 'left_four_raw_full_frames.png')), ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, default=bound_raw.TRIAL)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    export(args.trial, args.output)


if __name__ == '__main__':
    main()
