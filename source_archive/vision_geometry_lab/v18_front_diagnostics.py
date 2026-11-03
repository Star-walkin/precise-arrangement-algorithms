"""Independent V18 observability figures, never event inference or device control.

Fixed display rules: source-video 0--60 s, warmup 0--5 s hidden, +/-40 native
pixel displacement scale, keyframes every 10 s plus 20--34 s every 2 s. Nothing
is selected from event labels or ground truth. Normal-gradient peak ambiguity
is a diagnostic (<1.2), NOT an event threshold or proof of rope visibility.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE = Path(__file__).resolve().parent / 'outputs' / 'v18_front_observability'
KEY_TIMES = tuple(sorted(set(range(0, 61, 10)) | set(range(20, 35, 2))))
COLOR_LIMIT_PX = 40.
WARMUP_END_S = 5.
AMBIGUITY_RATIO = 1.2
NORMAL_SEARCH_RADIUS_PX = 65.
ROI_MARGIN_PX = 40
BLUE_BGR = (255, 140, 35)
BASELINE_BGR = (180, 235, 255)
MISSING_RGB = (73, 78, 87)
FONT_PATH = Path('C:/Windows/Fonts/msyh.ttc')
ARRAY_SHAPES = {
    'aligned_semantic_x': 'TJ', 'pose_valid': 'T', 'transforms': 'T23',
    'semantic_seed_x_current': 'TJ',
    'baseline_x': 'J', 'baseline_nx': 'J', 'baseline_ny': 'J',
    'normal_offset': 'TJ', 'normal_gradient': 'TJ', 'normal_peak_ratio': 'TJ',
    'normal_clipped_fraction': 'TJ', 'normal_profiles': 'TJ131',
    'baseline_normal_offset': 'J', 'normal_residual': 'TJ',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pixels_sha(frame):
    """V18 raw-BGR digest contract (deliberately not the V16 array digest)."""
    digest = hashlib.sha256(str((frame.shape, str(frame.dtype))).encode('ascii'))
    digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size) if FONT_PATH.is_file() else ImageFont.load_default()


def validate_measurement(arrays, index, protocol_sha256):
    """Verify frame/time/array identity; preserve NaNs, never fill missing data."""
    if index.get('protocol_sha256') != protocol_sha256:
        raise ValueError('measurement protocol hash mismatch')
    times, rows = np.asarray(arrays['times_s']), np.asarray(arrays['rows_y'])
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > 60 + 1e-8):
        raise ValueError('invalid source-video time axis')
    if (rows.ndim != 1 or len(rows) < 2 or not np.isfinite(rows).all()
            or np.any(np.diff(rows) <= 0)):
        raise ValueError('invalid reference row axis')
    t, j = len(times), len(rows)
    shapes = {'TJ': (t, j), 'T': (t,), 'J': (j,), 'T23': (t, 2, 3), 'TJ131': (t, j, 131)}
    for name, shape in ARRAY_SHAPES.items():
        if np.asarray(arrays[name]).shape != shapes[shape]:
            raise ValueError(f'array shape mismatch: {name}')
    pose = np.asarray(arrays['pose_valid'])
    if not np.isin(pose, [False, True]).all():
        raise ValueError('pose_valid must be boolean-valued')
    pose = pose.astype(bool)
    if not np.isfinite(arrays['transforms'][pose]).all():
        raise ValueError('valid pose has nonfinite transform')
    if np.any(np.linalg.det(arrays['transforms'][pose, :, :2]) <= 0):
        raise ValueError('valid pose transform must preserve orientation')
    video = index['video']
    fps, width, height = float(video['fps']), int(video['width']), int(video['height'])
    if not np.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        raise ValueError('invalid source video metadata')
    if video['camera'] not in ('left', 'right'):
        raise ValueError('camera must explicitly identify canonical mirror convention')
    records = index['records']
    if len(records) != t:
        raise ValueError('record count mismatch')
    previous = -1
    for k, record in enumerate(records):
        frame = record['frame_index']
        if (isinstance(frame, bool) or not isinstance(frame, int) or not previous < frame < video['frame_count']
                or abs(float(record['time_s']) - frame / fps) > 1e-8
                or abs(times[k] - frame / fps) > 1e-8):
            raise ValueError('record is not the matching source-video frame/time')
        digest = record['frame_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('missing or malformed original frame hash')
        if bool(record['pose']['valid']) != bool(pose[k]):
            raise ValueError('record/array pose validity mismatch')
        if pose[k] and not np.allclose(record['pose']['reference_to_current'], arrays['transforms'][k], atol=1e-10, rtol=0):
            raise ValueError('record/array pose matrix mismatch')
        previous = frame
    if rows[0] < 0 or rows[-1] >= height:
        raise ValueError('reference rows exceed original image bounds')
    baseline_ok = np.isfinite(arrays['baseline_x']) & np.isfinite(arrays['baseline_nx']) & np.isfinite(arrays['baseline_ny'])
    lengths = np.hypot(arrays['baseline_nx'][baseline_ok], arrays['baseline_ny'][baseline_ok])
    if np.any(np.abs(lengths - 1.) > .05):
        raise ValueError('baseline normals must use unit pixel coordinates')
    return True


def candidate_points(arrays, frame_row, baseline=False):
    offsets = arrays['baseline_normal_offset'] if baseline else arrays['normal_offset'][frame_row]
    return np.column_stack((arrays['baseline_x'] + offsets * arrays['baseline_nx'],
                            arrays['rows_y'] + offsets * arrays['baseline_ny']))


def reference_to_raw(points, reference_to_current, width, camera):
    """Transform in canonical coordinates FIRST, then undo left-camera mirror."""
    points = np.asarray(points, float)
    transform = np.asarray(reference_to_current, float)
    if points.ndim != 2 or points.shape[1:] != (2,) or transform.shape != (2, 3):
        raise ValueError('invalid point or transform dimensions')
    if not np.isfinite(transform).all() or width <= 0 or camera not in ('left', 'right'):
        raise ValueError('invalid coordinate transform')
    result = points @ transform[:, :2].T + transform[:, 2]
    if camera == 'left':
        result[:, 0] = width - 1 - result[:, 0]
    return result


def mirror_roi(canonical_xyxy, width, camera):
    x0, y0, x1, y1 = map(int, canonical_xyxy)
    if camera == 'right':
        return x0, y0, x1, y1
    if camera == 'left':
        # Half-open crop bounds differ by one from point coordinates.
        return width - x1, y0, width - x0, y1
    raise ValueError('unknown camera')


def fixed_roi(arrays, width, height):
    """One reference-coordinate crop from baseline/search radius; never recenter."""
    origin = np.column_stack((arrays['baseline_x'], arrays['rows_y']))
    normals = np.column_stack((arrays['baseline_nx'], arrays['baseline_ny']))
    points = np.concatenate((origin - NORMAL_SEARCH_RADIUS_PX * normals,
                             origin + NORMAL_SEARCH_RADIUS_PX * normals))
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError('cannot derive fixed ROI from baseline')
    low = np.maximum([0, 0], np.floor(points.min(axis=0) - ROI_MARGIN_PX)).astype(int)
    high = np.minimum([width, height], np.ceil(points.max(axis=0) + ROI_MARGIN_PX + 1)).astype(int)
    if np.any(high <= low):
        raise ValueError('baseline crop does not intersect original frame')
    return int(low[0]), int(low[1]), int(high[0]), int(high[1])


def diagnostic_roi(arrays, width, height):
    """Display crop only, based on startup semantic seeds, NEVER a new baseline.

    Use the protocol's full row range even when the actual baseline covers only
    a few rows. Current-canonical startup seeds are inverse-mapped to the fixed
    reference. Neither seeds nor crop bounds enter normal measurements.
    """
    seeds, frame_rows = [], []
    for i in np.flatnonzero((arrays['times_s'] <= WARMUP_END_S) & np.asarray(arrays['pose_valid'], bool)):
        points = np.column_stack((arrays['semantic_seed_x_current'][i], arrays['rows_y']))
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        inverse = cv2.invertAffineTransform(arrays['transforms'][i])
        mapped = points @ inverse[:, :2].T + inverse[:, 2]
        mapped = mapped[np.isfinite(mapped).all(axis=1)]
        if len(mapped):
            seeds.append(mapped)
            frame_rows.append(int(i))
    if seeds:
        points = np.concatenate(seeds)
        x0 = max(0, int(np.floor(points[:, 0].min() - NORMAL_SEARCH_RADIUS_PX - ROI_MARGIN_PX)))
        x1 = min(width, int(np.ceil(points[:, 0].max() + NORMAL_SEARCH_RADIUS_PX + ROI_MARGIN_PX + 1)))
        y0 = max(0, int(np.floor(arrays['rows_y'].min() - ROI_MARGIN_PX)))
        y1 = min(height, int(np.ceil(arrays['rows_y'].max() + ROI_MARGIN_PX + 1)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError('startup semantic seed crop outside original frame')
        return (x0, y0, x1, y1), dict(
            roi_source='startup_semantic_seed_inverse_mapped_reference_for_display_only',
            startup_frame_rows=frame_rows, finite_seed_count=len(points),
            seed_time_range_s=[0, 5], y_extent_source='complete_protocol_rows_y_range',
            creates_measurement_baseline=False, event_or_front_inference=False)
    try:
        crop = fixed_roi(arrays, width, height)
        return crop, dict(roi_source='existing_finite_baseline_for_display_only',
                          creates_measurement_baseline=False, event_or_front_inference=False)
    except ValueError:
        # No usable spatial information: a full-width protocol row band is
        # explicit failure-closed display, not a guessed contour location.
        return (0, max(0, int(arrays['rows_y'][0]) - ROI_MARGIN_PX), width,
                min(height, int(arrays['rows_y'][-1]) + ROI_MARGIN_PX + 1)), dict(
            roi_source='full_width_protocol_row_band_no_spatial_baseline_or_seed',
            creates_measurement_baseline=False, event_or_front_inference=False)


def baseline_status(arrays):
    support = (np.isfinite(arrays['baseline_x']) & np.isfinite(arrays['baseline_nx'])
               & np.isfinite(arrays['baseline_ny']) & np.isfinite(arrays['baseline_normal_offset']))
    count, total = int(support.sum()), len(support)
    if count == 0:
        return count, '启动基线不可用，法向变化未评估'
    if count < total:
        return count, f'启动基线仅 {count}/{total} 行，其他行法向变化未评估'
    return count, f'启动基线支持 {count}/{total} 行（不证明物理边界身份）'


def diagnostic_masks(arrays):
    pose = np.asarray(arrays['pose_valid'], bool)
    normal = np.isfinite(arrays['normal_offset']) & pose[:, None]
    ratio = arrays['normal_peak_ratio']
    ambiguous = normal & np.isfinite(ratio) & (ratio < AMBIGUITY_RATIO)
    undefined = normal & ~np.isfinite(ratio)
    warmup = arrays['times_s'] <= WARMUP_END_S
    gradient = arrays['normal_gradient']
    # Fixed warmup aggregate; no future running window, no smoothing of events.
    baseline_gradient = np.full(len(arrays['rows_y']), np.nan)
    for j in range(len(baseline_gradient)):
        values = gradient[warmup & normal[:, j], j]
        values = values[np.isfinite(values)]
        if len(values):
            baseline_gradient[j] = np.median(values)
    polarity_valid = normal & np.isfinite(gradient) & np.isfinite(baseline_gradient)[None, :]
    polarity_valid &= (gradient != 0) & (baseline_gradient[None, :] != 0)
    polarity_changed = polarity_valid & (np.sign(gradient) != np.sign(baseline_gradient)[None, :])
    return dict(pose=pose, normal=normal, ambiguous=ambiguous, undefined=undefined,
                warmup=warmup, polarity_valid=polarity_valid, polarity_changed=polarity_changed,
                baseline_gradient=baseline_gradient)


def displacement_rgb(values, valid):
    """Fixed blue/white/red +/-40 px map; invalid samples remain opaque gray."""
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool) & np.isfinite(values)
    safe = np.where(valid, values, 0.)
    amount = np.clip(safe / COLOR_LIMIT_PX, -1., 1.)
    white, blue, red = np.array([236, 239, 242]), np.array([45, 125, 225]), np.array([230, 70, 65])
    color = np.where((amount < 0)[..., None], blue, red)
    output = white + np.abs(amount)[..., None] * (color - white)
    output[~valid] = MISSING_RGB
    return np.rint(output).astype(np.uint8)


def _raster_by_time(rgb_tj, times, width, height):
    """Nearest-cell display only; wide gaps are missing, not interpolation."""
    t, j, channels = rgb_tj.shape
    if channels != 3 or t != len(times):
        raise ValueError('raster time dimensions mismatch')
    native = np.full((j, width, 3), MISSING_RGB, np.uint8)
    # A sample owns at most one typical sampling interval. This avoids painting
    # a long missing interval with the last available measurement.
    half = min(.5, float(np.median(np.diff(times))) / 2)
    for k, time_s in enumerate(times):
        x0 = max(0, int(np.floor((time_s - half) / 60 * width)))
        x1 = min(width, int(np.ceil((time_s + half) / 60 * width)))
        if x1 > x0:
            native[:, x0:x1] = rgb_tj[k, :, None, :]
    return Image.fromarray(native).resize((width, height), Image.Resampling.NEAREST)


def spacetime_figure(arrays, session):
    masks = diagnostic_masks(arrays)
    times, rows = arrays['times_s'], arrays['rows_y']
    semantic = arrays['aligned_semantic_x'] - arrays['baseline_x'][None, :]
    semantic_rgb = displacement_rgb(semantic, masks['pose'][:, None] & np.isfinite(semantic))
    residual_rgb = displacement_rgb(arrays['normal_residual'], masks['normal'])
    valid_rgb = np.full((*masks['normal'].shape, 3), MISSING_RGB, np.uint8)
    valid_rgb[masks['normal']] = (70, 185, 110)
    peak_rgb = np.full_like(valid_rgb, MISSING_RGB)
    peak_rgb[masks['normal'] & ~masks['undefined'] & ~masks['ambiguous']] = (60, 140, 210)
    peak_rgb[masks['ambiguous']] = (245, 180, 55)
    peak_rgb[masks['undefined']] = (185, 100, 200)
    polarity_rgb = np.full_like(valid_rgb, MISSING_RGB)
    polarity_rgb[masks['polarity_valid']] = (65, 150, 170)
    polarity_rgb[masks['polarity_changed']] = (245, 130, 55)
    canvas = Image.new('RGB', (2100, 1590), '#111820')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), f'V18 {session} · 完整 0–60s 可观测性诊断（不是自动事件检测）', font=font(28), fill='white')
    draw.text((20, 54), '固定参考：0–5s；仅 t>5s 可比较。像素 / 图像轴向投影，不是毫米、根数、实际转角或施力节点。', font=font(21), fill='#ffd580')
    draw.text((20, 87), '色标固定：蓝 -40px / 白 0 / 红 +40px，超范围截色；缺失灰色。没有将弱区域透明化或填补缺失。', font=font(20), fill='#ced8e2')
    draw.text((20, 119), baseline_status(arrays)[1] + '；灰色缺失不是零变化。', font=font(22), fill='#ffb76b')
    x, width = 150, 1860
    panels = [
        ('语义直接轮廓相对固定基线的横向偏移（仅诊断）', semantic_rgb, 240),
        ('固定法向候选残差 normal_offset − baseline_normal_offset（主候选；非实测绳身份）', residual_rgb, 240),
        ('法向数值可用：绿色 = 有限 offset 且背景配准通过；灰色 = 不可用', valid_rgb, 135),
        ('峰歧义：黄 <1.2；蓝 ≥1.2；紫 无次峰 / 比值未定义；灰 不可用（不证明可见性）', peak_rgb, 135),
        ('有符号法向梯度相对基线：橙色极性不同；青色相同；灰缺失 / 零梯度（不筛除极性）', polarity_rgb, 135),
    ]
    y = 162
    for title, raster, height in panels:
        # Exclude the whole warmup sample cell, including its half-cell beyond
        # exactly 5.0 s; the cover alone must not leak a startup measurement.
        raster[masks['warmup']] = MISSING_RGB
        draw.text((20, y), title, font=font(22), fill='white')
        y += 36
        canvas.paste(_raster_by_time(raster, times, width, height), (x, y))
        # Warmup is intentionally masked even if a numeric value exists.
        warmup_right = x + round(width * WARMUP_END_S / 60)
        draw.rectangle((x, y, warmup_right, y + height - 1), fill='#222a33')
        for hy in range(y, y + height, 18):
            draw.line((x, hy, warmup_right, min(y + height - 1, hy + 35)), fill='#46515d', width=1)
        draw.text((x + 8, y + 8), '0–5s\n基线准备', font=font(16), fill='#ffd580')
        draw.text((20, y), f'y={rows[0]:.0f}px', font=font(17), fill='#ced8e2')
        draw.text((20, y + height - 23), f'y={rows[-1]:.0f}px', font=font(17), fill='#ced8e2')
        for ts in range(0, 61, 10):
            tx = x + width * ts / 60
            draw.line((tx, y, tx, y + height), fill='#75808c', width=1)
            draw.text((tx - 12, y + height + 3), str(ts) + 's', font=font(16), fill='#ced8e2')
        y += height + 38
    draw.text((20, y), '背景配准', font=font(20), fill='white')
    pose_rgb = np.full((len(times), 1, 3), MISSING_RGB, np.uint8)
    pose_rgb[masks['pose'], 0] = (70, 185, 110)
    canvas.paste(_raster_by_time(pose_rgb, times, width, 25), (x, y))
    draw.text((20, y + 39), '低峰歧义、稳定背景或清晰轮廓均不等于活动绳可见、绳身份确认或进入新层。未导入人工节点真值。', font=font(20), fill='#ffd580')
    return canvas


def _dashed(canvas, points):
    # Keep missing-row gaps: only neighboring finite vertices are connected.
    distance_along = 0.
    for first, second in zip(points[:-1], points[1:]):
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            distance_along = 0.
            continue
        delta = second - first
        length = float(np.linalg.norm(delta))
        if length <= 0:
            continue
        for start in np.arange(-(distance_along % 18.), length, 18.):
            visible_start, visible_end = max(0., start), min(start + 10., length)
            if visible_end <= visible_start:
                continue
            a = first + delta * visible_start / length
            b = first + delta * visible_end / length
            cv2.line(canvas, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), BASELINE_BGR, 3, cv2.LINE_AA)
        distance_along += length


def overlay_raw(frame, arrays, frame_row, camera):
    result = frame.copy()
    if not bool(arrays['pose_valid'][frame_row]):
        return result
    transform = arrays['transforms'][frame_row]
    width = frame.shape[1]
    base = reference_to_raw(candidate_points(arrays, frame_row, baseline=True), transform, width, camera)
    current = reference_to_raw(candidate_points(arrays, frame_row), transform, width, camera)
    _dashed(result, base)
    for point in current[np.isfinite(current).all(axis=1)]:
        cv2.circle(result, tuple(np.rint(point).astype(int)), 4, BLUE_BGR, -1, cv2.LINE_AA)
    return result


def validate_original_frame(frame, actual_index, record, video):
    if actual_index != record['frame_index']:
        raise ValueError('actual decoded source index mismatch')
    if frame.shape != (video['height'], video['width'], 3):
        raise ValueError('decoded original dimensions mismatch')
    if pixels_sha(frame) != record['frame_sha256']:
        raise ValueError('decoded original frame hash mismatch')


def _paste_fit(canvas, picture, box):
    x, y, w, h = box
    image = ImageOps.contain(picture, (w, h), Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (w - image.width) // 2, y + (h - image.height) // 2))


def frame_strip(cards, session, baseline_info=None):
    cell_width, cell_height = 1000, 430
    canvas = Image.new('RGB', (2 * cell_width, 110 + int(np.ceil(len(cards) / 2)) * cell_height), '#111820')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), f'V18 {session} · 固定时刻原始侧轮廓小窗（固定原图 ROI，不逐帧居中）', font=font(26), fill='white')
    draw.text((20, 48), '每格左：原图；右：蓝点法向候选 / 浅黄虚线固定基线。左相机已翻回原图；不是绳身份或自动节点。', font=font(20), fill='#ffd580')
    if baseline_info is not None:
        draw.text((20, 79), baseline_info[1] + '；显示 ROI 来自启动语义种子，不是测量基线。', font=font(18), fill='#ffb76b')
    for i, card in enumerate(cards):
        x, y = (i % 2) * cell_width, 110 + (i // 2) * cell_height
        draw.text((x + 12, y), f"AVI {card['time_s']:.1f}s · 源帧 {card.get('source_frame', '缺失')} · pose={card.get('pose_valid', False)}", font=font(23), fill='white')
        if 'raw' not in card:
            draw.text((x + 12, y + 70), '缺少对应采样记录：不替换为邻近帧，不伪造叠加', font=font(20), fill='#ffd580')
            continue
        for j, key in enumerate(('raw', 'overlay')):
            _paste_fit(canvas, card[key], (x + 8 + j * 490, y + 43, 478, 350))
        note = '0–5s：基线准备，不解释残差' if card['time_s'] <= 5 else '仅候选位置 / 固定基线比较，未判断进入新层'
        if baseline_info is not None and baseline_info[0] == 0:
            note = '启动基线不可用；仅保留原图，未输出法向候选'
        draw.text((x + 12, y + 394), note, font=font(18), fill='#ced8e2')
    return canvas


def profile_strip(arrays, session):
    """Fixed gray 0..255 profile diagnostics; absent profiles remain gray."""
    cell_w, cell_h = 500, 330
    canvas = Image.new('RGB', (4 * cell_w, 130 + int(np.ceil(len(KEY_TIMES) / 4)) * cell_h), '#111820')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), f'V18 {session} · 已存储法向灰度 profiles（不重新采样、不制造基线）', font=font(26), fill='white')
    draw.text((20, 50), baseline_status(arrays)[1], font=font(22), fill='#ffb76b')
    draw.text((20, 86), '每格横向：固定法向 -65..+65px；纵向：协议 rows_y；灰度固定 0..255，缺失蓝灰色。', font=font(20), fill='#ced8e2')
    for i, time_s in enumerate(KEY_TIMES):
        x, y = (i % 4) * cell_w, 130 + (i // 4) * cell_h
        matches = np.flatnonzero(np.abs(arrays['times_s'] - time_s) <= 1e-8)
        draw.text((x + 10, y), f'AVI {time_s:.1f}s', font=font(21), fill='white')
        if not len(matches):
            draw.text((x + 10, y + 70), '缺少此时刻采样，不代入邻帧', font=font(18), fill='#ffd580')
            continue
        profile = arrays['normal_profiles'][matches[0]]
        finite = np.isfinite(profile)
        picture = np.repeat(np.rint(np.clip(np.where(finite, profile, 0.), 0, 255)).astype(np.uint8)[..., None], 3, axis=2)
        picture[~finite] = MISSING_RGB
        canvas.paste(Image.fromarray(picture).resize((cell_w - 24, 240), Image.Resampling.NEAREST), (x + 10, y + 35))
        message = '没有可用 profiles：未评估，不等于图像无纹理' if not finite.any() else f'有限采样 {int(finite.sum())}/{finite.size}，不证明材料身份'
        draw.text((x + 10, y + 284), message, font=font(17), fill='#ffd580')
    return canvas


def run(trial=BASE, output=None, sessions=('left', 'right')):
    trial = Path(trial).resolve()
    protocol_path = trial / 'protocol.json'
    protocol_hash = sha(protocol_path)
    output = Path(output).resolve() if output else trial / 'diagnostics' / ('fixed_rules_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists():
        raise FileExistsError('diagnostics output exists; never overwrite')
    loaded, input_hashes = {}, {str(protocol_path): protocol_hash}
    for session in sessions:
        npz_path, index_path = trial / session / 'measurement.npz', trial / session / 'index.json'
        index = json.loads(index_path.read_text(encoding='utf-8'))
        if index.get('output_status') != 'complete_observations_only_no_events':
            raise ValueError('measurement is not a completed V18 observability run')
        measurement_hash = sha(npz_path)
        if index.get('measurement_sha256') != measurement_hash:
            raise ValueError('measurement archive hash mismatch')
        with np.load(npz_path, allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in ('times_s', 'rows_y', *ARRAY_SHAPES)}
        validate_measurement(arrays, index, protocol_hash)
        if index['video']['camera'] != session:
            raise ValueError('session/camera mismatch')
        loaded[session] = (arrays, index)
        input_hashes[str(npz_path)] = measurement_hash
        input_hashes[str(index_path)] = sha(index_path)
    output.mkdir(parents=True)
    manifest = dict(schema='v18_fixed_rule_observability_diagnostics_1', protocol_sha256=protocol_hash,
        script_sha256=sha(__file__), input_hashes=input_hashes, sessions={},
        fixed_rules=dict(time_range_s=[0, 60], baseline_s=[0, 5], warmup_interpretable=False,
            key_times_s=KEY_TIMES, color_limits_px=[-40, 40], peak_ambiguity_ratio=1.2,
            peak_ratio_is_not_visibility_probability=True, roi_search_radius_px=65, roi_margin_px=ROI_MARGIN_PX),
        coordinate_contract='baseline/reference canonical native pixels; apply reference_to_current then left-camera x_raw=W-1-x_can',
        quantities='native pixels / image axial projection; NOT mm, rope counts or physical turn phase',
        ground_truth_files_opened=False, hardware_access=False, event_output=False, main_ui_modified=False)
    for session, (arrays, index) in loaded.items():
        folder = output / session
        folder.mkdir()
        video = index['video']
        video_hash = sha(video['path'])
        if video.get('sha256') and video['sha256'] != video_hash:
            raise ValueError('source video hash mismatch')
        chart_path = folder / 'spacetime.png'
        spacetime_figure(arrays, session).save(chart_path)
        canonical_roi, roi_info = diagnostic_roi(arrays, video['width'], video['height'])
        raw_roi = mirror_roi(canonical_roi, video['width'], session)
        records_by_frame = {record['frame_index']: (i, record) for i, record in enumerate(index['records'])}
        cap = cv2.VideoCapture(video['path'])
        if not cap.isOpened():
            raise ValueError('cannot open source video')
        if (abs(cap.get(cv2.CAP_PROP_FPS) - video['fps']) > 1e-6
                or int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) != video['frame_count']):
            cap.release()
            raise ValueError('source video FPS/frame count mismatch')
        cards, frames = [], []
        try:
            for time_s in KEY_TIMES:
                expected = round(time_s * video['fps'])
                if expected not in records_by_frame:
                    cards.append(dict(time_s=time_s))
                    frames.append(dict(reference_time_s=time_s, source_frame=expected, available=False, reason='missing_sample_record'))
                    continue
                row, record = records_by_frame[expected]
                if abs(record['time_s'] - time_s) > 1e-8:
                    raise ValueError('fixed requested time is not an exact source frame')
                cap.set(cv2.CAP_PROP_POS_FRAMES, expected)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('source decoding failed')
                actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                validate_original_frame(frame, actual, record, video)
                canonical = cv2.flip(frame, 1) if session == 'left' else frame.copy()
                overlay = overlay_raw(frame, arrays, row, session)
                raw_picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).crop(raw_roi)
                canonical_picture = Image.fromarray(cv2.cvtColor(canonical, cv2.COLOR_BGR2RGB)).crop(canonical_roi)
                overlay_picture = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).crop(raw_roi)
                prefix = f'f{actual:06d}_t{time_s:04.1f}'
                image_info = {}
                for suffix, picture in (('raw_crop', raw_picture), ('canonical_crop', canonical_picture), ('raw_overlay', overlay_picture)):
                    path = folder / (prefix + '_' + suffix + '.png')
                    picture.save(path)
                    image_info[suffix] = dict(path=str(path.relative_to(output)), sha256=sha(path))
                cards.append(dict(time_s=time_s, source_frame=actual, pose_valid=bool(arrays['pose_valid'][row]), raw=raw_picture, overlay=overlay_picture))
                frames.append(dict(reference_time_s=time_s, source_frame=actual, available=True,
                    original_frame_sha256=record['frame_sha256'], hash_and_actual_decode_index_verified=True,
                    pose_valid=bool(arrays['pose_valid'][row]), images=image_info))
        finally:
            cap.release()
        strip_path = folder / 'fixed_keyframe_strip.png'
        frame_strip(cards, session, baseline_status(arrays)).save(strip_path)
        profiles_path = folder / 'stored_normal_profiles.png'
        profile_strip(arrays, session).save(profiles_path)
        masks = diagnostic_masks(arrays)
        manifest['sessions'][session] = dict(video=video, actual_video_sha256=video_hash,
            fixed_canonical_crop_xyxy=canonical_roi, fixed_raw_crop_xyxy=raw_roi, frames=frames,
            roi=roi_info, baseline_supported_rows=baseline_status(arrays)[0], baseline_display_status=baseline_status(arrays)[1],
            figures={key: dict(path=str(path.relative_to(output)), sha256=sha(path))
                     for key, path in (('spacetime', chart_path), ('keyframe_strip', strip_path), ('stored_profiles', profiles_path))},
            normal_finite_pose_valid_cells=int(masks['normal'].sum()), peak_ambiguous_cells=int(masks['ambiguous'].sum()),
            peak_ratio_undefined_cells=int(masks['undefined'].sum()),
            baseline_gradient_source='median finite signed normal_gradient during fixed 0..5s, pose-valid samples only')
    for path, digest in input_hashes.items():
        if sha(path) != digest:
            raise ValueError('input changed during diagnostics generation')
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(output), 'manifest': str(output / 'manifest.json')}, ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, default=BASE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--session', choices=('left', 'right'), action='append')
    args = parser.parse_args()
    run(args.trial, args.output, tuple(args.session) if args.session else ('left', 'right'))


if __name__ == '__main__':
    main()
