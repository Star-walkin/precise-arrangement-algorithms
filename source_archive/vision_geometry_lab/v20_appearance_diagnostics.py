"""Read-only raw-image diagnostics for frozen V20 appearance controls.

Accepted means two flange descriptors passed their fixed rules. It does not
confirm material identity, rotational phase, a revolution, or any layer event.
Pair selection uses accepted flag and source time only; no geometry or truth.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import v20_appearance_control as control
from . import v20_material_review as raw_review
from .v18_front_diagnostics import font, sha

HISTORY_LABELS = {'local_2_to_12s': '局部历史 2–12 秒',
                  'fixed_startup_5_to_10s': '固定启动历史 5–10 秒'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bucket_records(records, width_s):
    if width_s not in (5, 10):
        raise ValueError('fixed diagnostic buckets must be 5 or 10 seconds')
    times = np.asarray([r['time_s'] for r in records], float)
    if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                       or times[0] < 0 or times[-1] > 60):
        raise ValueError('invalid complete source timeline')
    result = []
    for start in range(0, 60, width_s):
        end = start + width_s
        rows = [r for r in records if start <= r['time_s'] and
                (r['time_s'] <= end if end == 60 else r['time_s'] < end)]
        result.append(dict(start_s=start, end_s=end, end_inclusive=end == 60, records=rows))
    return result


def summarize_and_select(records, history):
    """Never rank by residuals, outcome or closeness to human event labels."""
    h = control.HISTORIES.index(history)
    summaries, selected = [], []
    for bucket in bucket_records(records, 5):
        accepted = sum(r['histories'][h]['match']['accepted'] is True for r in bucket['records'])
        summaries.append({k: v for k, v in bucket.items() if k != 'records'} |
                         dict(frames=len(bucket['records']), accepted=accepted,
                              not_accepted=len(bucket['records']) - accepted))
    for bucket in bucket_records(records, 10):
        matches = [r for r in bucket['records'] if r['histories'][h]['match']['accepted'] is True]
        selected.append({k: v for k, v in bucket.items() if k != 'records'} |
                        dict(accepted_count=len(matches), selected_record_index=matches[0]['index'] if matches else None,
                             status='accepted_appearance_only' if matches else 'no_accepted_pair'))
    return summaries, selected


def validate_timeline(report, extract, source_index, descriptor_times, descriptor_frames,
                      comparison_times, comparison_frames):
    source = source_index['records']
    records = report['records']
    if len(records) != len(source) or len(extract['records']) != len(source):
        raise ValueError('incomplete comparison/source timeline')
    times = np.asarray([r['time_s'] for r in source])
    frames = np.asarray([r['frame_index'] for r in source])
    for candidate, expected in ((descriptor_times, times), (descriptor_frames, frames),
                                (comparison_times, times), (comparison_frames, frames)):
        if not np.array_equal(candidate, expected):
            raise ValueError('array/source timeline mismatch')
    for i, (r, e, original) in enumerate(zip(records, extract['records'], source)):
        if r['index'] != i or any(r[k] != original[k] or e[k] != original[k]
                                 for k in ('frame_index', 'time_s', 'frame_sha256')):
            raise ValueError('comparison/extraction source provenance mismatch')
        if len(r['histories']) != len(control.HISTORIES):
            raise ValueError('missing history record')
        for history, item in zip(control.HISTORIES, r['histories']):
            match = item['match']
            if (match['history'] != history or match['current_index'] != i
                    or match['time_s'] != original['time_s'] or type(match['accepted']) is not bool):
                raise ValueError('match/source identity mismatch')
            for key in ('best_index', 'runner_index', 'control_index'):
                past = match[key]
                if past is not None and (isinstance(past, bool) or not isinstance(past, int) or not 0 <= past < i):
                    raise ValueError('match partner is not a strictly past source record')
            if match['accepted']:
                if match['best_index'] is None:
                    raise ValueError('accepted match without best partner')
                scores = np.asarray(match['best_scores'], float)
                if scores.shape != (2,) or not np.isfinite(scores).all() or not np.isfinite(match['runner_margin']):
                    raise ValueError('accepted match without finite independent scores')


def load_bound(base):
    base = Path(base).resolve()
    protocol = control.verify(base)
    phash = sha(base / 'protocol.json')
    loaded, hashes = {}, {str(base / 'protocol.json'): phash}
    for side in ('left', 'right'):
        paths = dict(extract=base / side / 'extract/index.json', descriptors=base / side / 'extract/descriptors.npz',
                     report=base / side / 'compare/report.json', comparisons=base / side / 'compare/comparisons.npz')
        digests = {key: sha(path) for key, path in paths.items()}
        e, r = _read(paths['extract']), _read(paths['report'])
        if (e.get('status') != 'complete' or r.get('status') != 'complete'
                or e['session'] != side or r['session'] != side
                or e['protocol_sha256'] != phash or r['protocol_sha256'] != phash
                or e['descriptor_sha256'] != digests['descriptors']
                or r['descriptor_sha256'] != digests['descriptors']
                or r['descriptor_index_sha256'] != digests['extract']
                or r['comparison_sha256'] != digests['comparisons']):
            raise ValueError('incomplete or unbound appearance outputs')
        for key in ('mechanical_phase_confirmed', 'physical_new_layer_confirmed', 'material_identity_confirmed'):
            if r.get(key) is not False:
                raise ValueError('diagnostic inputs must not claim physical confirmation')
        data = raw_review.load_bound_session(protocol['upstream_base'], side)
        if e['video'] != data['video'] or r['video'] != data['video'] or protocol['videos'][side] != data['video']:
            raise ValueError('appearance/raw source video mismatch')
        with np.load(paths['descriptors'], allow_pickle=False) as d, np.load(paths['comparisons'], allow_pickle=False) as c:
            validate_timeline(r, e, data['index'], d['times_s'], d['frame_indices'], c['times_s'], c['frame_indices'])
        hashes.update({str(paths[key]): digest for key, digest in digests.items()})
        hashes.update(data['input_hashes'])
        loaded[side] = dict(raw=data, report=r)
    return loaded, hashes, protocol


def summary_figure(side, history, buckets, selections):
    canvas = Image.new('RGB', (1480, 1110), '#151c25')
    d = ImageDraw.Draw(canvas)
    d.text((25, 14), f'V20 外观筛选对照 | {raw_review.LABELS[side]} | {HISTORY_LABELS[history]}', font=font(28), fill='white')
    d.text((25, 62), 'accepted 仅表示两块法兰外观通过固定规则，不表示同圈相位、材料身份或换层。', font=font(23), fill='#ffd388')
    d.text((25, 108), '固定 5 秒桶                  总帧数        accepted       未通过/不可用', font=font(25), fill='white')
    for i, row in enumerate(buckets):
        y = 152 + i * 42
        label = f"[{row['start_s']}, {row['end_s']}{']' if row['end_inclusive'] else ')'} 秒"
        d.text((30, y), label, font=font(24), fill='#d0dbe5')
        for x, value in zip((420, 620, 860), (row['frames'], row['accepted'], row['not_accepted'])):
            d.text((x, y), str(value), font=font(24), fill='#78d9db' if x == 620 else 'white')
        if row['accepted'] == 0:
            d.text((1110, y), '无可接受配对', font=font(22), fill='#c4c4c4')
    d.text((25, 690), '原图配对选择：每 10 秒桶第一条 accepted；不按几何变化或事件时间挑选。', font=font(23), fill='#ffd388')
    for i, row in enumerate(selections):
        label = f"{row['start_s']}–{row['end_s']} 秒："
        value = f"选记录 {row['selected_record_index']}（本桶 {row['accepted_count']} 条）" if row['selected_record_index'] is not None else '无可接受配对；未用拒绝结果补位'
        d.text((30, 733 + i * 48), label + value, font=font(23), fill='white')
    return canvas


def _number(value):
    return '不可用' if value is None else f'{float(value):.3f}'


def pair_figure(side, history, record, pictures):
    match = record['histories'][control.HISTORIES.index(history)]['match']
    canvas = Image.new('RGB', (2100, 1160), '#151c25')
    d = ImageDraw.Draw(canvas)
    d.text((22, 10), f"V20 原始外观配对 | {raw_review.LABELS[side]} | {HISTORY_LABELS[history]}", font=font(29), fill='white')
    d.text((22, 58), f"首条 accepted 原图；最佳两 ROI 相似度 {_number(match['best_scores'][0])} / {_number(match['best_scores'][1])}；margin {_number(match['runner_margin'])}", font=font(24), fill='#ffd388')
    d.text((22, 100), '仅外观对照，不是同圈相位；无增强/镜像/算法线。时间中位对照并不保证外观不相似。', font=font(23), fill='#d0dbe5')
    names = [('current', '当前帧'), ('appearance_best', '外观最佳历史帧'), ('temporal_midpoint_control', '时间中位历史对照')]
    for col, (role, label) in enumerate(names):
        x = col * 700
        d.text((x + 20, 151), label, font=font(27), fill='white')
        picture = pictures.get(role)
        if picture is None:
            d.text((x + 20, 220), '无该历史对照，不替换', font=font(24), fill='#ffd388')
            continue
        delta = record['time_s'] - picture['time_s']
        d.text((x + 20, 191), f"{picture['time_s']:.1f} 秒 | 原帧 {picture['source_frame']} | 延迟 {delta:.1f} 秒", font=font(22), fill='white')
        scores = None if role == 'current' else match['best_scores' if role == 'appearance_best' else 'control_scores']
        text = '源视频真实帧；整帧上下文 + 原始局部' if scores is None else f'ROI 相似度 {_number(scores[0])} / {_number(scores[1])}'
        d.text((x + 20, 226), text, font=font(21), fill='#d0dbe5')
        raw_review._fit(canvas, picture['full'], (x + 15, 267, 670, 250))
        raw_review._fit(canvas, picture['crop'], (x + 15, 545, 670, 572))
    return canvas


def run(base=control.BASE, output=None):
    base = Path(base).resolve()
    output = Path(output).resolve() if output else base.parent / 'appearance_diagnostics' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if output.exists():
        raise FileExistsError('fresh diagnostics output required; never overwrite')
    loaded, hashes, protocol = load_bound(base)
    output.mkdir(parents=True)
    manifest = dict(schema='v20_appearance_diagnostics_1', source_base=str(base), input_hashes=hashes,
        script_sha256=sha(__file__), helper_sha256=sha(raw_review.__file__),
        protocol_sha256=sha(base / 'protocol.json'),
        selection='each_history_first_accepted_in_each_fixed_10s_bucket_time_only_no_geometry_or_truth',
        last_bucket_end_inclusive=True, summary_bucket_seconds=5, sessions={},
        ground_truth_files_opened=False, event_output=False, hardware_access=False,
        mechanical_phase_confirmed=False, material_identity_confirmed=False,
        enhancement=False, mirrored=False, algorithm_overlays=False)
    for side, data in loaded.items():
        raw = data['raw']; records = data['report']['records']
        folder = output / side
        folder.mkdir()
        cap = None
        histories = {}
        try:
            for history in control.HISTORIES:
                hfolder = folder / history
                hfolder.mkdir()
                buckets, selections = summarize_and_select(records, history)
                overview = raw_review.save_new_image(summary_figure(side, history, buckets, selections), hfolder / 'accepted_summary.png', output)
                pairs = []
                for selection in selections:
                    chosen = selection['selected_record_index']
                    if chosen is None:
                        pairs.append(dict(**selection, figure=None, source_images={}))
                        continue
                    record = records[chosen]
                    match = record['histories'][control.HISTORIES.index(history)]['match']
                    if cap is None:
                        cap = raw_review.open_bound_video(raw['video'])
                    pictures, exports = {}, {}
                    for role, row in (('current', chosen), ('appearance_best', match['best_index']),
                                      ('temporal_midpoint_control', match['control_index'])):
                        if row is None:
                            exports[role] = None
                            continue
                        source = raw['index']['records'][row]
                        frame = raw_review.decode_bound(cap, source, raw['video'])
                        full, crop = raw_review.raw_images(frame, raw['raw_roi'])
                        preview = ImageOps.contain(full, (1200, 1200), Image.Resampling.LANCZOS)
                        name = f"bucket{selection['start_s']:02d}_{role}_f{source['frame_index']:06d}"
                        images = {'raw_crop_native': raw_review.save_new_image(crop, hfolder / (name + '_raw_crop.png'), output),
                                  'full_preview': raw_review.save_new_image(preview, hfolder / (name + '_full.png'), output)}
                        pictures[role] = dict(full=preview, crop=crop, source_frame=source['frame_index'], time_s=source['time_s'])
                        exports[role] = dict(source_frame=source['frame_index'], time_s=source['time_s'],
                            original_frame_sha256=source['frame_sha256'], hash_and_actual_decode_index_verified=True, images=images)
                    artifact = raw_review.save_new_image(pair_figure(side, history, record, pictures),
                        hfolder / f"bucket{selection['start_s']:02d}_raw_pair.png", output)
                    pairs.append(dict(**selection, match=match, figure=artifact, source_images=exports))
                histories[history] = dict(accepted_total=sum(x['accepted'] for x in buckets),
                    five_second_buckets=buckets, summary_figure=overview, ten_second_selections=pairs)
        finally:
            if cap is not None:
                cap.release()
        manifest['sessions'][side] = dict(video=raw['video'], fixed_raw_crop_xyxy=raw['raw_roi'],
            roi=raw['roi_info'], histories=histories)
    control.verify(base)
    raw_review.verify_unchanged(hashes)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), accepted={side: {h: data['accepted_total'] for h, data in value['histories'].items()}
                      for side, value in manifest['sessions'].items()}), ensure_ascii=False))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=control.BASE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.base, args.output)


if __name__ == '__main__':
    main()
