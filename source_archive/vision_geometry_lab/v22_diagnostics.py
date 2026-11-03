"""Independent full-timeline V22 diagnostics and unenhanced raw crop replay.

No new edge extraction, event labels, material identity, mechanical angles or UI.
The continuity audit separates image-pair continuity from contrast qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import v22_local_edges as local
from . import v22_pair_continuity_audit as continuity
from . import v22_flange_tracks as tracks
from . import v20_material_review as raw_reader
from .v18_front_diagnostics import font, sha, validate_original_frame

OUT = Path(__file__).resolve().parent / 'outputs' / 'v22_diagnostics'
UNKNOWN = (87, 88, 93)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_axis(records, source, index_key='frame_index', hash_key=None):
    if len(records) != len(source):
        raise ValueError('incomplete source timeline')
    for row, expected in zip(records, source):
        if row[index_key] != expected['frame_index'] or row['time_s'] != expected['time_s']:
            raise ValueError('source frame/time mismatch')
        if hash_key and row[hash_key] != expected['frame_sha256']:
            raise ValueError('source pixel hash mismatch')


def table_count(table, columns, times, rows, filter_name=None):
    columns = {name: i for i, name in enumerate(columns)}
    result = np.zeros((len(times), len(rows)), np.int32)
    row_lookup = {int(row): j for j, row in enumerate(rows)}
    for entry in table:
        i, r = entry[columns['time_index']], entry[columns['original_row_index']]
        if not np.isfinite([i, r]).all() or i != int(i) or r != int(r) or not 0 <= i < len(times) or int(r) not in row_lookup:
            raise ValueError('ragged table outside bound time/row axis')
        if filter_name:
            value = entry[columns[filter_name]]
            if value not in (0, 1):
                raise ValueError('nonboolean qualification/persistence')
            if not value:
                continue
        result[int(i), row_lookup[int(r)]] += 1
    return result


def load_all():
    lp, cp, tp = local.verify(), continuity.verify(), tracks.verify()
    paths = [local.BASE / 'protocol.json', local.BASE / 'left/report.json', local.BASE / 'left/observations.npz',
             continuity.BASE / 'protocol.json', continuity.BASE / 'left/report.json', continuity.BASE / 'left/continuity.npz',
             tracks.BASE / 'protocol.json', tracks.BASE / 'left/tracks.json', tracks.BASE / 'right/tracks.json', Path(lp['box_audit'])]
    hashes = {str(p): sha(p) for p in paths}
    lr, cr = read(paths[1]), read(paths[4])
    for report, protocol, array, hash_name in ((lr, paths[0], paths[2], 'observations_sha256'), (cr, paths[3], paths[5], 'table_sha256')):
        if report.get('status') != 'complete' or report['protocol_sha256'] != hashes[str(protocol)] or report[hash_name] != hashes[str(array)]:
            raise ValueError('incomplete or unbound V22 outputs')
    raw = raw_reader.load_bound_session(lp['upstream_base'], 'left')
    hashes.update(raw['input_hashes'])
    source = raw['index']['records']
    validate_axis(lr['records'], source, hash_key='frame_sha256')
    validate_axis(cr['records'], source, index_key='source_frame_index')
    with np.load(paths[2], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(paths[5], allow_pickle=False) as z:
        carrays = {k: z[k] for k in z.files}
    times, rows = arrays['times_s'], arrays['original_row_indices']
    expected_times = np.asarray([r['time_s'] for r in source])
    if (not np.array_equal(times, expected_times) or not np.array_equal(carrays['times_s'], times)
            or not np.array_equal(carrays['frame_indices'], [r['frame_index'] for r in source])
            or not np.array_equal(rows, np.arange(66, 90)) or arrays['usable'].shape != (len(times), len(rows))):
        raise ValueError('bound NPZ frame or row axes mismatch')
    counts = dict(
        all_edges=table_count(arrays['edge_table'], lr['edge_columns'], times, rows),
        bands=table_count(arrays['pair_table'], lr['pair_columns'], times, rows, 'band_candidate'),
        geometric_persistent=table_count(carrays['table'], cr['columns'], times, rows, 'persistent'),
        qualified_persistent=table_count(carrays['table'], cr['columns'], times, rows, 'qualified_persistent'))
    if not np.array_equal(counts['all_edges'], arrays['edge_counts']) or not np.array_equal(counts['bands'], arrays['band_counts']):
        raise ValueError('ragged tables/count arrays disagree')
    for i, record in enumerate(cr['records']):
        if (counts['geometric_persistent'][i].sum() != record['persistent_edge_pairs']
                or counts['qualified_persistent'][i].sum() != record['persistent_qualified_segments']):
            raise ValueError('continuity table/report counts disagree')
    tr = {}
    for side in ('left', 'right'):
        report = read(tracks.BASE / side / 'tracks.json')
        ip = Path(tp['source_indices'][side]['path'])
        index = read(ip)
        hashes[str(ip)] = sha(ip)
        if (report.get('status') != 'complete' or report['protocol_sha256'] != hashes[str(paths[6])]
                or report['source_video'] != index['video'] or report['side'] != side):
            raise ValueError('unbound flange tracking report')
        validate_axis(report['records'], index['records'], hash_key='source_frame_sha256')
        tr[side] = report
    return dict(local_report=lr, continuity_report=cr, arrays=arrays, counts=counts,
                tracks=tr, raw=raw, box_audit=read(paths[-1]), hashes=hashes)


def header(draw, title, note):
    draw.text((24, 12), title, font=font(29), fill='white')
    draw.text((24, 56), note, font=font(22), fill='#ffd388')


def timeline_ticks(draw, x, y, width, height):
    for t in range(0, 61, 10):
        xx = x + width * t / 60
        draw.line((xx, y, xx, y + height), fill='#40505c', width=1)
        draw.text((xx - 9, y + height + 3), str(t), font=font(17), fill='white')


def edge_figure(data):
    a, report = data['arrays'], data['local_report']
    rows, times = a['original_row_indices'], a['times_s']
    canvas = Image.new('RGB', (1700, 130 + 24 * 90), '#151c25')
    d = ImageDraw.Draw(canvas)
    header(d, 'V22 全部局部多边缘 · 原行 66–89 · 完整 0–60 秒',
           '青色=正梯度；橙色=负梯度；均为图像候选非换层。每行纵轴固定法向 −65…+65px；灰=不可用。')
    c = {n: i for i, n in enumerate(report['edge_columns'])}
    for j, row in enumerate(rows):
        y = 112 + j * 90
        d.text((15, y + 17), f'行{row}', font=font(19), fill='white')
        for i, valid in enumerate(a['usable'][:, j]):
            if not valid:
                xx = 112 + 1550 * times[i] / 60
                d.rectangle((xx, y, xx + 3, y + 58), fill=UNKNOWN)
        d.line((112, y + 29, 1662, y + 29), fill='#293947')
        selected = a['edge_table'][a['edge_table'][:, c['original_row_index']] == row]
        for entry in selected:
            xx = 112 + 1550 * times[int(entry[c['time_index']])] / 60
            yy = y + 58 * (65 - entry[c['normal_offset_px']]) / 130
            color = '#71e0ec' if entry[c['polarity']] > 0 else '#ffa361'
            d.ellipse((xx - 1, yy - 1, xx + 1, yy + 1), fill=color)
        timeline_ticks(d, 112, y, 1550, 58)
    return canvas


def matrix_figure(data):
    a = data['arrays']; valid = a['usable'].astype(bool)
    labels = [('all_edges', '所有直接边缘数量'), ('bands', '当前满足明暗带资格的边对数量'),
              ('geometric_persistent', '纠正统计：纯几何边对已持续数量（不因资格翻转断链）'),
              ('qualified_persistent', '纠正统计：明暗带资格连续满足并已持续数量')]
    canvas = Image.new('RGB', (1700, 1310), '#151c25'); d = ImageDraw.Draw(canvas)
    header(d, 'V22 图像边缘、资格与连续性分开显示',
           '蓝色仅为图像候选非换层；固定色阶0…8，超过8仅显示截顶（原数值保留）；灰=无观测，启动≤5秒不可用。')
    for k, (name, label) in enumerate(labels):
        y = 116 + k * 294
        d.text((24, y), label, font=font(25), fill='white')
        values = np.clip(data['counts'][name].T, 0, 8) / 8
        image = np.stack((20 + 40 * values, 32 + 143 * values, 45 + 190 * values), axis=-1).astype(np.uint8)
        image[~valid.T] = UNKNOWN
        canvas.paste(Image.fromarray(image).resize((1550, 218), Image.Resampling.NEAREST), (112, y + 39))
        d.text((23, y + 42), '行66', font=font(19), fill='white')
        d.text((23, y + 228), '行89', font=font(19), fill='white')
        timeline_ticks(d, 112, y + 39, 1550, 218)
    return canvas


def bucket_summary(data):
    times = data['arrays']['times_s']; counts = data['counts']; usable = data['arrays']['usable']
    output = []
    for start in range(0, 60, 5):
        use = (times >= start) & ((times <= 60) if start == 55 else (times < start + 5))
        output.append(dict(start_s=start, end_s=start + 5, end_inclusive=start == 55,
            frames=int(use.sum()), usable_row_cells=int(usable[use].sum()),
            multiple_edge_cells=int((counts['all_edges'][use] >= 2).sum()),
            band_cells=int((counts['bands'][use] > 0).sum()),
            geometric_persistent_cells=int((counts['geometric_persistent'][use] > 0).sum()),
            qualified_persistent_cells=int((counts['qualified_persistent'][use] > 0).sum())))
    return output


def bucket_figure(buckets):
    canvas = Image.new('RGB', (1700, 850), '#151c25'); d = ImageDraw.Draw(canvas)
    header(d, 'V22 固定5秒桶 · 24行×帧的图像观测统计', '这是观测单元数量，不是精度、材料根数或新层事件。几何连续性与明暗资格已分开统计。')
    fields = [('start_s', '时间桶', 25), ('frames', '帧', 250), ('usable_row_cells', '有效单元', 410),
              ('multiple_edge_cells', '有≥2边', 650), ('band_cells', '有明暗带', 900),
              ('geometric_persistent_cells', '有持续几何边对', 1130), ('qualified_persistent_cells', '有持续资格带', 1420)]
    for _, label, x in fields: d.text((x, 108), label, font=font(23), fill='white')
    for i, b in enumerate(buckets):
        for field, _, x in fields:
            text = f"[{b['start_s']},{b['end_s']}{']' if b['end_inclusive'] else ')'}秒" if field == 'start_s' else str(b[field])
            d.text((x, 160 + 50 * i), text, font=font(23), fill='#cddfea')
    return canvas


def tracks_figure(side, report):
    records = report['records']; times = np.asarray([r['time_s'] for r in records])
    reasons = sorted(set(reason for r in records for reason in r['failed_reasons']))
    height = 665 + max(1, len(reasons)) * 33
    canvas = Image.new('RGB', (1700, height), '#151c25'); d = ImageDraw.Draw(canvas)
    header(d, f'V22 法兰图像追踪 | {raw_reader.LABELS[side]} | 0–60秒',
           '这里只显示联合筛选候选、ID寿命与失败原因；不显示角度、不累计圈数。失败≠原图完全无纹理。')
    n = sum(r['candidate_available'] for r in records)
    d.text((24, 103), f'通过联合门控的相邻帧：{n}/{len(records)-1}；首帧仅初始化。', font=font(25), fill='white')
    strip = np.array([(80, 177, 225) if r['candidate_available'] else (102, 71, 71) for r in records], np.uint8)[None]
    canvas.paste(Image.fromarray(strip).resize((1320, 30), Image.Resampling.NEAREST), (355, 145))
    timeline_ticks(d, 355, 145, 1320, 30)
    d.text((24, 214), '联合保留下来的原ID寿命', font=font(23), fill='white')
    ages = np.array([max(r['native_track_age_s']) if r['native_track_age_s'] else np.nan for r in records])
    if np.isfinite(ages).any():
        limit = max(1., float(np.nanmax(ages)))
        for t, age in zip(times, ages):
            if np.isfinite(age):
                x = 355 + 1320 * t / 60; y = 321 - age / limit * 80
                d.ellipse((x-1,y-1,x+1,y+1), fill='#64c1eb')
        d.text((24, 251), f'显示范围0–{limit:.2f}秒', font=font(19), fill='white')
    else:
        d.text((355, 248), '没有通过联合门控的ID寿命样本；不能解释为天然纹理寿命为0。', font=font(23), fill='#ffd388')
    d.text((24, 348), 'LK/FB/当前ROI及颜色掩膜', font=font(22), fill='white')
    d.text((24, 379), '联合剩点：青=ROI1 橙=ROI2', font=font(22), fill='white')
    d.text((24, 410), '固定纵轴 0–100 个', font=font(21), fill='white')
    for region, color in ((0, '#71e0ec'), (1, '#ffa361')):
        points = [(355 + 1320*t/60, 476 - 110*min(100, r['fb_accepted_count_by_region'][region])/100) for t,r in zip(times,records)]
        d.line(points, fill=color, width=1)
    timeline_ticks(d, 355, 362, 1320, 115)
    d.text((24, 521), '全部失败门槛（同一帧可多项同时失败；深红=该原因发生）', font=font(25), fill='white')
    totals = Counter(reason for r in records for reason in r['failed_reasons'])
    for i, reason in enumerate(reasons):
        y = 565 + i*33
        d.text((15, y), f'{reason}: {totals[reason]}', font=font(15), fill='white')
        matrix = np.array([(191,87,77) if reason in r['failed_reasons'] else (29,36,45) for r in records],np.uint8)[None]
        canvas.paste(Image.fromarray(matrix).resize((1320,24),Image.Resampling.NEAREST),(355,y))
    timeline_ticks(d,355,565,1320,max(1,len(reasons))*33)
    return canvas


def crop_with_outer_label(raw, inclusive_box, source_frame, time_s):
    x0,y0,x1,y1 = map(int,inclusive_box)
    if not 0<=x0<=x1<raw.shape[1] or not 0<=y0<=y1<raw.shape[0]: raise ValueError('crop outside original')
    crop = Image.fromarray(cv2.cvtColor(raw[y0:y1+1,x0:x1+1],cv2.COLOR_BGR2RGB))
    width = crop.width + crop.width%2; height = crop.height+36; height += height%2
    labelled = Image.new('RGB',(width,height),'black');labelled.paste(crop,(0,36))
    ImageDraw.Draw(labelled).text((6,5),f'SOURCE f{source_frame:04d} | {time_s:04.1f}s | RAW CROP',font=font(17),fill='white')
    return crop,labelled


def array_pixel_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape,str(array.dtype))).encode('ascii')+array.tobytes()).hexdigest()


def verify_lossless_replay(path, expected_hashes, fps):
    check=cv2.VideoCapture(str(path))
    try:
        if (not check.isOpened() or round(check.get(cv2.CAP_PROP_FRAME_COUNT))!=len(expected_hashes)
                or abs(check.get(cv2.CAP_PROP_FPS)-fps)>1e-6):raise ValueError('encoded replay timeline mismatch')
        for i, expected in enumerate(expected_hashes):
            ok, frame=check.read()
            if not ok or array_pixel_hash(frame)!=expected:
                raise ValueError(f'lossless encoded replay pixel mismatch at frame {i}')
        if check.read()[0]:raise ValueError('unexpected extra encoded frame')
    finally:check.release()


def raw_context_replay(data, output):
    raw_data=data['raw']; video=raw_data['video'];box=data['box_audit']['diagnostic_crop_xyxy_inclusive']
    cap=raw_reader.open_bound_video(video); writer=None;frames=[];cards=[];encoded_hashes=[]
    path=output/'left_full_0_60s_raw_crop_FFV1.avi'
    try:
        for record in raw_data['index']['records']:
            ok,frame=cap.read()
            if not ok:raise ValueError('source decode ended early')
            actual=round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
            validate_original_frame(frame,actual,record,video)
            crop,labelled=crop_with_outer_label(frame,box,actual,record['time_s'])
            if writer is None:
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),video['fps'],labelled.size)
                if not writer.isOpened():raise ValueError('lossless FFV1 writer unavailable; no lossy fallback')
            encoded=cv2.cvtColor(np.asarray(labelled),cv2.COLOR_RGB2BGR)
            writer.write(encoded)
            encoded_hashes.append(array_pixel_hash(encoded))
            frames.append(dict(source_frame=actual,time_s=record['time_s'],original_frame_sha256=record['frame_sha256']))
            if actual%round(video['fps']*10)==0:
                png=raw_reader.save_new_image(crop,output/f'raw_crop_f{actual:06d}_t{record["time_s"]:.1f}.png',output)
                cards.append(dict(frame=actual,time=record['time_s'],picture=crop,artifact=png))
            if actual%100==0:print('V22 raw replay verified frame',actual,flush=True)
    finally:
        cap.release()
        if writer is not None:writer.release()
    verify_lossless_replay(path,encoded_hashes,video['fps'])
    canvas=Image.new('RGB',(1700,150+4*445),'#151c25');d=ImageDraw.Draw(canvas)
    header(d,'V22 原始固定邻域 · 每10秒均匀取帧（另存无损原尺寸PNG）',
           '原图无增强、镜像或算法线；固定裁剪来自既有红框空间审计；不是事件筛选或新层判断。')
    for k,card in enumerate(cards):
        x=(k%2)*850;y=130+(k//2)*445
        d.text((x+16,y),f"源帧{card['frame']} | {card['time']:.1f}秒",font=font(24),fill='white')
        raw_reader._fit(canvas,card['picture'],(x+14,y+40,820,392))
    sheet=raw_reader.save_new_image(canvas,output/'raw_fixed_10s_contact.png',output)
    return dict(path=str(path.relative_to(output)),sha256=sha(path),source_video=video,
        crop_box_raw_inclusive=box,source_frames=frames,frame_count=len(frames),fps=video['fps'],
        original_speed=True,source_pixel_hashes_verified=True,
        codec='FFV1_lossless_BGR_pixel_roundtrip_verified_all_frames',
        encoded_frame_pixel_hashes=encoded_hashes,encoded_pixels_verified=True,
        odd_dimensions_padded_black_without_crop_truncation=True,
        border_label_outside_image=True,enhancement=False,mirror=False,algorithm_overlay=False,
        fixed_ten_second_frames=[{k:v for k,v in c.items() if k!='picture'} for c in cards],contact_sheet=sheet)


def run(output=None):
    output=Path(output).resolve() if output else OUT/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if output.exists():raise FileExistsError('fresh diagnostic directory required')
    data=load_all();output.mkdir(parents=True)
    figures={}
    buckets=bucket_summary(data)
    for name,picture in [('all_signed_edges',edge_figure(data)),('band_and_continuity',matrix_figure(data)),
                         ('five_second_counts',bucket_figure(buckets)),
                         *[(side+'_tracking',tracks_figure(side,r)) for side,r in data['tracks'].items()]]:
        figures[name]=raw_reader.save_new_image(picture,output/(name+'.png'),output)
    replay=raw_context_replay(data,output)
    local.verify();continuity.verify();tracks.verify();raw_reader.verify_unchanged(data['hashes'])
    manifest=dict(schema='v22_independent_image_diagnostics_1',script_sha256=sha(__file__),
        helper_sha256=sha(raw_reader.__file__),input_hashes=data['hashes'],figures=figures,
        local_five_second_buckets=buckets,raw_replay=replay,
        interpretation='Image edges/contrast qualification/geometric pair continuity and gated tracking only; no new material or physical phase confirmation.',
        truth_files_opened=False,reference_event_lines_drawn=False,events_predicted=False,
        mechanical_angles_drawn=False,hardware_access=False,main_ui_modified=False)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(output=str(output)),ensure_ascii=False));return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    run(parser.parse_args().output)
