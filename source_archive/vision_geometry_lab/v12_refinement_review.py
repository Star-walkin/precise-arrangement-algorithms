"""TRAIN-only posthoc review of frozen V12 results; no model/refiner execution.

The accepted-point distances below are a selected, correlated point diagnostic,
NOT the frozen arc-length path score. GT is used only for scoring/display. We do
not inspect held-out images/results or use GT to run any inference. Failed local
refinements retain V11 candidates and are never displayed as new CV evidence.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .centerline_path_metrics_v6 import clip_scoring_segments, _nearest
from .centerline_quality_v5 import load_label_quality
from .centerline_review_v5 import validate_annotation, read_clean_image


BASE = Path(__file__).resolve().parent / 'outputs' / 'v12_profile_refinement'
METHODS = ('bright_ridge', 'paired_edges')
FIXED_IDS = ('dual_20260924_111500__left__f00246',
             'C07__R_run_stop_run_20260922_01__t012.033__f000368')
COLORS = dict(gt='#62f479', baseline='#29d8ff', accepted='#ff4ee2',
              shift='#ffc857', fallback='#999999', text='#e7edf5')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def length(pieces):
    return sum(float(np.linalg.norm(np.diff(np.asarray(p, float), axis=0), axis=1).sum())
               for p in pieces if len(p) >= 2)


def outside(point, polygons):
    return all(cv2.pointPolygonTest(np.asarray(p, np.float32), tuple(map(float, point)), False) < 0
               for p in polygons)


def trusted_distance(point, segments):
    """Reject projections onto any clipped/terminal GT endpoint, conservatively.

    A finite line's endpoint distance must not silently assess an unannotated
    extension. Ordinary interior annotation vertices remain valid. No distance
    cap is used to discard large errors from the accepted-point diagnostic.
    """
    if not segments:
        return None
    distance, foot = _nearest(np.asarray([point], float), segments)
    if not np.isfinite(distance[0]):
        return None
    endpoints = [end for piece in segments for end in (piece[0], piece[-1])]
    if any(np.linalg.norm(foot[0] - np.asarray(end, float)) <= 1e-6 for end in endpoints):
        return None
    return float(distance[0])


def audit_points(row, label, quality):
    """Score only accepted points on mutually valid before/after GT support."""
    polygons = label['ignore_polygons'] + quality['precision_exclude_polygons']
    support = row['metrics']['score_support']
    references = {s: support[s + '_reference_segments'] for s in ('full', 'downstream')}
    accepted, exclusions = [], {s: Counter() for s in references}
    seed = np.asarray(label['polylines']['free_seed'], float)
    unit = (seed[-1] - seed[0]) / np.linalg.norm(seed[-1] - seed[0])
    for record in row['refinement']['diagnostics']['records']:
        if not record['accepted']:
            continue
        before, after = np.asarray(record['input_xy']), np.asarray(record['final_xy'])
        shift_length = float(np.linalg.norm(after - before))
        item = dict(sample_id=row['sample_id'], method=row['method'],
                    segment_index=record['segment_index'], sample_index=record['sample_index'],
                    input_xy=before.tolist(), final_xy=after.tolist(),
                    applied_offset_px=record['applied_offset_px'], scopes={})
        for scope, reference in references.items():
            reason = None
            if not outside(before, polygons) or not outside(after, polygons):
                reason = 'before_or_after_in_unknown_or_ignore_region'
            elif shift_length > 1e-9 and abs(length(clip_scoring_segments(
                    [[before.tolist(), after.tolist()]], ignore_polygons=polygons)) - shift_length) > 1e-6:
                reason = 'shift_crosses_unknown_or_ignore_region'
            elif scope == 'downstream' and (float((before-seed[-1])@unit) < -1e-9 or
                                            float((after-seed[-1])@unit) < -1e-9):
                reason = 'before_or_after_outside_downstream_halfplane'
            d0, d1 = trusted_distance(before, reference), trusted_distance(after, reference)
            if reason is None and (d0 is None or d1 is None):
                reason = 'no_interior_projection_on_trusted_reference'
            if reason:
                exclusions[scope][reason] += 1
                item['scopes'][scope] = dict(eligible=False, reason=reason)
            else:
                item['scopes'][scope] = dict(eligible=True, before_px=d0, after_px=d1,
                    delta_px=d1-d0, changed_class='improved' if d1-d0 < -.05 else
                    'worsened' if d1-d0 > .05 else 'within_0.05px')
        accepted.append(item)
    return accepted, {s: dict(c) for s, c in exclusions.items()}


def summarize_points(items, scope):
    values = [p['scopes'][scope] for p in items if p['scopes'][scope]['eligible']]
    if not values:
        return dict(accepted_count=len(items), eligible_count=0, summary=None)
    before = np.array([v['before_px'] for v in values])
    after = np.array([v['after_px'] for v in values])
    delta = after-before
    return dict(accepted_count=len(items), eligible_count=len(values),
                median_before_px=float(np.median(before)), median_after_px=float(np.median(after)),
                mean_before_px=float(np.mean(before)), mean_after_px=float(np.mean(after)),
                median_delta_px=float(np.median(delta)), mean_delta_px=float(np.mean(delta)),
                p95_before_px=float(np.percentile(before, 95)), p95_after_px=float(np.percentile(after, 95)),
                improved_count=int(np.sum(delta < -.05)), worsened_count=int(np.sum(delta > .05)),
                within_0_05px_count=int(np.sum(abs(delta) <= .05)),
                note='selected accepted points, correlated and NOT arc-length-weighted; no inference of generalization')


def stress_summary(stress):
    result = []
    for offset in sorted({r['offset_widths'] for r in stress}):
        for method in ('unrefined',) + METHODS:
            selected = [r for r in stress if r['offset_widths'] == offset and r['method'] == method]
            for scope in ('full_path', 'downstream'):
                metrics = [r['metrics'][scope] for r in selected]
                ref = sum(m['reference_length_px'] for m in metrics)
                pred = sum(m['predicted_length_px'] for m in metrics)
                extra = sum(m.get('extra_predicted_length_px') or 0. for m in metrics)
                covered = sum(m.get('covered_reference_length_px') or 0. for m in metrics)
                total = sum(r.get('counts', {}).get('total_sampled_points', 0) for r in selected)
                accepted = sum(r.get('counts', {}).get('accepted_points', 0) for r in selected)
                result.append(dict(offset_widths=offset, method=method, scope=scope,
                    samples=len(selected), covered_fraction=covered/ref if ref else None,
                    prediction_precision_at_5px=1-extra/pred if pred else None,
                    accepted_points=accepted if method != 'unrefined' else None,
                    sampled_points=total if method != 'unrefined' else None,
                    gt_used_for_input=True, model_performance=False))
    return result


def font(size):
    for path in (Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/msyh.ttc')):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def at_fraction(segments, fraction):
    lengths = np.array([length([p]) for p in segments])
    target = fraction * lengths.sum()
    for piece, size in zip(segments, lengths):
        if target <= size:
            p = np.asarray(piece, float)
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            return np.array([np.interp(target, arc, p[:, d]) for d in (0, 1)])
        target -= size
    return np.asarray(segments[-1][-1], float)


def crop_for(center, width, height, image_size):
    w, h = image_size
    width, height = min(int(np.ceil(width)), w), min(int(np.ceil(height)), h)
    x = int(np.clip(round(center[0]-width/2), 0, w-width))
    y = int(np.clip(round(center[1]-height/2), 0, h-height))
    return [x, y, x+width, y+height]


def draw_panel(image, crop, size, reference, original, row, zoom=False):
    panel = Image.new('RGB', size, '#141a23')
    x0, y0, x1, y1 = crop
    scale = min(size[0]/(x1-x0), size[1]/(y1-y0))
    scaled = (round((x1-x0)*scale), round((y1-y0)*scale))
    origin = ((size[0]-scaled[0])//2, (size[1]-scaled[1])//2)
    panel.paste(image.crop(crop).resize(scaled, Image.Resampling.LANCZOS), origin)
    draw = ImageDraw.Draw(panel)
    def xy(p):
        return ((p[0]-x0)*scale+origin[0], (p[1]-y0)*scale+origin[1])
    def lines(pieces, color, width):
        for piece in pieces:
            if len(piece) > 1:
                draw.line([xy(p) for p in piece], fill=color, width=width)
    lines(reference, COLORS['gt'], 2)
    lines(original, COLORS['baseline'], 2)
    if row['method'] != 'original_v11':
        records = row['refinement']['diagnostics']['records']
        for r in records:
            if not r['accepted']:
                # Sparse outlined gray fallback markers, never a new measured line.
                if zoom and r['sample_index'] % 4 == 0:
                    q = xy(r['input_xy'])
                    draw.ellipse((q[0]-2, q[1]-2, q[0]+2, q[1]+2), outline=COLORS['fallback'])
                continue
            a, b = xy(r['input_xy']), xy(r['final_xy'])
            draw.line([a, b], fill=COLORS['shift'], width=2)
            radius = 3 if zoom else 2
            draw.ellipse((b[0]-radius,b[1]-radius,b[0]+radius,b[1]+radius), fill=COLORS['accepted'])
        # Only contiguous accepted runs are connected; fallback gaps stay gaps.
        lines(row['refinement']['accepted_segments'], COLORS['accepted'], 2)
    # Source geometry outside the crop must not leak into letterboxed margins.
    if origin[0] > 0:
        draw.rectangle((0,0,origin[0]-1,size[1]-1), fill='#141a23')
        draw.rectangle((origin[0]+scaled[0],0,size[0]-1,size[1]-1), fill='#141a23')
    if origin[1] > 0:
        draw.rectangle((0,0,size[0]-1,origin[1]-1), fill='#141a23')
        draw.rectangle((0,origin[1]+scaled[1],size[0]-1,size[1]-1), fill='#141a23')
    draw.rectangle((0,0,size[0]-1,size[1]-1), outline='#596270')
    if zoom:
        draw.rectangle((8,8,150,30), fill='#171d25')
        draw.text((12,10), f'{scale:.2f}x | 20 source px', font=font(14), fill='white')
        draw.line([(14,size[1]-18),(14+20*scale,size[1]-18)], fill='white', width=3)
    return panel


def figure(sample, image, rows, path):
    baseline = rows['original_v11']
    reference = baseline['metrics']['score_support']['full_reference_segments']
    downstream = baseline['metrics']['score_support']['downstream_reference_segments']
    original = baseline['decoded']['segments']
    width = baseline['independent_width']['width_px']
    points = np.vstack([np.asarray(p) for p in reference+original])
    lo, hi = points.min(axis=0)-60, points.max(axis=0)+60
    context = crop_for((lo+hi)/2, hi[0]-lo[0], hi[1]-lo[1], image.size)
    crops = [('Path context (same RGB, no brightness adjustment)', context)]
    crop_metadata = []
    for method in METHODS:
        accepted = [r for r in rows[method]['refinement']['diagnostics']['records'] if r['accepted']]
        if accepted:
            record = max(accepted, key=lambda r: abs(r['applied_offset_px']))
            center = np.asarray(record['input_xy'])
            criterion = f'{method}: largest accepted displacement (posthoc, not representative accuracy)'
            title = f"Zoom at {method} max shift {record['applied_offset_px']:+.2f}px"
        else:
            center = at_fraction(downstream, .6)
            criterion = f'{method}: none accepted; fixed 60% trusted downstream arclength for DISPLAY only'
            title = f'Zoom: {method} accepts 0 points; original retained, NOT new evidence'
        crop = crop_for(center, max(120., width*7), max(90., width*4.8), image.size)
        crops.append((title, crop))
        crop_metadata.append(dict(method=method, crop_xyxy=crop, selection=criterion))
    panel_size = (610, 315)
    canvas = Image.new('RGB', (1860, 1230), '#0e141c')
    draw = ImageDraw.Draw(canvas)
    draw.text((18,14), 'V12 TRAIN-ONLY POSTHOC | fixed image: '+sample['sample_id'], font=font(22), fill='white')
    draw.text((18,46), 'GT = trusted human reference (display/score only). Candidate identity and physical accuracy are NOT confirmed.',
              font=font(17), fill='#e0e4ec')
    legend = [('GT center', 'gt'), ('Original V11', 'baseline'), ('Accepted local candidate', 'accepted'),
              ('Actual shift vector', 'shift'), ('Fallback: NOT CV evidence', 'fallback')]
    cursor = 18
    for label, color in legend:
        draw.line([(cursor,83),(cursor+25,83)], fill=COLORS[color], width=4)
        draw.text((cursor+32,73), label, font=font(16), fill=COLORS[color])
        cursor += len(label)*8.7+58
    methods = ('original_v11',)+METHODS
    for k, method in enumerate(methods):
        row = rows[method]
        title = method
        if method != 'original_v11':
            d = row['refinement']['diagnostics']
            title += f" | accepted {d['accepted_points']}/{d['sampled_points']}"
        draw.text((15+615*k,107), title, font=font(19), fill='white')
    for j, (title,crop) in enumerate(crops):
        y = 141+j*350
        draw.text((18,y), title, font=font(16), fill='#d4d9e3')
        for k, method in enumerate(methods):
            panel = draw_panel(image, crop, panel_size, reference, original, rows[method], zoom=j>0)
            canvas.paste(panel,(10+k*615,y+27))
    draw.text((18,1200), 'Pixels are magnified, not evidence of subpixel physical accuracy. Training examples cannot establish generalization.',
              font=font(17), fill='#ffc857')
    canvas.save(path)
    return dict(sample_id=sample['sample_id'], image=str(path.resolve()), context_crop_xyxy=context,
                zoom_crops=crop_metadata, brightness_adjusted=False, training_only=True)


def self_test():
    reference = [[[0.,0.],[5.,0.],[10.,0.]]]
    assert trusted_distance([5.,2.], reference) == 2.
    assert trusted_distance([-1.,2.], reference) is None
    assert trusted_distance([10.,2.], reference) is None
    polygon = [[3.,-1.],[7.,-1.],[7.,1.],[3.,1.]]
    assert not outside([3.,0.], [polygon])
    assert outside([2.,0.], [polygon])
    assert abs(length(clip_scoring_segments([[[0.,0.],[10.,0.]]], ignore_polygons=[polygon]))-6.) < 1e-6


def run(output, destination=None):
    self_test()
    result_path = output/'train/results.json'
    manifest_path = output/'train_manifest.json'
    protocol_path = output/'protocol.json'
    for path in (result_path,manifest_path,protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f'Frozen TRAIN run must finish first: {path}')
    results, manifest, protocol = read(result_path), read(manifest_path), read(protocol_path)
    if results.get('split') != 'train' or results['protocol_sha256'] != sha(protocol_path):
        raise ValueError('TRAIN split / frozen protocol hash mismatch')
    samples = manifest['samples']
    if len(samples) != 10 or any(s['split'] != 'train' for s in samples):
        raise ValueError('Expected exactly the frozen 10 TRAIN samples; heldout access prohibited')
    if '20260925' in json.dumps(samples):
        raise ValueError('9/25 data are prohibited')
    for path, expected in protocol['source_snapshot'].items():
        if sha(path) != expected:
            raise ValueError('Frozen V12 source changed: '+path)
    destination = destination or output/'posthoc_train_review'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite posthoc output: {destination}')
    rows, annotation_data, inputs = {}, {}, {str(p.resolve()):sha(p) for p in (result_path,manifest_path,protocol_path)}
    for sample in samples:
        sid = sample['sample_id']
        annotation = Path(sample['annotation_path'])
        if sha(annotation) != sample['annotation_sha256']:
            raise ValueError('Changed TRAIN annotation: '+sid)
        label = validate_annotation(read(annotation), sample)
        quality = load_label_quality(sample)
        if quality['quality_sha256'] != sample.get('quality_sha256'):
            raise ValueError('Changed TRAIN trusted quality: '+sid)
        annotation_data[sid] = label, quality
        rows[sid] = {}
        for method in ('original_v11',)+METHODS:
            path = output/'train/rows'/(sid+'__'+method+'.json')
            row = read(path)
            if row['sample_id'] != sid or row['method'] != method or row['split'] != 'train':
                raise ValueError('Unexpected row identity')
            compact = next(r for r in results['rows'] if r['sample_id']==sid and r['method']==method)
            if compact != {k:v for k,v in row.items() if k != 'refinement'}:
                raise ValueError('Detailed row differs from frozen TRAIN results: '+sid+' '+method)
            reference = quality.get('supervision_polylines', label['polylines'])['centerline']
            ignored = label['ignore_polygons'] + quality['precision_exclude_polygons']
            for scope in ('full','downstream'):
                expected = clip_scoring_segments([reference], ignore_polygons=ignored,
                    free_seed=label['polylines']['free_seed'], downstream=scope=='downstream')
                if expected != row['metrics']['score_support'][scope+'_reference_segments']:
                    raise ValueError('Stored trusted scoring support differs from TRAIN quality: '+sid)
            rows[sid][method] = row
            inputs[str(path.resolve())] = sha(path)
    all_points, exclusions = [], {}
    for sid, methods in rows.items():
        for method in METHODS:
            items, omitted = audit_points(methods[method], *annotation_data[sid])
            all_points.extend(items)
            exclusions[sid+'__'+method] = omitted
    summaries = {method: {scope:summarize_points([p for p in all_points if p['method']==method],scope)
                          for scope in ('full','downstream')} for method in METHODS}
    main_scores = {m:[r for r in values if r['split']=='train' and r['cohort']=='all']
                   for m,values in results['summaries'].items()}
    stress = stress_summary(results['stress_rows'])
    destination.mkdir()
    figures = []
    for sid in FIXED_IDS:
        sample = next(s for s in samples if s['sample_id']==sid)
        # This is the only RGB read: two specified TRAIN images, hash-validated.
        image = read_clean_image(sample)
        path = destination/('f246_comparison.png' if sid==FIXED_IDS[0] else 'C07_comparison.png')
        figures.append(figure(sample,image,rows[sid],path))
    report = dict(purpose='posthoc TRAIN attribution, not model selection or generalized accuracy',
        protocol_sha256=sha(protocol_path), source_sha256=sha(__file__), input_sha256=inputs,
        fixed_images=list(FIXED_IDS), images_read=2, heldout_read=False, blind_0925_read=False,
        model_or_refiner_executed=False, hardware_access=False, force_command=None,
        point_metric_policy='Euclidean distance to trusted finite GT pieces; exclude masks and shift crossings; '
            'exclude projections on GT piece endpoints; GT used only for score/display; 0.05px ties are numerical only',
        point_metric_warning='Accepted-only selection, correlated points, no arc-length weighting; not comparable to frozen path score',
        accepted_point_summaries=summaries, accepted_points=all_points, exclusion_counts=exclusions,
        frozen_train_scores=main_scores, frozen_point_counts=results['counts'],
        gt_derived_stress=stress, stress_warning=results['stress_warning'], figures=figures)
    (destination/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines = ['# V12 训练集事后修正审查','',
        '本报告仅解释已冻结结果，不改模型、阈值、评分或选模。只读取 10 张训练标注和指定 f246/C07 两张训练原图；不读取留出图或 9/25 数据。', '',
        '绿线为可信人工中心，青线为原 V11 候选，粉点/线仅表示接受的局部光度修正，黄色矢量为实际位移。灰色回退仍是旧候选，不是 CV 新证据。图不作亮度调整。', '',
        '放大框按各方法最大已接受位移选取（无接受点时为可信下游路径 60% 处），只用于事后查看，不代表典型准确率，不参与任何评分或推理。','',
        '## 已接受点相对人工中心的变化','',
        '|方法|可信下游点数/接受总数|修正前均距 px|修正后均距 px|改善/恶化/近似不变|',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        s = summaries[method]['downstream']
        lines.append(f"|{method}|{s['eligible_count']}/{s['accepted_count']}|{s.get('mean_before_px',0):.3f}|"
                     f"{s.get('mean_after_px',0):.3f}|{s.get('improved_count',0)}/{s.get('worsened_count',0)}/{s.get('within_0_05px_count',0)}|")
    lines += ['', '只统计两端和位移路径均在可信范围、且最近投影不是可信曲线端点的点。尊重人工 ignore 与 quality 排除区，C07/C08 仅到可信 C7。',
        '这是经接受条件筛选的相关点诊断，非独立样本、非弧长加权，也不是原冻结路径评分。不能由训练内改善/恶化推断泛化。','',
        '## GT 派生压力测试（非模型成绩）','',
        '将可信人工中心沿法线人为平移 ±0.45 个独立自由绳宽；真值参与输入，仅检验修正器对这种特定扰动的行为，不模拟完整真实错误分布，不用于参数选择。','',
        '|偏移/绳宽|方法|下游覆盖率|下游 5 px 路径贴合率|', '|---:|---|---:|---:|']
    for s in stress:
        if s['scope']=='downstream':
            lines.append(f"|{s['offset_widths']:+.2f}|{s['method']}|{s['covered_fraction']:.2%}|{s['prediction_precision_at_5px']:.2%}|")
    lines += ['', '完整逐点读数、排除原因、输入哈希及原冻结分数见 review.json。无硬件访问、无施力或新层确认。']
    (destination/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(destination.resolve()),accepted_point_summaries=summaries),ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--destination',type=Path,help='New posthoc output directory; never overwrite an existing directory')
    args = parser.parse_args()
    run(args.output.resolve(), args.destination.resolve() if args.destination else None)
