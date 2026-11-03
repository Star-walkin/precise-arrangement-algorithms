"""V13 supervision preparation only: trusted center and independent edge targets.

Uses the unchanged V11 seed/independent-width input geometry. All intersections
come from actual annotated finite segments. Unknown edges are never completed
by parallel offsets. Targets never modify image, input strip, seed or width.
The reported width is a projected strip-normal image distance, NOT diameter.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import math

import numpy as np

from .v11_path_contract import (HEIGHT, WIDTH, STEP_WIDTHS, _checked_geometry,
    _geometry, _line, _polygon_mask, build_targets, image_to_strip)


LAB = Path(__file__).resolve().parent
DEFAULT_OUTPUT = LAB/'outputs/v13_boundary_targets_train'
TRAIN_MANIFEST = LAB/'outputs/v12_profile_refinement/train_manifest.json'
CACHE_INDEX = LAB/'outputs/v11_path_identity/development_cache/index.json'
TARGET_CONTRACT = dict(version=1, scope='TRAIN supervision availability only; no training or prediction',
    input_geometry='unchanged V11; source shape, free_seed and independent RGB optical width only',
    section='fixed strip column, parallel to strip normal n; not local centerline normal',
    exact_intersection='finite annotation segments; shared vertex duplicate tolerance 1e-6 strip row',
    ambiguous='multiple distinct intersections or collinear nonzero overlap are unknown',
    target_guard='unchanged V11 Gaussian sigma=max(1,2/native_px_per_row), 3-sigma padded/ignore guard',
    side_order='low/high strip-normal row, not camera identity or flange/surface semantic name',
    one_edge='one visible semantic edge assigned to low/high only when trusted center determines its side',
    width='requires trusted center and both semantic edges, center strictly between edges, entire cross-section allowed',
    unknown='independent masks; zero-filled masked targets are NOT background or zero-width evidence',
    quality='validated quality.supervision_polylines overrides center and BOTH edges; C07/C08 only through C7',
    dimensions='local width in strip rows/native image px/free optical-width units; not physical diameter',
    input_modified=False, training=False, hardware_access=False, physical_confirmation=False)
TRAIN_IDS = {
    'dual_20260924_111500__left__f00240', 'dual_20260924_111500__left__f00246',
    'dual_20260924_111500__left__f00273', 'dual_20260924_111143__right__f00114',
    'dual_20260924_111143__right__f00130', 'dual_20260924_111143__right__f00190',
    'C01__dual_20260923_101517__pair0050__right',
    'C02__dual_20260923_101517__pair0550__right',
    'C07__R_run_stop_run_20260922_01__t012.033__f000368',
    'C08__R_run_stop_run_20260922_01__t047.607__f001456'}


def _intersections(polyline, geometry):
    """Exact per-column intersections; no extension or target-nearest selection."""
    line = _line(polyline, 0)
    found = [[] for _ in range(WIDTH)]
    rows = np.zeros(WIDTH, np.float64)
    unique = np.zeros(WIDTH, bool)
    ambiguous = np.zeros(WIDTH, bool)
    if len(line) < 2:
        return rows, unique, ambiguous
    points = image_to_strip(line, geometry)
    for a, b in zip(points[:-1], points[1:]):
        dx, dy = b-a
        if np.linalg.norm(b-a) <= 1e-9:
            continue
        if abs(dx) <= 1e-9:
            nearest = int(round(float(a[0])))
            if 0 <= nearest < WIDTH and abs(a[0]-nearest) < 1e-8:
                found[nearest].extend([float(a[1]), float(b[1])])
            continue
        first = max(0, int(math.ceil(min(a[0],b[0])-1e-9)))
        last = min(WIDTH-1, int(math.floor(max(a[0],b[0])+1e-9)))
        for column in range(first,last+1):
            fraction = (column-a[0])/dx
            if -1e-9 <= fraction <= 1+1e-9:
                found[column].append(float(a[1]+fraction*dy))
    for column, values in enumerate(found):
        if values:
            if np.ptp(values) > 1e-6:
                ambiguous[column] = True
            else:
                unique[column], rows[column] = True, float(np.mean(values))
    return rows, unique, ambiguous


def _curve_targets(geometry, polyline, ignored, valid_mask):
    target = build_targets(geometry, polyline, ignored, valid_mask)
    rows, unique, ambiguous = _intersections(polyline, geometry)
    if np.any(target['column_valid'] & ~unique):
        raise ValueError('V11 supervision and exact intersection disagree')
    rows[~target['column_valid']] = 0.
    return dict(rows=rows.astype(np.float32), valid=target['column_valid'],
        distribution=target['target_distribution'], row_valid=target['row_valid'],
        slope_target=target['slope_target'], slope_valid=target['slope_valid'],
        intersection_unique=unique, ambiguous=ambiguous,
        diagnostics=target['diagnostics'])


def _crosses_polygon(column, low, high, polygon):
    """Continuous section/polygon overlap, including boundary contact."""
    for a, b in zip(polygon, np.roll(polygon,-1,axis=0)):
        dx, dy = b-a
        if abs(dx) <= 1e-10:
            if abs(column-a[0]) <= 1e-8 and max(low,min(a[1],b[1])) <= min(high,max(a[1],b[1]))+1e-8:
                return True
        else:
            fraction = (column-a[0])/dx
            if -1e-9 <= fraction <= 1+1e-9:
                row = a[1]+fraction*dy
                if low-1e-8 <= row <= high+1e-8:
                    return True
    # If no boundary crosses the section, its midpoint may still be inside.
    y, inside = (low+high)/2, False
    for a, b in zip(polygon,np.roll(polygon,-1,axis=0)):
        if (a[1] > y) != (b[1] > y):
            cross_x = a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if column < cross_x:
                inside = not inside
    return inside


def build_boundary_targets(geometry, centerline, edge_flange, edge_surface,
                           ignore_polygons, valid_mask):
    """Return V11-compatible center targets plus independently masked edges.

    edge_rows/edge_valid have shape [2,512]: index 0 is lower n coordinate,
    index 1 higher n. edge_source is 1=flange, 2=surface, 0=unknown. Semantic
    edge rows/masks remain separately available even when side is ambiguous.
    Missing values are zero-filled ONLY with validity false. Width is allowed
    only if both labeled edges bracket a valid center and the whole section is
    visible/trusted. No physical units or contact/layer inference is produced.
    Caller must supply TRUSTED polylines; prefer targets_for_annotation below.
    """
    _checked_geometry(geometry)
    valid = np.asarray(valid_mask,bool)
    if valid.shape != (HEIGHT,WIDTH):
        raise ValueError('Expected V11 256x512 valid mask')
    curves = [_curve_targets(geometry,polyline,ignore_polygons,valid)
              for polyline in (centerline,edge_flange,edge_surface)]
    center, flange, surface = curves
    edge_rows = np.zeros((2,WIDTH),np.float32)
    edge_valid = np.zeros((2,WIDTH),bool)
    edge_distribution = np.zeros((2,HEIGHT,WIDTH),np.float32)
    edge_row_valid = np.zeros((2,HEIGHT,WIDTH),bool)
    edge_source = np.zeros((2,WIDTH),np.int8)
    width_rows = np.zeros(WIDTH,np.float32)
    width_valid = np.zeros(WIDTH,bool)
    width_reasons = Counter()
    allowed = valid & ~_polygon_mask(ignore_polygons,geometry)
    polygons = [image_to_strip(_line(p,3),geometry) for p in ignore_polygons]
    for column in range(WIDTH):
        available = [index for index,c in enumerate((flange,surface)) if c['valid'][column]]
        assigned = []
        if len(available)==2:
            ordered = sorted(available,key=lambda i:curves[i+1]['rows'][column])
            if curves[ordered[1]+1]['rows'][column]-curves[ordered[0]+1]['rows'][column] > 1e-6:
                assigned = list(enumerate(ordered))
        elif len(available)==1 and center['valid'][column]:
            index = available[0]
            offset = curves[index+1]['rows'][column]-center['rows'][column]
            if abs(offset) > 1e-6:
                assigned = [(0 if offset<0 else 1,index)]
        for side,index in assigned:
            curve = curves[index+1]
            edge_rows[side,column] = curve['rows'][column]
            edge_valid[side,column] = True
            edge_distribution[side,:,column] = curve['distribution'][:,column]
            edge_row_valid[side,:,column] = curve['row_valid'][:,column]
            edge_source[side,column] = index+1
        reason = None
        if not center['valid'][column]:
            reason = 'center_unknown'
        elif not edge_valid[:,column].all():
            reason = 'one_or_both_sides_unknown_or_coincident'
        else:
            low, high = edge_rows[:,column].astype(float)
            c = float(center['rows'][column])
            if not low < c < high:
                reason = 'center_not_strictly_bracketed_by_edges'
            elif any(_crosses_polygon(column,low,high,p) for p in polygons):
                reason = 'cross_section_intersects_ignore_or_quality'
            else:
                first, last = int(math.floor(low)),int(math.ceil(high))
                if first<0 or last>=HEIGHT or not allowed[first:last+1,column].all():
                    reason = 'cross_section_intersects_padding_or_conservative_guard'
        if reason:
            width_reasons[reason] += 1
        else:
            width_valid[column] = True
            width_rows[column] = high-low
    native_step = float(geometry['native_px_per_row'])
    result = dict(target_distribution=center['distribution'],column_valid=center['valid'],
        row_valid=center['row_valid'],slope_target=center['slope_target'],slope_valid=center['slope_valid'],
        center_rows=center['rows'],center_valid=center['valid'].copy(),
        edge_rows=edge_rows,edge_valid=edge_valid,edge_distribution=edge_distribution,
        edge_row_valid=edge_row_valid,edge_source=edge_source,
        edge_flange_rows=flange['rows'],edge_flange_valid=flange['valid'],
        edge_surface_rows=surface['rows'],edge_surface_valid=surface['valid'],
        local_width_rows=width_rows,local_width_px=width_rows*native_step,
        local_width_free_units=width_rows*STEP_WIDTHS,width_valid=width_valid)
    result['diagnostics'] = dict(contract=TARGET_CONTRACT,
        center=dict(center['diagnostics']),
        edge_flange=dict(flange['diagnostics']),edge_surface=dict(surface['diagnostics']),
        supervised_center_columns=int(center['valid'].sum()),
        supervised_low_columns=int(edge_valid[0].sum()),supervised_high_columns=int(edge_valid[1].sum()),
        supervised_width_columns=int(width_valid.sum()),width_rejection_reasons=dict(width_reasons),
        edge_side_unassigned_columns=int(np.sum((flange['valid']|surface['valid']) & ~edge_valid.any(axis=0))),
        geometric_diameter_or_physical_confirmation=False)
    return result


def targets_for_annotation(geometry, label, quality, valid_mask):
    """Quality override applies to all three target curves, not just center."""
    curves = quality.get('supervision_polylines',label['polylines'])
    if label.get('status') not in ('clear','partial'):
        curves = {}
    result = build_boundary_targets(geometry,curves.get('centerline',[]),curves.get('edge_flange',[]),
        curves.get('edge_surface',[]),label.get('ignore_polygons',[])+quality.get('precision_exclude_polygons',[]),valid_mask)
    result['diagnostics']['reference_source'] = ('hash_bound_quality_polylines' if
        'supervision_polylines' in quality else 'original_trusted_partial_annotation')
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def freeze(output):
    """Freeze source/parameters and train-only artifact hashes before auditing."""
    if output.exists():
        raise FileExistsError('Refusing to overwrite protocol/output: '+str(output))
    manifest,index = read(TRAIN_MANIFEST),read(CACHE_INDEX)
    samples = manifest['samples']
    if ({s['sample_id'] for s in samples} != TRAIN_IDS or len(samples)!=10 or
            any(s['split']!='train' for s in samples) or '20260925' in json.dumps(samples)):
        raise ValueError('Only the fixed ten TRAIN sources are permitted')
    entries = {e['sample_id']:e for e in index['entries'] if e['split']=='train'}
    snapshots = {str(TRAIN_MANIFEST.resolve()):sha(TRAIN_MANIFEST),str(CACHE_INDEX.resolve()):sha(CACHE_INDEX)}
    selected = []
    for sample in samples:
        entry = entries[sample['sample_id']]
        if not entry['available'] or entry['image_sha256'] != sample['image_sha256']:
            raise ValueError('Missing/mismatched train input cache')
        for key in ('annotation_sha256','quality_sha256'):
            if entry.get(key)!=sample.get(key):
                raise ValueError('TRAIN snapshot mismatch: '+key)
        paths = [(sample['annotation_path'],sample['annotation_sha256']),
                 (entry['cache'],entry['cache_sha256'])]
        if sample.get('quality_sha256'):
            paths.append((sample['quality_path'],sample['quality_sha256']))
        for name,expected in paths:
            actual=sha(name)
            if actual!=expected:
                raise ValueError('Changed TRAIN source: '+name)
            snapshots[str(Path(name).resolve())]=actual
        selected.append(dict(sample=sample,cache=entry))
    sources = [Path(__file__), LAB/'tests/test_v13_boundary_targets.py',
        LAB/'v11_path_contract.py',LAB/'centerline_quality_v5.py',LAB/'centerline_quality_cutoff_v8.py',
        LAB/'centerline_review_v5.py']
    protocol = dict(contract=TARGET_CONTRACT,source_snapshot={str(p.resolve()):sha(p) for p in sources},
        artifact_snapshot=snapshots,train_records=selected,training=False,
        image_rgb_read=False,holdout_images_or_labels_read=False,blind_0925_read=False,
        freeze_before_target_audit=True,cache_use='only selected TRAIN valid_mask; no input RGB loaded',
        denominator='all 512 strip columns and separately center-valid columns; availability not predictive accuracy')
    output.mkdir(parents=True)
    write_json(output/'protocol.json',protocol)
    print(json.dumps(dict(protocol=str((output/'protocol.json').resolve()),sha256=sha(output/'protocol.json'))))


def audit(output):
    """Run support audit after the separately frozen protocol; no RGB inference."""
    from .centerline_review_v5 import validate_annotation
    from .centerline_quality_v5 import load_label_quality
    protocol_path=output/'protocol.json'
    protocol=read(protocol_path)
    if protocol['contract']!=TARGET_CONTRACT:
        raise ValueError('Target contract changed after freeze')
    for group in ('source_snapshot','artifact_snapshot'):
        for path,expected in protocol[group].items():
            if sha(path)!=expected:
                raise ValueError('Source changed after freeze: '+path)
    if (output/'rows').exists() or (output/'summary.json').exists():
        raise FileExistsError('Refusing to overwrite target audit')
    (output/'rows').mkdir()
    rows=[]
    for record in protocol['train_records']:
        sample,cache=record['sample'],record['cache']
        label=validate_annotation(read(sample['annotation_path']),sample)
        quality=load_label_quality(sample)
        if quality['quality_sha256']!=sample.get('quality_sha256'):
            raise ValueError('Quality changed after freeze')
        geometry=cache['geometry']
        rebuilt=_geometry(sample['image_shape_hw'],label['polylines']['free_seed'],cache['width']['width_px'])
        if geometry!=rebuilt:
            raise ValueError('V11 geometry differs from shape, seed and independent width')
        with np.load(cache['cache'],allow_pickle=False) as data:
            targets=targets_for_annotation(geometry,label,quality,data['valid_mask'])
            for key in ('target_distribution','column_valid','row_valid','slope_target','slope_valid'):
                if not np.array_equal(targets[key],data[key]):
                    raise ValueError('V11 center targets must stay bit-identical: '+key)
        output_npz=output/'rows'/(sample['sample_id']+'.npz')
        np.savez_compressed(output_npz,**{k:v for k,v in targets.items() if isinstance(v,np.ndarray)})
        d=targets['diagnostics']
        center=int(targets['center_valid'].sum())
        good=targets['width_valid']
        ratio=targets['local_width_free_units'][good].astype(float)
        row=dict(sample_id=sample['sample_id'],split='train',
            center_columns=center,flange_edge_columns=int(targets['edge_flange_valid'].sum()),
            surface_edge_columns=int(targets['edge_surface_valid'].sum()),
            low_columns=int(targets['edge_valid'][0].sum()),high_columns=int(targets['edge_valid'][1].sum()),
            both_edges_columns=int(targets['edge_valid'].all(axis=0).sum()),width_columns=int(good.sum()),
            width_fraction_of_all_columns=float(good.mean()),
            width_fraction_of_center_columns=float(good.sum()/center) if center else None,
            projected_width_over_independent_free_width=(dict(p05=float(np.percentile(ratio,5)),
                median=float(np.median(ratio)),p95=float(np.percentile(ratio,95))) if len(ratio) else None),
            diagnostics=d,geometry=geometry,center_targets_bit_identical_to_v11=True,
            trusted_vertex_counts={k:len(v) for k,v in quality.get('supervision_polylines',label['polylines']).items()
                                   if k in ('centerline','edge_flange','edge_surface')},
            target_npz=str(output_npz.resolve()),target_npz_sha256=sha(output_npz))
        rows.append(row)
    counts={k:sum(r[k] for r in rows) for k in ('center_columns','flange_edge_columns','surface_edge_columns',
        'low_columns','high_columns','both_edges_columns','width_columns')}
    summary=dict(protocol_sha256=sha(protocol_path),samples=10,strip_columns=WIDTH*10,counts=counts,
        width_fraction_of_all_columns=counts['width_columns']/(WIDTH*10),
        width_fraction_of_center_columns=counts['width_columns']/counts['center_columns'],
        rows=rows,training=False,physical_confirmation=False,
        warning='Label supervision support only. Width is strip-normal projected separation, not diameter. No claim of accuracy/generalization.')
    write_json(output/'summary.json',summary)
    table=['# V13 中心/双边监督可用性审计','',
        '只验证 10 张训练图既有可信标注是否支持下一轮联合监督；没有训练、预测、读留出图/标注或 9/25 数据，也没有读取 RGB。',
        '沿 V11 固定条带法向截面求真实标注线段交点，禁止延长线补边。双边按条带法向坐标低/高排序。每个中心/边界各有独立 valid；未知不是负样本。',
        'C07/C08 使用既有 hash-bound quality 的 C7 截止和可信边界；宽度必须中心及双边同时可信、中心严格夹在双边内、截面不穿忽略区/质量排除区/图像 padding。',
        '宽度是条带法向上的图像投影距离，不是钢丝绳物理直径。单边可独立监督，但不能生成宽度。', '',
        '|训练图|中心列|法兰边列|卷面边列|低侧列|高侧列|宽度列|宽度/中心|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        table.append(f"|{r['sample_id']}|{r['center_columns']}|{r['flange_edge_columns']}|{r['surface_edge_columns']}|"
            f"{r['low_columns']}|{r['high_columns']}|{r['width_columns']}|{r['width_fraction_of_center_columns']:.2%}|")
    table += ['',f"合计：{counts['width_columns']}/{counts['center_columns']} 个可信中心列有可用双边宽度监督"
        f"（{summary['width_fraction_of_center_columns']:.2%}）；占全部 5120 条带列 {summary['width_fraction_of_all_columns']:.2%}。",
        '', '中心监督五个 V11 字段逐数组完全一致。可用列高度相关，不是独立图像数量；可用性不等于模型效果或现场物理精度。',
        '协议先固定后执行，源码与训练标注/cache 哈希写于 protocol.json。每图监督数组位于 rows/*.npz；原始输入与标签未修改。']
    (output/'README.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','audit'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    (freeze if args.command=='freeze' else audit)(args.output.resolve())
