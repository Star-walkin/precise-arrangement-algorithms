"""Read-only, posthoc V24 input sampling inspection; no new inference.

GT selects display windows only. Both panels use identical native xy pixels.
The right panel back-projects the actual cached RGB; it is not a newly built
input, nor an image-quality or identity-accuracy measurement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .centerline_review_v5 import read_clean_image
from .v11_path_contract import STEP_WIDTHS, image_to_strip, strip_to_image
from .v11_path_experiment import read, sha, write_json
from .v18_front_diagnostics import font
from .v24_anchor_evaluate import MANIFEST, _safe_path, _check_json, load_inference_input
from .v24_anchor_trial import BASE, verify


SIDE = 256
DISPLAY_SCALE = 3


def sampling_record(key, entry):
    result = dict(key=key, sample_id=entry['sample_id'], split=entry['split'],
        variant=entry['variant'], available=entry['available'], reason=entry['reason'],
        optical_width_native_px=None, native_px_per_row=None,
        native_px_per_column=None, optical_width_input_pixels=None)
    if not entry['available']:
        return result
    geometry = entry['geometry']
    width = float(geometry['width_px'])
    row = float(geometry['native_px_per_row'])
    column = float(geometry['native_px_per_column'])
    if not np.isfinite([width, row, column]).all() or min(width, row, column) <= 0:
        raise ValueError('Nonpositive/nonfinite sampling scale')
    if geometry['step_widths'] != STEP_WIDTHS or not np.allclose([row, column], STEP_WIDTHS*width, atol=1e-12, rtol=0):
        raise ValueError('Cache does not obey the frozen width-normalized grid')
    result.update(optical_width_native_px=width, native_px_per_row=row,
        native_px_per_column=column, optical_width_input_pixels=width/row)
    return result


def distribution(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(count=0, minimum=None, median=None, maximum=None)
    return dict(count=len(values), minimum=float(values.min()),
        median=float(np.median(values)), maximum=float(values.max()))


def summarize(rows):
    result = []
    for variant in ('manual', 'auto'):
        for split in ('all', 'train', 'holdout'):
            selected = [r for r in rows if r['variant'] == variant and (split == 'all' or r['split'] == split)]
            available = [r for r in selected if r['available']]
            result.append(dict(variant=variant, split=split, total=len(selected), available=len(available),
                unavailable=len(selected)-len(available), **{field: distribution([r[field] for r in available])
                for field in ('optical_width_native_px', 'native_px_per_row', 'native_px_per_column', 'optical_width_input_pixels')}))
    return result


def viewport_bounds(center_xy, shape_hw, side=SIDE):
    center = np.asarray(center_xy, float)
    height, width = shape_hw
    if center.shape != (2,) or not np.isfinite(center).all() or side < 2 or min(height, width) < side:
        raise ValueError('Invalid display center/shape/window')
    x0, y0 = np.floor(center).astype(int)-side//2
    x0, y0 = int(np.clip(x0, 0, width-side)), int(np.clip(y0, 0, height-side))
    return [x0, y0, x0+side, y0+side]


def compare_same_region(source_rgb, cached_rgb_chw, valid_mask, geometry, bounds):
    """Original integer xy pixels versus actual cache at the same integer xy.

    No GT argument exists. Display support is not model confidence. Bilinear
    upsampling cannot recover information absent from the stored RGB channels.
    """
    source = np.asarray(source_rgb)
    cached = np.asarray(cached_rgb_chw)
    valid = np.asarray(valid_mask)
    x0, y0, x1, y1 = bounds
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError('Expected original uint8 RGB image')
    if geometry.get('source_shape_hw') != list(source.shape[:2]):
        raise ValueError('Cache geometry source shape differs from original image')
    if cached.shape != (3, *valid.shape) or not np.isfinite(cached).all():
        raise ValueError('Expected actual cached RGB and matching valid mask')
    if not (0 <= x0 < x1 <= source.shape[1] and 0 <= y0 < y1 <= source.shape[0]):
        raise ValueError('Display bounds outside original image')
    gy, gx = np.indices((y1-y0, x1-x0), dtype=np.float64)
    xy = np.stack([gx+x0, gy+y0], axis=-1)
    uv = image_to_strip(xy, geometry)
    roundtrip_error = float(np.max(np.abs(strip_to_image(uv, geometry)-xy)))
    if roundtrip_error > 1e-7:
        raise ValueError('Native xy/strip mapping does not round-trip')
    map_x, map_y = uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
    back = cv2.remap(cached.transpose(1, 2, 0).astype(np.float32), map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_values = cv2.remap(valid.astype(np.float32), map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    supported = ((uv[..., 0] >= 0) & (uv[..., 0] < valid.shape[1]-1) &
        (uv[..., 1] >= 0) & (uv[..., 1] < valid.shape[0]-1) & (mask_values >= 1-1e-6))
    recovered = np.rint(np.clip(back, 0, 1)*255).astype(np.uint8)
    recovered[~supported] = 90
    native = source[y0:y1, x0:x1].copy()
    diagnostic = dict(bounds_xyxy_exclusive=list(bounds), native_integer_pixel_centers=True,
        both_panels_same_xy_grid=True, native_xy_strip_roundtrip_max_error_px=roundtrip_error,
        viewport_native_shape_hw=list(native.shape[:2]), supported_native_pixels=int(supported.sum()),
        total_native_pixels=int(supported.size), supported_fraction=float(supported.mean()),
        unsupported_cache_display='neutral gray RGB(90,90,90), never interpreted as observed texture',
        cache_backprojection='cv2.INTER_LINEAR from actual stored input[0:3], image_to_strip(xy)',
        geometry_used_for_both_panels='same native integer xy window; no rotation or region mismatch')
    return native, recovered, supported, diagnostic


def render_pair(native, recovered, sample_id, measurement, mapping, center):
    side = native.shape[0]*DISPLAY_SCALE
    image = Image.new('RGB', (2*side+60, side+296), '#141d29')
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), sample_id, font=font(24), fill='white')
    draw.text((20, 48), 'GT只选择事后显示窗口；不是推理输入，不是新预测，也不证明细纹是失败唯一原因。', font=font(23), fill='#ffd58a')
    draw.text((20, 84), '左：原图整数像素 RGB', font=font(25), fill='white')
    draw.text((side+40, 84), '右：实际 auto 缓存 RGB 回映射至相同区域', font=font(25), fill='white')
    for x, pixels in ((20, native), (side+40, recovered)):
        image.paste(Image.fromarray(pixels).resize((side, side), Image.Resampling.NEAREST), (x, 125))
    y = side+143
    draw.text((20, y), f"两侧同一原图窗口 {mapping['bounds_xyxy_exclusive']}，统一 nearest ×{DISPLAY_SCALE}；右侧双线性回映射。", font=font(21), fill='#d4e2ed')
    draw.text((20, y+33), f"GT偏离点({center[0]:.2f}, {center[1]:.2f})；光学宽 {measurement['optical_width_native_px']:.3f}px；每输入行 {measurement['native_px_per_row']:.3f} 原像素。", font=font(21), fill='#d4e2ed')
    draw.text((20, y+66), f"归一化代表宽度 = {measurement['optical_width_input_pixels']:.6f} 输入像素；窗口实测输入支持 {mapping['supported_fraction']*100:.2f}%；无锐化、无色调增强。", font=font(21), fill='#d4e2ed')
    draw.text((20, y+99), '右图含本次回映射插值，不能把全部视觉变糊归因原采样；仅展示RGB，不代表全部8个输入通道。', font=font(21), fill='#ffd58a')
    return image


def inspect_sampling(base, output=None):
    base = _safe_path(base).resolve()
    output = _safe_path(output).resolve() if output is not None else base/'sampling_inspection'
    if output.exists():
        raise FileExistsError('Posthoc diagnostics are immutable; do not overwrite')
    protocol = read(base/'protocol.json'); _check_json(protocol)
    verify(base)
    index_path, result_path = base/'cache/input_index.json', base/'evaluation/results.json'
    index, results, manifest = read(index_path), read(result_path), read(MANIFEST)
    for value in (index, manifest):
        _check_json(value)
    protocol_hash = sha(base/'protocol.json')
    if index['protocol_sha256'] != protocol_hash or results['protocol_sha256'] != protocol_hash:
        raise ValueError('Input/evaluation protocol binding changed')
    hashes = {str(path.resolve()): sha(path) for path in (base/'protocol.json', index_path, result_path, MANIFEST)}
    # All source image and cache paths are validated before reading their bytes.
    samples = {s['sample_id']: s for s in manifest['samples']}
    if len(samples) != 18:
        raise ValueError('Expected fixed 18-sample manifest')
    entries = {key: index['entries'][key] for key in index['evaluation_keys']
               if index['entries'][key]['variant'] in ('manual', 'auto')}
    if len(entries) != 36 or {(e['sample_id'], e['variant']) for e in entries.values()} != {
        (sid, variant) for sid in samples for variant in ('manual', 'auto')}:
        raise ValueError('Expected manual and auto for every fixed sample')
    selected = sorted((key for key, e in entries.items() if e['split'] == 'holdout' and e['variant'] == 'auto' and e['available']),
        key=lambda key: entries[key]['sample_id'])[:2]
    if len(selected) != 2:
        raise ValueError('Need two available auto holdout samples')
    stats, selected_arrays = [], {}
    for key, entry in entries.items():
        if entry.get('original_image_sha256') != samples[entry['sample_id']]['image_sha256']:
            raise ValueError('Cache RGB source image differs from fixed manifest')
        arrays = load_inference_input(entry, hashes)
        stats.append(sampling_record(key, entry))
        if key in selected:
            selected_arrays[key] = arrays
    records, figures = [], []
    for key in selected:
        entry = entries[key]; sid = entry['sample_id']; sample = samples[sid]
        _safe_path(sample['image_path'])
        source = np.asarray(read_clean_image(sample))
        hashes[str(Path(sample['image_path']).resolve())] = sha(sample['image_path'])
        row = next(r for r in results['rows'] if r['sample_id'] == sid and r['arm'] == 'baseline' and r['variant'] == 'auto')
        center = row['GT_departure'].get('point_xy')
        if center is None:
            records.append(dict(sample_id=sid, status='unknown_GT_departure_no_image', used_for_inference=False))
            continue
        arrays = selected_arrays[key]
        bounds = viewport_bounds(center, source.shape[:2])
        native, recovered, valid, mapping = compare_same_region(source, arrays['input'][:3], arrays['valid_mask'], entry['geometry'], bounds)
        measurement = next(r for r in stats if r['key'] == key)
        records.append(dict(sample_id=sid, key=key, status='rendered_same_native_region',
            GT_departure_point_xy=center, GT_used_only_to_select_display_viewport=True,
            source_image_path=sample['image_path'], source_image_sha256=sample['image_sha256'],
            input_path=entry['input_path'], input_sha256=entry['input_sha256'],
            geometry=entry['geometry'], sampling=measurement, mapping=mapping,
            new_prediction=False, used_for_inference=False))
        figures.append((sid, native, recovered, valid, render_pair(native, recovered, sid, measurement, mapping, center)))
    verify(base)
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('Source changed during sampling inspection')
    output.mkdir(exist_ok=False)
    artifact_rows = []
    for sid, native, recovered, valid, panel in figures:
        for suffix, content in (('source_native_crop', Image.fromarray(native)),
                                ('cache_on_same_native_grid', Image.fromarray(recovered)),
                                ('actual_input_support', Image.fromarray(valid.astype(np.uint8)*255)),
                                ('comparison', panel)):
            path = output/(sid+'__'+suffix+'.png')
            content.save(path)
            artifact_rows.append(dict(sample_id=sid, kind=suffix, path=str(path), sha256=sha(path)))
    report = dict(schema='v24_posthoc_sampling_inspection_1', script_sha256=sha(__file__),
        source_sha256=hashes, protocol_sha256=protocol_hash,
        dependency_sha256={name: sha(Path(__file__).with_name(name)) for name in
            ('v11_path_contract.py', 'v24_anchor_evaluate.py', 'v24_anchor_trial.py', 'centerline_review_v5.py', 'v18_front_diagnostics.py')},
        rows=stats, summaries=summarize(stats), views=records, artifacts=artifact_rows,
        selection='First two available automatic-input development holdout sample IDs, lexicographic order; not selected by scores.',
        display_window_native_px=SIDE, display_scale=DISPLAY_SCALE,
        sampling_equation='native_px_per_row = width_px * STEP_WIDTHS; representative optical width / native_px_per_row = 1 / .1875',
        representative_optical_width_input_pixels=1/STEP_WIDTHS,
        interpretation='A high-resolution short-segment experiment is motivated by the coarse representative-width sampling; these two posthoc RGB comparisons do not isolate failure causality.',
        limitations='Representative free-rope optical width is not local wound-strand texture pitch. Actual input also has normalized gray/query/coordinate/valid channels not displayed here. RGB backprojection adds display interpolation and is not the network tensor grid. No learned trial or accuracy comparison performed.',
        GT_used_only_for_display=True, new_inference=False, new_training=False, threshold_changed=False,
        original_results_changed=False, hardware_access=False, blind_20260925_read=False,
        identity_confirmed=False, sole_cause_established=False, deployment_authorized=False)
    write_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, help='Optional new directory; existing outputs are never overwritten')
    args = parser.parse_args()
    report = inspect_sampling(args.base, args.output)
    import json
    print(json.dumps(dict(output=str(args.output or args.base/'sampling_inspection'), summaries=report['summaries'],
        selected=[v['sample_id'] for v in report['views']]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
