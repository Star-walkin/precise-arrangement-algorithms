"""GT-conditioned TRAIN-ONLY optical-profile/annotation consistency audit.

Not a detector, model score, calibration, or learned correction. Ground truth
chooses all cross-sections and the local edge search intervals explicitly.
No frozen source, image, label, model, or quality file is edited.
"""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .centerline_quality_v5 import load_label_quality
from .centerline_review_v5 import validate_annotation, read_clean_image
from .centerline_path_metrics_v6 import clip_scoring_segments
from .evaluate_centerline_multiscale_v9 import independent_width


LAB = Path(__file__).resolve().parent
MANIFEST = LAB / 'outputs/centerline_v9/dataset/manifest.json'
DEFAULT_OUTPUT = LAB / 'outputs/v12_training_profile_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def selected_train_samples(manifest):
    selected = [sample for sample in manifest['samples'] if sample['split'] == 'train']
    if len(selected) != 10 or any(sample['sample_id'].startswith(('C03__', 'C06__')) for sample in selected):
        raise ValueError('Expected the fixed 10 geometry-training records')
    if '20260925' in json.dumps(selected, ensure_ascii=False):
        raise ValueError('9/25 blind source prohibited')
    return selected


def arc_geometry(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite polyline with at least two vertices')
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    positions = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if positions[-1] <= 1e-8:
        raise ValueError('Degenerate reference polyline')
    return points, positions


def arc_point(points, positions, distance):
    return np.array([np.interp(distance, positions, points[:, dim]) for dim in (0, 1)])


def normal_intersections(edge, center, tangent):
    """Intersect actual labeled segments with one normal section, no extension."""
    edge = np.asarray(edge, float)
    if len(edge) < 2:
        return [], 'missing_edge_polyline'
    normal = np.array([-tangent[1], tangent[0]])
    found = []
    for first, last in zip(edge[:-1], edge[1:]):
        first_t, last_t = (float((point-center) @ tangent) for point in (first, last))
        denominator = last_t-first_t
        if abs(denominator) < 1e-9:
            if abs(first_t) < 1e-7:
                return [], 'edge_collinear_with_section'
            continue
        fraction = -first_t/denominator
        if -1e-8 <= fraction <= 1+1e-8:
            point = first + np.clip(fraction, 0, 1)*(last-first)
            coordinate = float((point-center) @ normal)
            if not any(np.linalg.norm(point-old['xy']) < 1e-6 for old in found):
                found.append(dict(xy=point, normal_px=coordinate))
    if len(found) == 0:
        return found, 'no_edge_intersection_no_extrapolation'
    if len(found) > 1:
        return found, 'multiple_edge_intersections'
    return found, None


def point_excluded(point, polygons):
    return any(cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float, point)), False) >= 0
               for polygon in polygons)


def section_excluded(first, last, polygons):
    if point_excluded(first, polygons) or point_excluded(last, polygons):
        return True
    source_length = float(np.linalg.norm(last-first))
    pieces = clip_scoring_segments([[first.tolist(), last.tolist()]], ignore_polygons=polygons)
    remaining = sum(float(np.linalg.norm(np.diff(np.asarray(piece), axis=0), axis=1).sum()) for piece in pieces)
    return remaining < source_length-1e-6


def stats(values):
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], float)
    if not len(values):
        return dict(count=0, mean=None, median=None, p05=None, p95=None, abs_median=None, abs_p95=None)
    return dict(count=int(len(values)), mean=float(values.mean()), median=float(np.median(values)),
                p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)),
                abs_median=float(np.median(np.abs(values))), abs_p95=float(np.quantile(np.abs(values), .95)))


def optical_profile(gray, unknown_mask, center, tangent, free_width, flange_tau, surface_tau):
    normal = np.array([-tangent[1], tangent[0]])
    extent = 1.5*free_width
    offsets = np.arange(-np.ceil(extent/.5), np.ceil(extent/.5)+1)*.5
    tangent_offsets = np.linspace(-.25*free_width, .25*free_width, 9)
    xy = center + offsets[None, :, None]*normal + tangent_offsets[:, None, None]*tangent
    h, w = gray.shape
    if np.any(xy[..., 0] < 0) or np.any(xy[..., 0] >= w-1) or np.any(xy[..., 1] < 0) or np.any(xy[..., 1] >= h-1):
        return dict(available=False, reason='profile_patch_outside_image')
    xmap, ymap = xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32)
    excluded = cv2.remap(unknown_mask, xmap, ymap, cv2.INTER_LINEAR)
    if np.any(excluded > 0):
        return dict(available=False, reason='profile_patch_crosses_unknown_quality')
    samples = cv2.remap(gray, xmap, ymap, cv2.INTER_LINEAR)
    raw = np.median(samples.astype(float), axis=0)
    kernel = cv2.getGaussianKernel(9, 2.)[:, 0][None]  # 0.5px samples => sigma 1 native px.
    smooth = cv2.filter2D(raw[None], -1, kernel, borderType=cv2.BORDER_REPLICATE)[0]
    derivative = np.gradient(smooth, .5)
    low, high = sorted((flange_tau, surface_tau))
    label_width = high-low
    if low < offsets[0] or high > offsets[-1]:
        return dict(available=False, reason='labeled_body_outside_fixed_profile_extent')
    inside = np.flatnonzero((offsets >= low) & (offsets <= high))
    if len(inside) < 3:
        return dict(available=False, reason='too_few_body_profile_samples')
    orientation = 1. if surface_tau > flange_tau else -1.
    bright_index, dark_index = (int(inside[np.argmax(smooth[inside])]), int(inside[np.argmin(smooth[inside])]))
    dynamic_range = float(smooth[inside].max()-smooth[inside].min())
    bright_support = inside[smooth[inside] >= smooth[bright_index]-2.]
    dark_support = inside[smooth[inside] <= smooth[dark_index]+2.]
    bright_span = float(offsets[bright_support].max()-offsets[bright_support].min())
    dark_span = float(offsets[dark_support].max()-offsets[dark_support].min())
    edge_results = []
    for edge_tau in (flange_tau, surface_tau):
        nearby = np.flatnonzero(np.abs(offsets-edge_tau) <= .3*label_width)
        if len(nearby) < 3:
            return dict(available=False, reason='insufficient_gt_conditioned_edge_search_interval')
        gradient_index = int(nearby[np.argmax(np.abs(derivative[nearby]))])
        valley_index = int(nearby[np.argmin(smooth[nearby])])
        edge_results.append(dict(annotated_tau_px=float(edge_tau), gradient_tau_px=float(offsets[gradient_index]),
            valley_tau_px=float(offsets[valley_index]), gradient_abs=float(abs(derivative[gradient_index])),
            gradient_at_search_border=bool(gradient_index in (nearby[0], nearby[-1])),
            valley_at_search_border=bool(valley_index in (nearby[0], nearby[-1]))))
    gradient_mid = .5*sum(edge['gradient_tau_px'] for edge in edge_results)
    valley_mid = .5*sum(edge['valley_tau_px'] for edge in edge_results)
    return dict(available=True, gt_conditioned_not_prediction=True,
        offsets_px=offsets.tolist(), raw_gray=raw.tolist(), smoothed_gray=smooth.tolist(),
        gt_center_gray=float(np.interp(0, offsets, smooth)),
        geometric_midpoint_gray=float(np.interp(.5*(flange_tau+surface_tau), offsets, smooth)),
        body_gray_dynamic_range=dynamic_range, photometric_contrast_at_least_5=dynamic_range >= 5.,
        bright_offset_px=float(offsets[bright_index]*orientation),
        bright_offset_in_labeled_width=float(offsets[bright_index]*orientation/label_width),
        dark_offset_px=float(offsets[dark_index]*orientation),
        dark_offset_in_labeled_width=float(offsets[dark_index]*orientation/label_width),
        bright_near_max_span_in_labeled_width=bright_span/label_width,
        dark_near_min_span_in_labeled_width=dark_span/label_width,
        bright_peak_narrow=bool(dynamic_range >= 5 and bright_span <= .3*label_width),
        dark_peak_narrow=bool(dynamic_range >= 5 and dark_span <= .3*label_width),
        gt_conditioned_gradient_mid_offset_in_labeled_width=gradient_mid*orientation/label_width,
        gt_conditioned_valley_mid_offset_in_labeled_width=valley_mid*orientation/label_width,
        edge_searches=edge_results, optical_search_was_gt_conditioned=True,
        warning='bright/dark extrema may be strand texture or glare; edge windows use labels, not a deployable detector')


def summarize_sections(sections):
    geom = [row for row in sections if row.get('geometry_available')]
    photo = [row for row in geom if row.get('profile', {}).get('available')]
    contrast = [row for row in photo if row['profile']['photometric_contrast_at_least_5']]
    narrow_bright = [row for row in photo if row['profile']['bright_peak_narrow']]
    narrow_dark = [row for row in photo if row['profile']['dark_peak_narrow']]
    return dict(total_sampled_sections=len(sections), geometry_sections=len(geom),
        profile_sections=len(photo), contrast_ge5_sections=len(contrast), narrow_bright_sections=len(narrow_bright),
        narrow_dark_sections=len(narrow_dark),
        exclusion_counts=dict(Counter(row['reason'] for row in sections if not row.get('geometry_available'))),
        profile_exclusion_counts=dict(Counter(row['profile']['reason'] for row in geom if not row['profile']['available'])),
        center_outside_labeled_edges=sum(not row['center_inside_edges'] for row in geom),
        geometric_mid_minus_manual_center_px=stats([row['midpoint_minus_center_px'] for row in geom]),
        geometric_mid_minus_manual_center_in_labeled_width=stats([row['midpoint_minus_center_in_labeled_width'] for row in geom]),
        labeled_width_over_independent_free_width=stats([row['labeled_width_over_free_width'] for row in geom]),
        bright_offset_all_profiles_in_labeled_width=stats([row['profile']['bright_offset_in_labeled_width'] for row in photo]),
        dark_offset_all_profiles_in_labeled_width=stats([row['profile']['dark_offset_in_labeled_width'] for row in photo]),
        bright_offset_narrow_peaks_in_labeled_width=stats([row['profile']['bright_offset_in_labeled_width'] for row in narrow_bright]),
        dark_offset_narrow_peaks_in_labeled_width=stats([row['profile']['dark_offset_in_labeled_width'] for row in narrow_dark]),
        gt_conditioned_gradient_mid_offset_in_labeled_width=stats([
            row['profile']['gt_conditioned_gradient_mid_offset_in_labeled_width'] for row in photo]),
        gt_conditioned_valley_mid_offset_in_labeled_width=stats([
            row['profile']['gt_conditioned_valley_mid_offset_in_labeled_width'] for row in photo]),
        correlated_sections_not_independent_trials=True)


def audit_sample(sample):
    annotation_sha = sha(sample['annotation_path'])
    if sample.get('annotation_sha256') != annotation_sha:
        raise ValueError('Training annotation snapshot changed')
    label = validate_annotation(json.loads(Path(sample['annotation_path']).read_text('utf-8-sig')), sample)
    quality = load_label_quality(sample)
    if quality['quality_sha256'] != sample.get('quality_sha256'):
        raise ValueError('Training quality snapshot changed')
    trusted = quality.get('supervision_polylines')
    curves = dict(label['polylines'])
    if trusted is not None:
        curves.update(trusted)
    if sample['sample_id'].startswith(('C07__', 'C08__')) and (trusted is None or len(trusted['centerline']) != 7):
        raise ValueError('C07/C08 require confirmed C1-C7 trusted target geometry')
    rgb = np.asarray(read_clean_image(sample))
    image_bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    free_width_result = independent_width(image_bgr, label['polylines']['free_seed'])
    record = dict(sample_id=sample['sample_id'], group_id=sample['group_id'], split='train',
        annotation_sha256=annotation_sha, image_sha256=sample['image_sha256'], quality_sha256=quality['quality_sha256'],
        annotation_notes=label.get('notes', ''), status=label['status'],
        trusted_source='confirmed_trusted_polylines' if trusted is not None else 'original_polylines_with_quality_exclusions',
        counts={key: len(curves[key]) for key in ('centerline', 'edge_flange', 'edge_surface')},
        independent_width=free_width_result, excluded_quality_regions=quality['regions'])
    if free_width_result['width_px'] is None:
        return dict(record, available=False, reason='independent_rgb_free_width_unavailable', sections=[])
    free_width = free_width_result['width_px']
    points, positions = arc_geometry(curves['centerline'])
    polygons = label['ignore_polygons'] + quality['precision_exclude_polygons']
    unknown_mask = np.zeros(rgb.shape[:2], np.uint8)
    for polygon in polygons:
        cv2.fillPoly(unknown_mask, [np.rint(polygon).astype(np.int32)], 255)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    seed = np.asarray(label['polylines']['free_seed'], float)
    seed_direction = seed[-1]-seed[0]
    seed_direction /= np.linalg.norm(seed_direction)
    spacing = max(12., free_width*.5)
    distance_samples = np.arange(0., positions[-1]+1e-6, spacing)
    sections = []
    for distance in distance_samples:
        center = arc_point(points, positions, distance)
        tangent = arc_point(points, positions, min(positions[-1], distance+free_width*.5)) - arc_point(
            points, positions, max(0., distance-free_width*.5))
        magnitude = np.linalg.norm(tangent)
        row = dict(arc_distance_px=float(distance), arc_fraction=float(distance/positions[-1]), center_xy=center.tolist(),
            zone='seed_downstream_half_plane' if (center-seed[-1])@seed_direction >= 0 else 'seed_upstream_half_plane')
        if magnitude <= 1e-8:
            sections.append(dict(row, geometry_available=False, reason='degenerate_center_tangent')); continue
        tangent /= magnitude
        row['tangent_xy'] = tangent.tolist()
        if point_excluded(center, polygons):
            sections.append(dict(row, geometry_available=False, reason='center_in_unknown_quality')); continue
        intersections, errors = [], []
        for key in ('edge_flange', 'edge_surface'):
            found, error = normal_intersections(curves[key], center, tangent)
            intersections.append(found)
            errors.append(key + ':' + error if error else None)
        if any(errors):
            sections.append(dict(row, geometry_available=False, reason=';'.join(error for error in errors if error))); continue
        flange, surface = intersections[0][0], intersections[1][0]
        width = abs(surface['normal_px']-flange['normal_px'])
        if width <= 1e-6:
            sections.append(dict(row, geometry_available=False, reason='degenerate_labeled_width')); continue
        if section_excluded(flange['xy'], surface['xy'], polygons):
            sections.append(dict(row, geometry_available=False, reason='labeled_cross_section_intersects_unknown_quality')); continue
        orientation = np.sign(surface['normal_px']-flange['normal_px'])
        midpoint_offset = .5*(surface['normal_px']+flange['normal_px'])*orientation
        row.update(geometry_available=True, flange_xy=flange['xy'].tolist(), surface_xy=surface['xy'].tolist(),
            flange_normal_px=flange['normal_px'], surface_normal_px=surface['normal_px'],
            center_inside_edges=min(flange['normal_px'], surface['normal_px']) <= 0 <= max(flange['normal_px'], surface['normal_px']),
            labeled_width_px=float(width), labeled_width_over_free_width=float(width/free_width),
            midpoint_minus_center_px=float(midpoint_offset), midpoint_minus_center_in_labeled_width=float(midpoint_offset/width),
            profile=optical_profile(gray, unknown_mask, center, tangent, free_width, flange['normal_px'], surface['normal_px']))
        sections.append(row)
    return dict(record, available=True, sample_spacing_px=spacing, sections=sections,
                all_sections_summary=summarize_sections(sections),
                downstream_summary=summarize_sections([row for row in sections if row['zone'] == 'seed_downstream_half_plane']))


def profile_figure(records, path):
    """Fixed sample IDs and quartiles, not best-looking/least-error selection."""
    wanted = ('dual_20260924_111500__left__f00246', 'dual_20260924_111143__right__f00130', 'C07__', 'C08__')
    selected = []
    for key in wanted:
        record = next((r for r in records if r['sample_id'] == key or r['sample_id'].startswith(key)), None)
        if record:
            selected.append(record)
    panel_w, panel_h = 630, 300
    canvas = Image.new('RGB', (panel_w*2, 70+panel_h*len(selected)), '#111827')
    draw = ImageDraw.Draw(canvas)
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    f = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    draw.text((12, 12), 'GT-conditioned training profiles, NOT detector outputs', fill='white', font=f)
    draw.text((12, 34), 'Gray/blue: raw/smoothed gray 0..255 | Red: manual center | Green: human edges | Gold: human-edge midpoint', fill='white', font=f)
    for row_index, record in enumerate(selected):
        valid = [row for row in record['sections'] if row.get('profile', {}).get('available')]
        for column, fraction in enumerate((.4, .75)):
            left, top = column*panel_w, 70+row_index*panel_h
            draw.text((left+10, top+5), record['sample_id'][:65], fill='white', font=f)
            if not valid:
                draw.text((left+10, top+30), 'No fully trusted profile', fill='#ff8a8a', font=f); continue
            row = min(valid, key=lambda r: abs(r['arc_fraction']-fraction))
            profile = row['profile']; free_width = record['independent_width']['width_px']
            offsets = np.array(profile['offsets_px'])/free_width
            x0, x1, y0, y1 = left+44, left+610, top+265, top+66
            def pixel(x, y):
                return (x0+(x+1.5)/3*(x1-x0), y0-y/255*(y0-y1))
            draw.rectangle([x0, y1, x1, y0], outline='#64748b')
            for key, color in (('raw_gray', '#9ca3af'), ('smoothed_gray', '#60a5fa')):
                draw.line([pixel(x, y) for x, y in zip(offsets, profile[key])], fill=color, width=2)
            for offset, color in ((0., '#ff5555'), (row['flange_normal_px']/free_width, '#4aff79'),
                                  (row['surface_normal_px']/free_width, '#4aff79'),
                                  ((row['flange_normal_px']+row['surface_normal_px'])/(2*free_width), '#ffd04c')):
                draw.line([pixel(offset, 0), pixel(offset, 255)], fill=color, width=2)
            draw.text((left+10, top+28), f'arc {row["arc_fraction"]:.2f}; labeled/free width {row["labeled_width_over_free_width"]:.2f}; '
                      f'mid-center {row["midpoint_minus_center_in_labeled_width"]:+.3f} labeled widths', fill='white', font=f)
            draw.text((x0, y0+7), '-1.5                  Normal offset / free optical width                  +1.5', fill='#cbd5e1', font=f)
    with path.open('xb') as stream:
        canvas.save(stream, format='PNG')


def self_test():
    center, tangent = np.array([5., 0.]), np.array([1., 0.])
    one, error = normal_intersections([[0., -3.], [10., -3.]], center, tangent)
    two, second_error = normal_intersections([[0., 5.], [10., 5.]], center, tangent)
    assert error is None and second_error is None
    assert one[0]['normal_px'] == -3 and two[0]['normal_px'] == 5
    assert .5*(one[0]['normal_px']+two[0]['normal_px']) == 1
    assert normal_intersections([[6., -3.], [10., -3.]], center, tangent)[1] == 'no_edge_intersection_no_extrapolation'
    assert normal_intersections([[5., -3.], [5., 3.]], center, tangent)[1] == 'edge_collinear_with_section'
    assert normal_intersections([[0., -1.], [10., 0.], [0., 1.]], center, tangent)[1] == 'multiple_edge_intersections'
    assert section_excluded(np.array([0., 0.]), np.array([10., 0.]), [[[4., -1.], [6., -1.], [6., 1.], [4., 1.]]])
    assert not section_excluded(np.array([0., 0.]), np.array([3., 0.]), [[[4., -1.], [6., -1.], [6., 1.], [4., 1.]]])
    print('Synthetic section geometry, no extrapolation, ambiguity and quality crossing checks passed')


def run(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Training audit output already exists; do not overwrite')
    manifest = json.loads(MANIFEST.read_text('utf-8'))
    samples = selected_train_samples(manifest)  # Select BEFORE reading any sample file.
    output.mkdir(parents=True)
    protocol = dict(schema_version=1, source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
        manifest_sha256=sha(MANIFEST), selected_ids=[sample['sample_id'] for sample in samples],
        selected_snapshots={sample['sample_id']: {key: sample[key] for key in
            ('image_sha256', 'annotation_sha256', 'quality_sha256')} for sample in samples},
        status='predeclared training-only diagnostic', input_is_gt_conditioned=True, prediction_accuracy_claim=False,
        centerline='confirmed three trusted polylines if provided, otherwise original minus quality regions',
        cross_sections='normal to centered arc chord of +/-0.5 independently measured free optical width',
        spacing_native_px='max(12, 0.5 * independent free optical width)',
        exclusions='unknown center/edge/cross-section; zero/multiple/collinear edge intersections; no edge extrapolation',
        profile='normal +/-1.5 free widths at 0.5 native-pixel step, tangent median of 9 samples across 0.5 free width',
        smoothing='Gaussian sigma1 native px; fixed 9 samples kernel at 0.5px',
        edge_search='GT-conditioned +/-0.3 labeled width around each annotated edge: max abs gradient and darkest point',
        bright_dark='global maximum/minimum within the labeled body; 2-gray-level support span reported',
        narrow_peak_diagnostic='body range>=5 gray levels AND near-extremum span<=0.3 labeled width; never relabel unknown',
        sign='positive points from flange-side edge toward other/surface-side edge',
        interpretation='apparent width and photometric bias, not physical diameter/calibration; correlated sections',
        holdout_content_opened=False, future_blind_content_opened=False, model_training=False, hardware_access=False)
    write_json(output/'protocol.json', protocol)
    records = []
    for sample in samples:
        record = audit_sample(sample)
        records.append(record)
        print(json.dumps(dict(sample_id=record['sample_id'], available=record['available'],
            geometry_sections=record.get('all_sections_summary', {}).get('geometry_sections', 0),
            profile_sections=record.get('all_sections_summary', {}).get('profile_sections', 0)), ensure_ascii=False), flush=True)
    all_sections = [row for record in records for row in record['sections']]
    downstream = [row for row in all_sections if row['zone'] == 'seed_downstream_half_plane']
    summary = dict(protocol_sha256=sha(output/'protocol.json'), images_requested=len(samples),
        images_available=sum(record['available'] for record in records),
        all_sections=summarize_sections(all_sections), downstream=summarize_sections(downstream),
        per_image_downstream_median_geometry_bias=stats([
            record.get('downstream_summary', {}).get('geometric_mid_minus_manual_center_in_labeled_width', {}).get('median')
            for record in records]),
        per_image_downstream_median_bright_bias=stats([
            record.get('downstream_summary', {}).get('bright_offset_all_profiles_in_labeled_width', {}).get('median')
            for record in records]),
        gt_conditioned_not_prediction=True, universal_correction_fitted=False,
        warning='No statistical independence or physical truth inferred from linearly interpolated human geometry')
    for sample in samples:
        check = load_label_quality(sample)
        if sha(sample['annotation_path']) != sample['annotation_sha256'] or check['quality_sha256'] != sample['quality_sha256']:
            raise ValueError('Training annotation or quality changed during audit')
        if sha(sample['image_path']) != sample['image_sha256']:
            raise ValueError('Training image changed during audit')
    if sha(__file__) != protocol['source_sha256']:
        raise ValueError('Audit source changed during execution')
    write_json(output/'records.json', records)
    write_json(output/'summary.json', summary)
    profile_figure(records, output/'fixed_training_profiles.png')
    print(json.dumps(dict(summary=str(output/'summary.json'), downstream=summary['downstream']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.output)
