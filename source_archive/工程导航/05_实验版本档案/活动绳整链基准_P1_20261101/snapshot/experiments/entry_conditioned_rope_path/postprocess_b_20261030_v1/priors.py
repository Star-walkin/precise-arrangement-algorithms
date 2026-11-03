"""Read-only F3/A6/A7 adapters for UNet B postprocessing.

All persisted geometry and widths use ORIGINAL image pixels. Arrays ending in
``_half`` use the FULL original image reduced by two, not the cropped model ROI.
These are auxiliary measurements, never labels or proof of same-rope identity.
No legacy detection pipeline is imported, and no source file is modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / '数据资产/05_派生与实验结果'
DATA = ASSETS / 'camera_condition_images_20261009_v2'
A6 = ASSETS / 'camera_condition_incoming_template_20261029_v6/all_001'
A7 = ASSETS / 'camera_condition_free_straight_20261030_v7/all_001'
A2 = ASSETS / 'camera_condition_incoming_rope_20261026_v2/all_001'
F3 = ASSETS / 'camera_condition_full_surface_joint_20261028_v3/all_001'
C4 = ASSETS / 'camera_condition_full_surface_20261026_v7/all_001'
C3 = ASSETS / 'camera_condition_grouped_conic_20261026_v6/all_001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=1)
def _allowed_inputs():
    with (DATA / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 400:
        raise ValueError('Expected the fixed open development400 manifest')
    lookup = {}
    for row in rows:
        relative = Path(row['output_file'])
        if (relative.is_absolute() or '..' in relative.parts or
                len(relative.parts) != 3 or relative.parts[0] != 'images' or
                relative.parts[1] != row['camera_role'] + '_camera'):
            raise ValueError('Unsafe original image path')
        if ('20260925' in row['source_file'] + row['output_file'] or
                any(s in row['dataset_use_status'].lower() for s in ('reserved', 'heldout'))):
            raise ValueError('Reserved data are not allowed')
        if relative.stem in lookup:
            raise ValueError('Duplicate original image identifier')
        lookup[relative.stem] = row
    return lookup


def _image(path, mode):
    out = cv2.imdecode(np.fromfile(str(path), np.uint8), mode)
    if out is None:
        raise ValueError('Cannot decode ' + str(path))
    return out


def _half_boundary(mask):
    """Max pooling preserves a one-pixel F3 line when reducing resolution."""
    h, w = mask.shape
    if h % 2 or w % 2:
        raise ValueError('A factor-two boundary adapter requires even dimensions')
    return mask.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))


def load_priors(job):
    """Return full-image half-resolution arrays and original-coordinate records.

    Expected job fields: id,image_path,image_sha256,shape,seeds. Optional
    ``seed_source_hashes`` receipts from preparation are checked when present.
    Original and every prior actually read are listed in input_dependency_sha256.
    No F3 exclusion mask is loaded: old rope occlusion must not forbid this rope.
    """
    row = _allowed_inputs().get(job['id'])
    if row is None:
        raise ValueError('Job is outside the fixed open400 manifest')
    original_path = DATA / row['output_file']
    if Path(job['image_path']).resolve() != original_path.resolve():
        raise ValueError('Job original path differs from the manifest')
    role = row['camera_role'] + '_camera'
    name = Path(row['output_file']).name
    stem = Path(name).stem
    paths = {
        'original': original_path,
        'surface': C4 / 'roi/surface_masks' / role / name,
        'merged': F3 / '记录/masks/merged' / role / name,
        'inferred': F3 / '记录/masks/inferred' / role / name,
        'a6': A6 / '记录' / role / (stem + '.json'),
        'a7': A7 / '记录' / role / (stem + '.json'),
        'pitch': C3 / 'boundaries/records' / role / (stem + '.json'),
    }
    hashes = {str(path): sha(path) for path in paths.values()}
    hashes[str(DATA / 'manifest.csv')] = sha(DATA / 'manifest.csv')
    if hashes[str(original_path)] != row['output_sha256'] or hashes[str(original_path)] != job['image_sha256']:
        raise ValueError('Original hash differs from job or open400 manifest')
    a6 = _json(paths['a6'])['result']
    a7 = _json(paths['a7'])['result']
    # The three historical free-only seeds came from A2 instead of A6. Keep
    # their exact recorded origin; never silently replace the UNet entry.
    source_records = {str(paths['a6'].resolve()): {'kind': 'A6', 'data': a6}}
    allowed_source_paths = {paths['a6'].resolve(), (A2 / 'records' / role / (stem + '.json')).resolve()}
    receipts = job.get('seed_source_hashes', job.get('seed_sources', []))
    expected_sources = {str(Path(x['path']).resolve()): x['sha256'] for x in receipts}
    for seed in job.get('seeds', []):
        source = Path(seed['source']).resolve()
        if source not in allowed_source_paths:
            raise ValueError('Seed source is not this image A6/A2 automatic record')
        key = str(source)
        digest = sha(source)
        hashes[key] = digest
        if key in expected_sources and expected_sources[key] != digest:
            raise ValueError('Prepared seed source hash changed')
        if key not in source_records:
            source_records[key] = {'kind': 'A2', 'data': _json(source)}
    image = _image(original_path, cv2.IMREAD_COLOR)
    shape = tuple(int(v) for v in job['shape'])
    if image.shape[:2] != shape:
        raise ValueError('Original dimensions differ from the prepared job')
    masks = {k: _image(paths[k], cv2.IMREAD_GRAYSCALE) for k in ('surface', 'merged', 'inferred')}
    if any(a.shape != shape for a in masks.values()):
        raise ValueError('Prior mask dimensions differ from original')
    pitch = float(_json(paths['pitch'])['record']['pitch_px'])
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError('Invalid saved pixel pitch')
    half_size = (shape[1] // 2, shape[0] // 2)
    observed = np.where((masks['merged'] > 0) & (masks['inferred'] == 0), 255, 0).astype(np.uint8)
    return {
        'original_half': cv2.resize(image, half_size, interpolation=cv2.INTER_AREA),
        'surface_half': cv2.resize(masks['surface'], half_size, interpolation=cv2.INTER_NEAREST),
        'f3_merged_half': _half_boundary(masks['merged']),
        'f3_inferred_half': _half_boundary(masks['inferred']),
        'f3_observed_half': _half_boundary(observed),
        'a6_entries': a6.get('entries', []), 'a6_status': a6.get('status'),
        'a7_candidates': a7.get('candidates', []), 'a7_status': a7.get('status'),
        'seed_source_records': source_records,
        'pitch': pitch, 'pitch_px': pitch, 'pitch_half_px': pitch / 2.,
        'original_shape': list(shape), 'half_shape': [half_size[1], half_size[0]],
        'coordinate_scale': .5, 'geometry_coordinate_system': 'original_image_xy_pixels',
        'camera_role': row['camera_role'], 'image_id': stem,
        'input_dependency_sha256': hashes,
        'array_resampling': {'original': 'area', 'surface': 'nearest', 'F3': '2x2_max_preserves_thin_lines'},
        'labels_read': False, 'exclusion_used_as_forbidden_region': False,
    }


def _axis_match(seed_xy, origin, direction, width):
    """Undirected line agreement; matching cannot relocate the supplied seed."""
    p = np.asarray(seed_xy, float).reshape(-1, 2)
    d = p[-1] - p[0]
    d /= max(float(np.linalg.norm(d)), 1e-9)
    axis = np.asarray(direction, float).reshape(2)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    if not np.isfinite(np.r_[p.ravel(), axis, origin, width]).all() or np.linalg.norm(axis) < .9:
        return {'compatible': False, 'reason': 'invalid_saved_geometry'}
    angle = math.degrees(math.acos(float(np.clip(abs(d @ axis), 0., 1.))))
    normal = np.array([-axis[1], axis[0]])
    distances = np.abs((p - np.asarray(origin)) @ normal)
    offset = float(np.median(distances))
    return {'compatible': bool(angle <= 6. and offset <= max(6., 2. * width)),
            'undirected_angle_deg': angle, 'perpendicular_distance_px': offset,
            'perpendicular_distance_widths': offset / max(width, 1.),
            'matching_only_not_identity_probability': True}


def geometry_for_seed(priors, seed):
    """Keep the exact UNet seed; provide compatible original-pixel width hints.

    width_px is ORIGINAL resolution; width_half_px is its half-size equivalent.
    direction_xy/origin_xy/seed_xy describe the unchanged UNet seed. Saved A6
    and A7 axes are separate optional hints, never substituted as a new entry.
    A6 same-entry geometry takes precedence, A7 supplies an independent hint.
    """
    xy = np.asarray(seed['xy'], float).reshape(-1, 2)
    if len(xy) < 2 or not np.isfinite(xy).all() or np.linalg.norm(xy[-1] - xy[0]) < 1.:
        raise ValueError('Expected a finite nonzero automatic UNet seed')
    direction = xy[-1] - xy[0]
    direction /= np.linalg.norm(direction)
    matches = []
    for entry in priors.get('a6_entries', []):
        g = entry.get('free', {}).get('geometry', {})
        if not all(k in g for k in ('origin_xy', 'direction_xy', 'width_px')):
            continue
        match = _axis_match(xy, g['origin_xy'], g['direction_xy'], float(g['width_px']))
        if match['compatible']:
            same_id = entry.get('entry_id') == seed.get('entry_id')
            cost = match['perpendicular_distance_widths'] + match['undirected_angle_deg'] / 3.
            matches.append((0 if same_id else 1, cost, entry, match))
    matches.sort(key=lambda item: (item[0], item[1]))
    selected = matches[0] if matches else None
    template = None
    a6_geometry = None
    evidence = {'a6_matching_candidates': len(matches), 'a6_same_entry_matched': False}
    width = None
    width_source = None
    if selected:
        same_id, _, entry, match = selected
        a6_geometry = entry['free']['geometry']
        width = float(a6_geometry['width_px'])
        width_source = 'A6_same_entry_geometry' if same_id == 0 else 'A6_compatible_geometry'
        template = entry.get('template')
        evidence.update(a6_entry_id=entry['entry_id'], a6_match=match, a6_same_entry_matched=same_id == 0)
    source = str(Path(seed['source']).resolve())
    saved_source = priors.get('seed_source_records', {}).get(source)
    if width is None and saved_source and saved_source['kind'] == 'A2':
        segment = saved_source['data'].get('free_segment', {})
        if segment.get('width_px') and segment.get('direction_xy') and segment.get('center_xy'):
            match = _axis_match(xy, segment['center_xy'], segment['direction_xy'], float(segment['width_px']))
            if match['compatible']:
                width = float(segment['width_px']); width_source = 'A2_exact_free_only_seed_source'
                evidence['a2_match'] = match
    a7_matches = []
    for candidate in priors.get('a7_candidates', []):
        if not candidate.get('plausible') or candidate.get('anchor_span_px', 0) < 100.:
            continue
        match = _axis_match(xy, candidate['origin_xy'], candidate['direction_xy'], float(candidate['width_px']))
        if match['compatible']:
            cost = match['perpendicular_distance_widths'] + match['undirected_angle_deg'] / 3.
            a7_matches.append((0 if candidate.get('accepted') else 1, cost, candidate, match))
    a7_matches.sort(key=lambda item: (item[0], item[1]))
    a7_hint = None
    if a7_matches:
        _, _, candidate, match = a7_matches[0]
        a7_hint = {k: candidate.get(k) for k in ('candidate_id', 'origin_xy', 'direction_xy', 'width_px',
                  'anchor_segment_xy', 'anchor_span_px', 'accepted', 'plausible', 'fit')}
        evidence['a7_match'] = match
        if width is None:
            width = float(candidate['width_px']); width_source = 'A7_compatible_straight_width_hint'
        evidence['a7_to_selected_width_ratio'] = float(candidate['width_px']) / width
    if width is None:
        width = float(np.clip(.55 * priors['pitch_px'], 6., 70.))
        width_source = 'pitch_fallback_not_measured'
    if not np.isfinite(width) or width <= 1.:
        raise ValueError('No valid positive width hint')
    return {'width_px': width, 'width_half_px': width / 2., 'width_source': width_source,
            'template': template, 'a6_geometry': a6_geometry, 'a7_axis_hint': a7_hint,
            'seed_xy': xy.tolist(), 'origin_xy': xy[-1].tolist(), 'direction_xy': direction.tolist(),
            'actual_seed_source': source, 'actual_seed_entry_id': seed.get('entry_id'),
            'seed_entry_unchanged': True, 'coordinate_system': 'original_image_xy_pixels',
            'match_evidence': evidence, 'same_material_identity_confirmed': False}
