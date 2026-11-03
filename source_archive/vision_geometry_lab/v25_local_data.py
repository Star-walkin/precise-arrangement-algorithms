"""Prepare explicitly GT-prefix-conditioned V25 patches after protocol freeze.

This is not automatic inference. Automatic initial sources are published in a
separate strict-whitelist JSON containing no annotations or references. Train
inputs, train targets and evaluation references are separate artifacts; holdout
targets are never constructed. No model fitting or width remeasurement occurs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .v11_path_experiment import read, sha, write_json, write_npz
from .v25_local_contract import build_input_pair, targets, TARGET_KEYS


LAB = Path(__file__).resolve().parent
MANIFEST = LAB/'outputs/v11_path_identity/development_manifest.json'
V24_BASE = LAB/'outputs/v24_anchor_consistency/trial_001'
V24_INDEX = V24_BASE/'cache/input_index.json'
V24_PROTOCOL = V24_BASE/'protocol.json'
INPUT_KEYS = ('high', 'low', 'valid_mask')
AUTO_ROW_KEYS = ('sample_id', 'split', 'group_id', 'source_image_path',
                 'source_image_sha256', 'width_px', 'past_seed_xy', 'available', 'reason')
PATCH_RULES = dict(prefix_widths=2., first_anchor_arc_widths=2., anchor_stride_widths=4.,
    future_reference_widths=4., require_full_future_in_same_trusted_piece=True,
    anchor_must_be_downstream_of_original_manual_seed_plane=True,
    no_crossing_quality_gaps=True, no_future_dependent_window_fit_filter=True,
    condition='oracle_given_correct_GT_prefix', unknown_tail_is_background=False)


def _safe_path(path):
    value = Path(path)
    if '20260925' in str(value):
        raise ValueError('9/25 blind sources must not be read')
    return value


def _safe_json_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and ('/' in key or '\\' in key):
                _safe_path(key)
            if isinstance(item, str) and ('path' in key or key in ('sample_id', 'manifest')):
                _safe_path(item)
            _safe_json_paths(item)
    elif isinstance(value, list):
        for item in value:
            _safe_json_paths(item)


def _read(path):
    result = read(_safe_path(path))
    _safe_json_paths(result)
    return result


def _load_records():
    from .evaluate_centerline_multiscale_v9 import _dataset
    return _dataset(MANIFEST)[1]


def _read_image(sample):
    from .centerline_review_v5 import read_clean_image
    return np.asarray(read_clean_image(sample))


def _reference(record):
    from .evaluate_centerline_multiscale_v9 import scorable_reference
    return scorable_reference(record['label'], record['quality'])


def _clean_line(points):
    line = np.asarray(points, float)
    if line.ndim != 2 or line.shape[1] != 2 or len(line) < 2 or not np.isfinite(line).all():
        raise ValueError('Expected finite polyline with at least two points')
    line = line[np.r_[True, np.linalg.norm(np.diff(line, axis=0), axis=1) > 1e-10]]
    if len(line) < 2:
        raise ValueError('Zero-length polyline')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    return line, arc


def slice_arc(points, start_px, end_px):
    """Exact subpolyline with all interior vertices; never chord across bends."""
    line, arc = _clean_line(points)
    if (not np.isfinite([start_px, end_px]).all() or start_px < -1e-8 or
            end_px > arc[-1]+1e-8 or end_px <= start_px+1e-10):
        raise ValueError('Requested arc interval lies outside one contiguous piece')
    start, end = max(0., float(start_px)), min(float(arc[-1]), float(end_px))
    first = np.array([np.interp(start, arc, line[:, k]) for k in (0, 1)])
    last = np.array([np.interp(end, arc, line[:, k]) for k in (0, 1)])
    inside = line[(arc > start+1e-9) & (arc < end-1e-9)]
    return np.vstack([first, inside, last]).tolist()


def patch_plan(reference, ignore_polygons, manual_seed, width_px):
    """Fixed arclength slots on quality-clipped pieces, not on image evidence."""
    from .centerline_path_metrics_v6 import clip_scoring_segments
    width = float(width_px)
    if not math.isfinite(width) or width <= 0:
        raise ValueError('Required original manual width is unavailable')
    seed, _ = _clean_line(manual_seed)
    vector = seed[-1]-seed[0]
    if np.linalg.norm(vector) <= 1e-10:
        raise ValueError('Original manual seed plane is undefined')
    direction = vector/np.linalg.norm(vector)
    pieces = clip_scoring_segments([reference], ignore_polygons=ignore_polygons)
    patches = []
    for piece_index, piece in enumerate(pieces):
        line, arc = _clean_line(piece)
        length = float(arc[-1])
        anchor = 2*width
        slot = 0
        while anchor+4*width <= length+1e-8:
            past = slice_arc(line, anchor-2*width, anchor)
            point = np.asarray(past[-1])
            downstream = float((point-seed[-1])@direction)
            if downstream >= -1e-8:
                future = slice_arc(line, anchor, anchor+4*width)
                patches.append(dict(piece_index=piece_index, piece_slot_index=slot,
                    anchor_arc_in_piece_px=float(anchor), trusted_piece_length_px=length,
                    past_seed_xy=past, future_reference=future, trusted_piece=line.tolist(),
                    anchor_downstream_distance_px=downstream,
                    original_manual_seed_plane=dict(origin_xy=seed[-1].tolist(), d_xy=direction.tolist()),
                    width_px=width))
            anchor += 4*width
            slot += 1
    return patches


def source_entries(index, records):
    samples = {r['sample']['sample_id']: r['sample'] for r in records}
    if len(samples) != 18 or len(records) != 18:
        raise ValueError('Exactly 18 unique geometry samples required')
    if (sum(s['split'] == 'train' for s in samples.values()) != 10 or
            sum(s['split'] == 'holdout' for s in samples.values()) != 8):
        raise ValueError('Fixed 10 train / 8 development holdout split required')
    train_groups = {s['group_id'] for s in samples.values() if s['split'] == 'train'}
    holdout_groups = {s['group_id'] for s in samples.values() if s['split'] == 'holdout'}
    if train_groups & holdout_groups:
        raise ValueError('Train and holdout source groups overlap')
    selected = {}
    for key in index['evaluation_keys']:
        entry = index['entries'][key]
        if entry['variant'] not in ('manual', 'auto'):
            continue
        identity = entry['sample_id'], entry['variant']
        if identity in selected:
            raise ValueError('Duplicate source sample/variant')
        selected[identity] = entry
    if set(selected) != {(sid, variant) for sid in samples for variant in ('manual', 'auto')}:
        raise ValueError('Missing original manual/auto sample rows')
    for sid, sample in samples.items():
        if sid.startswith(('C03__', 'C06__')) or sample.get('geometry_role') == 'diagnostic_only':
            raise ValueError('Diagnostic-only sample cannot enter geometry data')
        if any(character in sid for character in ('/', '\\', ':')):
            raise ValueError('Sample ID cannot contain path delimiters')
        for variant in ('manual', 'auto'):
            entry = selected[sid, variant]
            if entry['split'] != sample['split'] or entry['group_id'] != sample['group_id']:
                raise ValueError('Original source group/split changed')
            if type(entry['available']) is not bool:
                raise ValueError('Availability must be boolean')
            if entry.get('original_image_sha256') != sample['image_sha256']:
                raise ValueError('Original source image binding changed')
            if entry['available'] and entry['geometry']['source_shape_hw'] != sample['image_shape_hw']:
                raise ValueError('Original geometry image shape mismatch')
        if not selected[sid, 'manual']['available']:
            raise ValueError('A fixed sample has no original manual width/seed')
    return selected


def automatic_rows(records, selected):
    """The only exported auto metadata: no label path, reference or oracle pose."""
    rows = []
    for record in records:
        sample = record['sample']; entry = selected[sample['sample_id'], 'auto']
        geometry = entry['geometry'] if entry['available'] else None
        row = dict(sample_id=sample['sample_id'], split=sample['split'], group_id=sample['group_id'],
            source_image_path=sample['image_path'], source_image_sha256=sample['image_sha256'],
            width_px=float(geometry['width_px']) if geometry else None,
            past_seed_xy=geometry['free_seed_xy'] if geometry else None,
            available=entry['available'], reason=entry['reason'])
        if tuple(row) != AUTO_ROW_KEYS:
            raise AssertionError('Automatic inference metadata whitelist changed')
        if geometry and (not np.isfinite(row['width_px']) or row['width_px'] <= 0):
            raise ValueError('Available automatic source lacks a valid frozen width')
        rows.append(row)
    return rows


def collect_sources():
    """Read-only source list for the caller's protocol freeze; no preparation."""
    paths = {Path(__file__).resolve(), MANIFEST.resolve(), V24_INDEX.resolve(), V24_PROTOCOL.resolve()}
    for name in ('v25_local_contract.py', 'centerline_path_metrics_v6.py',
                 'evaluate_centerline_multiscale_v9.py', 'centerline_review_v5.py',
                 'centerline_quality_v5.py', 'centerline_quality_cutoff_v8.py', 'v11_path_experiment.py'):
        paths.add((LAB/name).resolve())
    for sample in _read(MANIFEST)['samples']:
        paths.update(_safe_path(sample[name]).resolve() for name in ('image_path', 'annotation_path'))
        quality = Path(sample['annotation_path']).with_suffix('.quality.json')
        if quality.exists():
            paths.add(quality.resolve())
    index = _read(V24_INDEX)
    for key in index['evaluation_keys']:
        entry = index['entries'][key]
        if entry['variant'] in ('manual', 'auto') and entry['available']:
            paths.add(_safe_path(entry['input_path']).resolve())
    return sorted((_safe_path(path) for path in paths), key=str)


def _verify_protocol(base, protocol_hash):
    path = _safe_path(base)/'protocol.json'
    if not path.is_file():
        raise ValueError('Freeze the parent protocol before real preparation')
    if sha(path) != protocol_hash:
        raise ValueError('Parent protocol hash mismatch')
    protocol = _read(path)
    for category in ('source_hashes', 'source_snapshot'):
        for source, digest in protocol.get(category, {}).items():
            source = _safe_path(source)
            if digest is None:
                if source.exists():
                    raise ValueError('Previously absent source appeared')
            elif sha(source) != digest:
                raise ValueError('Frozen parent source changed: '+str(source))


def prepare(base, protocol_hash):
    """Publish cache/index.json after checks; no overwrite or silent image drop."""
    base = _safe_path(base).resolve()
    _verify_protocol(base, protocol_hash)
    folder = base/'cache'
    if folder.exists():
        raise FileExistsError('Do not overwrite V25 prepared data')
    source_index = _read(V24_INDEX)
    if source_index['protocol_sha256'] != sha(V24_PROTOCOL):
        raise ValueError('Original V24 input index protocol mismatch')
    records = _load_records()
    selected = source_entries(source_index, records)
    hashes = {str(path.resolve()): sha(path) for path in collect_sources()}
    for entry in selected.values():
        if entry['available'] and sha(_safe_path(entry['input_path'])) != entry['input_sha256']:
            raise ValueError('Frozen original input cache changed')
    plans = {}
    for record in records:
        sample, label, quality = record['sample'], record['label'], record['quality']
        sid = sample['sample_id']; manual = selected[sid, 'manual']['geometry']
        if not np.array_equal(manual['free_seed_xy'], label['polylines']['free_seed']):
            raise ValueError('Original manual seed differs from verified annotation')
        reference, reference_source = _reference(record)
        ignored = label['ignore_polygons']+quality['precision_exclude_polygons']
        patches = patch_plan(reference, ignored, manual['free_seed_xy'], manual['width_px'])
        if not patches:
            raise ValueError('Fixed sample has no complete oracle patch: '+sid)
        plans[sid] = dict(patches=patches, ignored=ignored, reference_source=reference_source)
    # No cache is created until every one of the fixed eighteen images has a
    # nonempty, predetermined geometric plan. Input support never selects plans.
    folder.mkdir(parents=True, exist_ok=False)
    entries, train_keys, holdout_keys, evaluation_keys = {}, {}, {}, []
    for record in records:
        sample = record['sample']; sid, split = sample['sample_id'], sample['split']
        rgb = _read_image(sample)
        plan = plans[sid]
        keys = []
        for patch_index, patch in enumerate(plan['patches']):
            key, stem = sid+'::p'+str(patch_index).zfill(4), 'p'+str(patch_index).zfill(4)
            pair = build_input_pair(rgb, patch['past_seed_xy'], patch['width_px'])
            # Geometry records the explicit oracle condition; automatic rollout
            # builds new geometry from auto_sources.json and its own history.
            geometry = dict(pair['geometry'], past_seed_source='trusted_GT_prefix_only_oracle_condition',
                ground_truth_prefix_conditioned=True, future_GT_used_to_build_input=False)
            input_path = folder/split/'inputs'/sid/(stem+'.npz')
            reference_path = folder/split/'references'/sid/(stem+'.json')
            write_npz(input_path, high=pair['high'], low=pair['low'], valid_mask=pair['valid_mask'])
            reference_record = dict(key=key, sample_id=sid, split=split, group_id=sample['group_id'],
                future_reference=patch['future_reference'], future_reference_length_px=4*patch['width_px'],
                ignore_polygons=plan['ignored'], original_manual_seed_plane=patch['original_manual_seed_plane'],
                manual_width_px=patch['width_px'], reference_source=plan['reference_source'],
                annotation_sha256=record['annotation_sha256'], quality_sha256=record['quality']['quality_sha256'],
                ground_truth_prefix_conditioned=True, automatic_evaluation=False,
                prefix_not_part_of_scored_future=True, unknown_tail_is_background=False,
                protocol_sha256=protocol_hash)
            write_json(reference_path, reference_record)
            entry = dict(key=key, sample_id=sid, split=split, group_id=sample['group_id'], cohort=record['cohort'],
                patch_index=patch_index, piece_index=patch['piece_index'], piece_slot_index=patch['piece_slot_index'],
                anchor_arc_in_piece_px=patch['anchor_arc_in_piece_px'], trusted_piece_length_px=patch['trusted_piece_length_px'],
                anchor_downstream_distance_px=patch['anchor_downstream_distance_px'],
                geometry=geometry, input_path=str(input_path), input_sha256=sha(input_path),
                reference_path=str(reference_path), reference_sha256=sha(reference_path),
                ground_truth_prefix_conditioned=True, automatic_evaluation=False,
                prefix_provenance='exact arclength [anchor-2w,anchor] within one quality-clipped GT piece',
                future_GT_used_to_build_input=False)
            if split == 'train':
                target = targets(patch['trusted_piece'], plan['ignored'], geometry, pair['valid_mask'])
                if not np.any(target['column_valid']):
                    raise ValueError('Predetermined train patch has no trusted supervision: '+key)
                target_path = folder/'train/targets'/sid/(stem+'.npz')
                write_npz(target_path, **{name: target[name] for name in TARGET_KEYS})
                entry.update(target_path=str(target_path), target_sha256=sha(target_path),
                    target_diagnostics=target['diagnostics'])
            entries[key] = entry; keys.append(key); evaluation_keys.append(key)
        (train_keys if split == 'train' else holdout_keys)[sid] = keys
    auto_path = folder/'auto_sources.json'
    write_json(auto_path, dict(schema='v25_automatic_initial_sources_1', protocol_sha256=protocol_hash,
        rows=automatic_rows(records, selected)))
    _verify_protocol(base, protocol_hash)
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise ValueError('Source changed during oracle data preparation')
    index = dict(schema='v25_GT_prefix_conditioned_local_cache_1', status='complete',
        protocol_sha256=protocol_hash, entries=entries,
        train_sample_ids=list(train_keys), holdout_sample_ids=list(holdout_keys),
        train_keys=train_keys, holdout_keys=holdout_keys, evaluation_keys=evaluation_keys,
        source_sha256=hashes, manifest_sha256=sha(MANIFEST), v24_input_index_sha256=sha(V24_INDEX),
        auto_sources_path=str(auto_path), auto_sources_sha256=sha(auto_path),
        ground_truth_prefix_conditioned=True, automatic_evaluation=False,
        training_schedule='caller rotates each fixed image through its own frozen patch-key list',
        rules=PATCH_RULES, train_input_and_target_archives_separate=True,
        holdout_targets_built=False, all_fixed_samples_retained=True,
        new_training=False, hardware_access=False, identity_confirmed=False)
    write_json(folder/'index.json', index)
    return index


def load_input(base, key):
    index = _read(Path(base)/'cache/index.json')
    if index['protocol_sha256'] != sha(Path(base)/'protocol.json'):
        raise ValueError('Input cache protocol changed')
    entry = index['entries'][key]
    if sha(_safe_path(entry['input_path'])) != entry['input_sha256']:
        raise ValueError('Oracle input artifact changed')
    with np.load(entry['input_path'], allow_pickle=False) as archive:
        if set(archive.files) != set(INPUT_KEYS):
            raise ValueError('Input archive must contain high,low,valid_mask only')
        arrays = {name: archive[name].copy() for name in INPUT_KEYS}
    return entry, arrays


def load_train_target(base, key):
    index = _read(Path(base)/'cache/index.json')
    if index['protocol_sha256'] != sha(Path(base)/'protocol.json'):
        raise ValueError('Target cache protocol changed')
    entry = index['entries'][key]
    if entry['split'] != 'train' or 'target_path' not in entry:
        raise ValueError('Holdout cannot have or request training targets')
    if sha(_safe_path(entry['target_path'])) != entry['target_sha256']:
        raise ValueError('Training target artifact changed')
    with np.load(entry['target_path'], allow_pickle=False) as archive:
        if set(archive.files) != set(TARGET_KEYS):
            raise ValueError('Unexpected training-target archive fields')
        arrays = {name: archive[name].copy() for name in TARGET_KEYS}
    return entry, arrays
