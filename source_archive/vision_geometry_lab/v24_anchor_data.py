"""Fixed legal free-anchor subsets; inference caches separate from train labels.

Preparation only, after a parent protocol is frozen. No fitting, scoring, new
width extraction, wound-label-driven anchor selection, or hardware access.
"""
from pathlib import Path
import json
import math

import numpy as np

from .v11_path_contract import build_input, build_targets, array_sha256
from .v11_path_experiment import TARGET_KEYS, verify as verify_v11
from .v23_auto_seed_chain import read, sha, write, save, verify as verify_v23

LAB = Path(__file__).resolve().parent
OLD = LAB/'outputs/v11_path_identity'
AUTO = LAB/'outputs/v23_auto_seed_chain'
MANIFEST = OLD/'development_manifest.json'
OLD_INDEX = OLD/'development_cache/index.json'
AUTO_INDEX = AUTO/'inputs.json'
TRAIN_FRACTIONS = ((0., .45), (.10, .70), (.25, 1.))
PROBE_FRACTIONS = ((.15, .62), (.35, .85))
TRAIN_VARIANTS = ('manual', 'subset0', 'subset1', 'subset2', 'auto')
EVALUATION_VARIANTS = ('manual', 'auto', 'probe0', 'probe1')
INPUT_ARRAY_KEYS = ('input', 'valid_mask')


def subpolyline(points, low, high):
    """Exact arclength subset; retain interior vertices and source orientation."""
    p = np.asarray(points, np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('Finite Nx2 free seed required')
    if not np.isfinite([low, high]).all() or not 0 <= low < high <= 1:
        raise ValueError('Fractions must satisfy 0 <= low < high <= 1')
    distances = np.linalg.norm(np.diff(p, axis=0), axis=1)
    p = p[np.r_[True, distances > 1e-9]]
    if len(p) < 2: raise ValueError('Zero-length seed')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    a, b = low*arc[-1], high*arc[-1]
    start = [np.interp(a, arc, p[:, k]) for k in range(2)]
    end = [np.interp(b, arc, p[:, k]) for k in range(2)]
    return np.vstack([start, p[(arc > a+1e-9) & (arc < b-1e-9)], end]).tolist()


def input_variants(split):
    if split == 'train': return ('manual', 'subset0', 'subset1', 'subset2', 'auto', 'probe0', 'probe1')
    if split == 'holdout': return EVALUATION_VARIANTS
    raise ValueError('Only fixed train/holdout geometry records permitted')


def entry_key(sample_id, variant):
    return sample_id+'::'+variant


def collect_sources():
    """All existing read sources for the parent's immutable snapshot."""
    paths = {Path(__file__), MANIFEST, OLD_INDEX, AUTO_INDEX, OLD/'protocol.json', AUTO/'protocol.json'}
    for name in ('v11_path_contract.py', 'v11_path_experiment.py', 'v23_auto_seed_chain.py',
                 'centerline_quality_v5.py', 'centerline_quality_cutoff_v8.py',
                 'centerline_review_v5.py', 'evaluate_centerline_multiscale_v9.py'):
        paths.add(LAB/name)
    old_protocol = read(OLD/'protocol.json')
    for category in ('source_snapshot', 'old_source_snapshot', 'dataset_snapshot'):
        paths.update(Path(p) for p, digest in old_protocol[category].items() if digest is not None)
    paths.update(Path(p) for p in read(AUTO/'protocol.json')['source_hashes'])
    for row in read(OLD_INDEX)['entries']:
        if row['available']: paths.add(Path(row['cache']))
    for row in read(AUTO_INDEX)['entries']:
        if row['available']: paths.add(Path(row['input_path']))
    for sample in read(MANIFEST)['samples']:
        paths.add(Path(sample['image_path']))
        if sample['split'] == 'train':
            paths.add(Path(sample['annotation_path']))
            q = Path(sample['annotation_path']).with_suffix('.quality.json')
            if q.exists(): paths.add(q)
    if any('20260925' in str(p) for p in paths): raise ValueError('Forbidden blind source')
    return sorted(paths, key=lambda p: str(p))


def _read_npz(path, expected, keys):
    if sha(path) != expected: raise ValueError('Cached source hash mismatch: '+str(path))
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in keys}


def _validate_source_sets(samples, original, automatic):
    if len(samples) != 18 or len({s['sample_id'] for s in samples}) != 18:
        raise ValueError('Expected 18 unique geometry samples')
    if sum(s['split'] == 'train' for s in samples) != 10 or sum(s['split'] == 'holdout' for s in samples) != 8:
        raise ValueError('Expected unchanged 10 train / 8 holdout')
    train_groups = {s['group_id'] for s in samples if s['split'] == 'train'}
    held_groups = {s['group_id'] for s in samples if s['split'] == 'holdout'}
    if train_groups & held_groups: raise ValueError('Source groups overlap')
    for s in samples:
        sid = s['sample_id']
        if sid.startswith(('C03__', 'C06__')): raise ValueError('Diagnostic-only sample cannot enter training')
        for row in (original[sid], automatic[sid]):
            if row['split'] != s['split'] or row['group_id'] != s['group_id']:
                raise ValueError('Source cohort mismatch')
        if not original[sid]['available']: raise ValueError('Required original input unavailable')


def train_target_arrays(split, variant, geometry, valid_mask, label, quality):
    if split != 'train' or variant not in TRAIN_VARIANTS:
        raise ValueError('No targets may be built for holdout or unseen probes')
    reference = quality.get('supervision_polylines', {}).get('centerline', label['polylines']['centerline'])
    target = build_targets(geometry, reference,
        label['ignore_polygons']+quality['precision_exclude_polygons'], valid_mask)
    if not np.any(target['column_valid']):
        raise ValueError('Training variant has no trustworthy supervised columns; do not silently skip')
    return {k: target[k] for k in TARGET_KEYS}, target['diagnostics']


def _schedule(sample_ids, entries):
    result = {}
    for sid in sample_ids:
        slots = []
        for variant in TRAIN_VARIANTS:
            scheduled = entry_key(sid, variant)
            available = entries[scheduled]['available']
            if not available and variant != 'auto': raise ValueError('Missing required training subset')
            slots.append(dict(scheduled_variant=variant, scheduled_key=scheduled,
                resolved_key=scheduled if available else entry_key(sid, 'manual'),
                fallback_reason=None if available else 'auto_unavailable: '+entries[scheduled]['reason']))
        result[sid] = slots
    return result


def prepare(base, protocol_sha=None):
    base = Path(base); protocol_path = base/'protocol.json'
    if not protocol_path.is_file(): raise ValueError('Freeze parent protocol before preparation')
    frozen = sha(protocol_path)
    if protocol_sha is not None and frozen != protocol_sha: raise ValueError('Parent protocol hash mismatch')
    folder = base/'cache'
    if folder.exists(): raise FileExistsError('Do not overwrite V24 cache')
    verify_v11(OLD); verify_v23(AUTO)
    snapshots = {str(p.resolve()): sha(p) for p in collect_sources()}
    manifest = read(MANIFEST); samples = manifest['samples']
    if '20260925' in json.dumps(manifest): raise ValueError('Forbidden blind source')
    old_index = read(OLD_INDEX)
    if old_index['manifest_sha256'] != sha(MANIFEST) or old_index['protocol_sha256'] != sha(OLD/'protocol.json'):
        raise ValueError('Old input cache binding changed')
    auto_index = read(AUTO_INDEX)
    if auto_index['protocol_sha256'] != sha(AUTO/'protocol.json'):
        raise ValueError('Automatic input binding changed')
    original = {r['sample_id']: r for r in old_index['entries']}
    automatic = {r['sample_id']: r for r in auto_index['entries']}
    _validate_source_sets(samples, original, automatic)
    train_ids = [r['sample_id'] for r in old_index['entries'] if r['split'] == 'train']
    entries = {}; targets = {}
    from .centerline_review_v5 import read_clean_image, validate_annotation
    from .centerline_quality_v5 import load_label_quality
    for sample in samples:
        sid = sample['sample_id']; split = sample['split']; old = original[sid]; auto = automatic[sid]
        seed = old['geometry']['free_seed_xy']; width = old['geometry']['width_px']
        if not math.isfinite(width) or width <= 0: raise ValueError('Invalid frozen width')
        rgb = np.asarray(read_clean_image(sample))
        label = quality = None
        if split == 'train':
            if sha(sample['annotation_path']) != old['annotation_sha256']:
                raise ValueError('Training annotation changed')
            label = validate_annotation(read(sample['annotation_path']), sample)
            quality = load_label_quality(sample)
            if quality['quality_sha256'] != old['quality_sha256']:
                raise ValueError('Training quality changed')
        for variant in input_variants(split):
            key = entry_key(sid, variant)
            entry = dict(key=key, sample_id=sid, split=split, group_id=sample['group_id'],
                variant=variant, available=True, reason='ok', input_path=None,
                input_sha256=None, geometry=None, target_used_for_input=False,
                original_image_sha256=sample['image_sha256'])
            if variant == 'manual':
                arrays = _read_npz(old['cache'], old['cache_sha256'], INPUT_ARRAY_KEYS)
                geometry = old['geometry']; source = 'unchanged_V11_manual_cache'
                entry['anchor_kind'] = 'original_manual'
            elif variant == 'auto':
                if not auto['available']:
                    entry.update(available=False, reason=auto['reason'], source='unchanged_V23_auto_unavailable')
                    entries[key] = entry
                    continue
                arrays = _read_npz(auto['input_path'], auto['input_sha256'], INPUT_ARRAY_KEYS)
                geometry = auto['geometry']; source = 'unchanged_V23_automatic_cache'
            else:
                is_probe = variant.startswith('probe')
                fraction = (PROBE_FRACTIONS if is_probe else TRAIN_FRACTIONS)[int(variant[-1])]
                selected_seed = subpolyline(seed, *fraction)
                built = build_input(rgb, selected_seed, width)
                arrays = {k: built[k] for k in INPUT_ARRAY_KEYS}; geometry = built['geometry']
                source = 'arclength_subset_of_manual_free_seed_only'
                entry['seed_arclength_fraction'] = list(fraction)
                entry['unseen_training_probe'] = is_probe
            path = folder/'inputs'/sid/(variant+'.npz')
            save(path, **arrays)
            entry.update(source=source, input_path=str(path), input_sha256=sha(path),
                input_array_sha256=array_sha256(arrays['input']), geometry=geometry)
            entries[key] = entry
            if split == 'train' and variant in TRAIN_VARIANTS:
                target, diagnostic = train_target_arrays(split, variant, geometry, arrays['valid_mask'], label, quality)
                tpath = folder/'train_targets'/sid/(variant+'.npz')
                save(tpath, **target)
                targets[key] = dict(key=key, sample_id=sid, split=split, variant=variant,
                    target_path=str(tpath), target_sha256=sha(tpath),
                    input_sha256=entry['input_sha256'], input_array_sha256=entry['input_array_sha256'],
                    diagnostics=diagnostic, annotation_sha256=old['annotation_sha256'],
                    quality_sha256=old['quality_sha256'], unknown_tail_is_background=False)
    schedule = _schedule(train_ids, entries)
    for slots in schedule.values():
        if any(slot['resolved_key'] not in targets for slot in slots):
            raise ValueError('Every scheduled train slot must resolve to a target')
    evaluation = [entry_key(s['sample_id'], v) for s in samples for v in EVALUATION_VARIANTS]
    index = dict(status='complete', protocol_sha256=frozen, entries=entries,
        train_sample_ids=train_ids, train_keys=schedule, evaluation_keys=evaluation,
        source_sha256=snapshots, rules=dict(train_fractions=TRAIN_FRACTIONS,
            probe_fractions=PROBE_FRACTIONS, width_remeasured=False,
            subset_width='unchanged manual V11 width', automatic='unchanged V23 input including its prior width',
            input_and_target_archives_separate=True, evaluation_auto_unavailable_retained=True))
    for path, expected in snapshots.items():
        if sha(path) != expected: raise ValueError('Source changed during preparation: '+path)
    if sha(protocol_path) != frozen: raise ValueError('Parent protocol changed during preparation')
    verify_v11(OLD); verify_v23(AUTO)
    write(folder/'input_index.json', index)
    write(folder/'train_target_index.json', dict(status='complete', protocol_sha256=frozen,
        input_index_sha256=sha(folder/'input_index.json'), entries=targets,
        no_holdout_or_probe_targets=True))
    return index


def load_input(base, key):
    base = Path(base); index = read(base/'cache/input_index.json')
    if index['protocol_sha256'] != sha(base/'protocol.json'): raise ValueError('Protocol mismatch')
    entry = index['entries'][key]
    if not entry['available']: return entry, None
    arrays = _read_npz(entry['input_path'], entry['input_sha256'], INPUT_ARRAY_KEYS)
    if array_sha256(arrays['input']) != entry['input_array_sha256']:
        raise ValueError('Input array identity mismatch')
    return entry, arrays


def load_train_target(base, key):
    base = Path(base); index = read(base/'cache/train_target_index.json')
    if index['protocol_sha256'] != sha(base/'protocol.json') or index['input_index_sha256'] != sha(base/'cache/input_index.json'):
        raise ValueError('Target cache binding mismatch')
    entry = index['entries'][key]
    if entry['split'] != 'train' or entry['variant'] not in TRAIN_VARIANTS:
        raise ValueError('Non-training target requested')
    return entry, _read_npz(entry['target_path'], entry['target_sha256'], TARGET_KEYS)
