"""Saved-prediction-only P0/P1 comparison on eight frozen open development images.

No candidate generation, threshold selection, training, or image decoding occurs
here. Segment evidence is a detector claim, never a material-identity label.
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import evaluate as old
from compare_evaluate import first_deviation, read, save, sha
from scene import stage_at_points
from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval

SOURCE = BASE / 'runs/run_20261030_002'
MODES = ('CONDITIONAL', 'AUTO')
BRANCHES = ('A', 'B')
METHODS = ('P0', 'P1')
SUPPORT = {'heatmap_supported', 'image_supported'}
STATES = ('entry_condition', 'heatmap_supported', 'image_supported', 'bridge', 'ambiguous')
PROTOCOL = {
    **old.PROTOCOL, 'version': 2,
    'comparison': 'same frozen A/B common750 float responses and identical entry; P0 vs P1; no training',
    'population_images_per_mode_per_combination': 8, 'source_groups': 3,
    'pair0075': 'training-source development demonstration only; excluded from all formal denominators',
    'geometry_views': ['all_geometry', 'supported_only'],
    'all_geometry': 'all non-input saved selected geometry, including bridge and ambiguous; not all detected',
    'supported_only': 'continuous runs of heatmap_supported or image_supported; split at every bridge/ambiguous/input segment',
    'P1_status_authority': 'segment_status has exactly N-1 entries; per-point status is display compatibility only',
    'P0_status_mapping': 'legacy destination-point visible_supported maps to image_supported (joint heatmap+image proxy); inferred maps to bridge; definitions not calibrated equal across methods',
    'first_sustained_deviation_minimum_reference_gap_px': 60.0,
    'first_sustained_deviation': 'earliest >=60px uncovered trusted-reference interval at5px tolerance; geometric proxy, not first material identity error',
    'bridge': 'endpoint interpolation/inference; not detected rope or recovered real occlusion',
    'distance_denominator': 'distance undefined for empty prediction; report finite-distance reference mass separately; rejection contributes zero coverage',
    'timing': 'actual recorded wall time; report extraction/cache and path timing if available; not equal compute by construction',
}


def evidence_labels(candidate, method):
    n = len(candidate['points_xy'])
    if method == 'P1':
        labels = candidate.get('segment_status')
        if labels is None or len(labels) != n-1:
            raise ValueError('P1 requires authoritative N-1 segment_status')
        labels = ['bridge' if s == 'bridged' else s for s in labels]
    else:
        labels = candidate['point_status'][1:]
        labels = [{'visible_supported': 'image_supported', 'inferred': 'bridge'}.get(s, s) for s in labels]
    if set(labels) - set(STATES):
        raise ValueError('Unknown evidence state: ' + str(set(labels)-set(STATES)))
    return labels


def segments_for(candidate, method, wanted):
    """Keep consecutive eligible edges; never bridge a discarded segment."""
    if candidate is None:
        return []
    points = old._line(candidate['points_xy'])
    labels = evidence_labels(candidate, method)
    result, run = [], []
    for a, b, label in zip(points[:-1], points[1:], labels):
        if label in wanted:
            if not run:
                run = [a.tolist()]
            run.append(b.tolist())
        else:
            if len(run) >= 2:
                result.append(run)
            run = []
    if len(run) >= 2:
        result.append(run)
    return result


def lengths(candidate, method):
    result = {s: 0. for s in STATES}
    if candidate is not None:
        points = old._line(candidate['points_xy'])
        for length, status in zip(np.linalg.norm(np.diff(points, axis=0), axis=1), evidence_labels(candidate, method)):
            result[status] += float(length)
    noninput = sum(v for k, v in result.items() if k != 'entry_condition')
    return {'by_status_px': result, 'noninput_length_px': noninput,
            'supported_length_px': sum(result[s] for s in SUPPORT),
            'bridge_fraction_of_noninput': result['bridge']/noninput if noninput else None,
            'ambiguous_fraction_of_noninput': result['ambiguous']/noninput if noninput else None,
            'interpretation': 'algorithm evidence categories, not independent identity or visibility truth'}


def _path(value, folder):
    p = Path(value)
    return p if p.is_absolute() else (folder/p).resolve()


def _seed(item, mode):
    if mode == 'CONDITIONAL':
        return item['known_seed_xy'], True
    entry = read(item['auto_entry_path'])
    return (entry.get('free_seed', []) if entry['available'] else []), bool(entry['available'])


def validate_saved(run, branch, method, mode, item, hashes):
    sid = item['id']; folder = run/'formal'/branch/method/'predictions'/mode/sid
    file = folder/'paths.json'
    if not file.is_file():
        raise FileNotFoundError('All64 prediction-save gate: ' + str(file))
    value = read(file)
    meta = read(folder/'input.json') if (folder/'input.json').is_file() else {}
    meta = {**meta, **value.get('input_provenance', {})}
    response_value = value.get('response_path', meta.get('response_path'))
    # Explicit resolver accepts original floats by path: no forced copy/symlink.
    response_path = _path(response_value, folder) if response_value else folder/'response.npy'
    canonical = SOURCE/'branches'/branch/'predictions'/mode/sid/'response.npy'
    for p in [response_path, canonical, file]:
        if not p.is_file():
            raise FileNotFoundError(p)
        if str(p) not in hashes:
            hashes[str(p)] = sha(p)
    if hashes[str(response_path)] != hashes[str(canonical)]:
        raise ValueError('Response differs from frozen common750 source')
    recorded_sha = value.get('response_sha256', meta.get('response_sha256'))
    if recorded_sha and recorded_sha != hashes[str(response_path)]:
        raise ValueError('Recorded response hash mismatch')
    response = np.load(response_path, mmap_mode='r', allow_pickle=False)
    if response.shape != tuple(item['shape']) or response.dtype != np.float32 or np.isinf(response).any():
        raise ValueError('Expected original-shape float32 response, finite or NaN')
    finite = np.isfinite(response); vals = response[finite]
    if vals.size and (vals.min() < 0 or vals.max() > 1):
        raise ValueError('Response intensity outside0..1')
    seed, available = _seed(item, mode)
    actual_seed = value.get('actual_seed_xy', meta.get('actual_seed_xy'))
    if actual_seed is None:
        entry = value.get('entry', {})
        actual_seed = entry.get('actual_seed_xy', entry.get('seed_xy', entry.get('free_seed')))
    if actual_seed is not None and not np.array_equal(old._line(actual_seed), old._line(seed)):
        raise ValueError('Saved algorithm seed differs from frozen actual input')
    if method == 'P1' and actual_seed is None:
        raise ValueError('P1 must record its actual supplied seed, including [] when unavailable')
    declared_available = value.get('entry_available', meta.get('entry_available'))
    if declared_available is not None and bool(declared_available) != available:
        raise ValueError('Entry-availability mismatch')
    if not available and (value.get('selected_candidate_id') is not None or value.get('candidates')):
        raise ValueError('AUTO without entry cannot be rescued by a known seed')
    if value['mode'] != mode or len(value['candidates']) > 3:
        raise ValueError('Invalid mode/candidate count')
    ids = [c['id'] for c in value['candidates']]
    if len(ids) != len(set(ids)) or (value.get('selected_candidate_id') is not None and value['selected_candidate_id'] not in ids):
        raise ValueError('Invalid selected candidate')
    for c in value['candidates']:
        points = old._line(c['points_xy'])
        if len(points) < 2 or (points < 0).any() or (points[:, 0] >= item['shape'][1]).any() or (points[:, 1] >= item['shape'][0]).any():
            raise ValueError('Invalid original-image path coordinates')
        evidence_labels(c, method)
    original_file = SOURCE/'branches'/branch/'predictions'/mode/sid/'paths.json'
    original = read(original_file)
    if original['response_sha256'] != hashes[str(canonical)]:
        raise ValueError('Original source float no longer matches its frozen P0 receipt')
    paired = read(SOURCE/'paired_comparison.json')
    inference = read(canonical.parent/'inference.json')['phases']['final']
    if paired['main_comparison_step'] != 750 or inference['model_step'] != 750 or inference['model_sha256'] != paired['main_checkpoints'][branch]['sha256']:
        raise ValueError('Expected frozen common750 checkpoint provenance')
    if method == 'P0':
        for key in ['candidates', 'selected_candidate_id', 'parameters']:
            if value.get(key) != original.get(key):
                raise ValueError('P0 baseline must retain exact original '+key)
    bundle = None
    segments_file = value.get('segments_path', meta.get('segments_path'))
    segment_provenance = None
    if segments_file:
        segments_file = _path(segments_file, folder)
        segment_hash = sha(segments_file)
        recorded = value.get('segments_sha256', meta.get('segments_sha256'))
        if recorded and recorded != segment_hash:
            raise ValueError('Extracted-segment artifact hash mismatch')
        if segments_file.suffix == '.gz':
            with gzip.open(segments_file, 'rt', encoding='utf-8') as handle:
                bundle = json.load(handle)
        else:
            bundle = read(segments_file)
        segment_provenance = {'path': str(segments_file), 'sha256': segment_hash}
    scene_path = Path(item['scene_path'])
    runtime = value.get('runtime', value.get('timing', meta.get('runtime', {})))
    elapsed = value.get('elapsed_seconds', meta.get('elapsed_seconds'))
    if elapsed is None and isinstance(runtime, dict):
        elapsed = runtime.get('elapsed_seconds', runtime.get('total_seconds'))
    diagnostics = value.get('search', value.get('diagnostics', {}))
    if isinstance(diagnostics, dict):
        diagnostics = {k:v for k,v in diagnostics.items() if k not in {'graph_edges','entry_candidate_checks'}}
    return dict(sample_id=sid, branch=branch, method=method, mode=mode,
        group_id=item['source_group'], prediction=value, actual_input_seed_xy=seed,
        entry_available=available, actual_seed_recorded=actual_seed is not None,
        scoring_scene=read(scene_path), elapsed_seconds=elapsed, runtime=runtime,
        extracted_bundle=bundle, segment_provenance=segment_provenance,
        algorithm_diagnostics=diagnostics,
        files={'paths': str(file), 'paths_sha256': hashes[str(file)],
               'response': str(response_path), 'response_sha256': hashes[str(response_path)],
               'original_P0_paths': str(original_file), 'original_P0_paths_sha256': sha(original_file),
               'scene': str(scene_path), 'scene_sha256': sha(scene_path)},
        response_finite_pixels=int(finite.sum()))


def fragment_availability(case, pieces, reference, exclusion):
    """Posthoc global candidate availability, with no path/identity assertion.

    Fragment direction has not been selected, so support requires both endpoints
    supported. Exact point-to-segment distances are vectorized in bounded chunks;
    this is the same geometry as the legacy nearest-segment metric.
    """
    bundle = case.get('extracted_bundle')
    if bundle is None:
        return {'status': 'unavailable_no_saved_fragment_bundle', 'identity_precision': None}
    fragments = bundle['segments']; views = {}
    for view in ['all_geometry', 'supported_only']:
        paths = []
        for fragment in fragments:
            points = fragment['points_xy']; states = fragment['point_evidence_state']
            if len(states) != len(points): raise ValueError('Fragment evidence length mismatch')
            if view == 'all_geometry': paths.append(points)
            else:
                candidate = {'points_xy': points, 'segment_status': [
                    'heatmap_supported' if a in SUPPORT and b in SUPPORT else 'ambiguous'
                    for a,b in zip(states[:-1], states[1:])]}
                paths.extend(segments_for(candidate, 'P1', SUPPORT))
        paths = old.clip_scoring_segments(paths, ignore_polygons=[*reference['ignore_polygons'], *exclusion])
        edges = np.array([[a,b] for p in paths for a,b in zip(p[:-1],p[1:])], float).reshape(-1,2,2)
        if len(edges): edges = edges[np.linalg.norm(edges[:,1]-edges[:,0], axis=1)>1e-9]
        distances=[]; weights=[]; phases=[]
        for piece in pieces:
            p = piece['points']; best=np.full(len(p), np.inf)
            for lo in range(0,len(edges),256):
                a,b=edges[lo:lo+256,0],edges[lo:lo+256,1];delta=b-a
                relative=p[:,None,:]-a[None,:,:]
                t=np.clip(np.sum(relative*delta[None,:,:],axis=2)/np.sum(delta*delta,axis=1)[None,:],0,1)
                distance=np.sqrt(np.sum((relative-t[:,:,None]*delta[None,:,:])**2,axis=2))
                best=np.minimum(best,distance.min(1))
            distances.extend(best);weights.extend(piece['weights']);phases.extend(piece['stages'])
        d,w,st=map(np.asarray,(distances,weights,phases));metrics={}
        for phase,name in [(None,'all'),*old.STAGES.items()]:
            mask=np.ones(len(d),bool) if phase is None else st==phase
            dd,ww=d[mask],w[mask];length=float(ww.sum());covered=float(ww[dd<=5].sum())
            metrics[name]={'reference_length_px':length,'covered_length_px':covered,
                'coverage_5px':covered/length if length else None,
                'reference_to_fragments_mean_px':float(np.sum(dd*ww)/length) if length and np.isfinite(dd).all() else None,
                'reference_to_fragments_p95_px':old._weighted_quantile(dd,ww,.95)}
        views[view]={'metrics':metrics,'fragment_piece_count':len(paths),'fragment_length_px':old._length(paths)}
    return {'status':'posthoc_global_fragment_availability','raw_fragment_count':len(fragments),
        'geometry':views,'extraction_diagnostics':bundle.get('diagnostics'),
        'source':case['segment_provenance'],'identity_precision':None,
        'interpretation':'global unordered fragments may cover target and many wrong neighboring/background lines; availability is not an entry-connected identity path',
        'fragment_support_rule':'both segment endpoints have heatmap/image evidence; orientation independent; actual selected-path segment_status remains authoritative'}


def score_candidate(candidate, case, reference, pieces, exclusion):
    scores = {}
    for name, wanted in [('all_geometry', set(STATES)-{'entry_condition'}), ('supported_only', SUPPORT)]:
        uncut = segments_for(candidate, case['method'], wanted)
        clipped = old.clip_scoring_segments(uncut, ignore_polygons=[*reference['ignore_polygons'], *exclusion])
        score = {'candidate_id': candidate['id'] if candidate else None,
                 'all': old._summarize_reference_rows(pieces, clipped),
                 'stages': {name: old._summarize_reference_rows(pieces, clipped, k) for k, name in old.STAGES.items()},
                 'prediction_length_after_input_and_quality_exclusion_px': old._length(clipped),
                 'prediction_continuous_segment_count': len(clipped)}
        score['first_sustained_deviation_proxy'] = first_deviation(score, reference)
        scores[name] = score
    scores['candidate_id'] = candidate['id'] if candidate else None
    scores['claimed_evidence_lengths'] = lengths(candidate, case['method'])
    scores['true_complete_endpoint_success'] = None
    return scores


def score_case(case, reference):
    quality = old.clip_scoring_segments([reference['centerline']], ignore_polygons=reference['ignore_polygons'])
    exclusion = old._entry_polygons(case['actual_input_seed_xy'])
    clipped = old.clip_scoring_segments(quality, ignore_polygons=exclusion)
    pieces = old._reference_samples(clipped, case['scoring_scene'], stage_at_points)
    value = case['prediction']
    selected = next((c for c in value['candidates'] if c['id'] == value.get('selected_candidate_id')), None)
    selected_score = score_candidate(selected, case, reference, pieces, exclusion)
    candidates = [score_candidate(c, case, reference, pieces, exclusion) for c in value['candidates']]
    pool = {}
    for view in ['all_geometry', 'supported_only']:
        pool[view] = {}
        for stage in ['all', *old.STAGES.values()]:
            def s(c): return c[view]['all'] if stage == 'all' else c[view]['stages'][stage]
            best = max(candidates, key=lambda c: s(c)['covered_fraction'] or 0, default=None)
            primary = selected_score[view]['all'] if stage == 'all' else selected_score[view]['stages'][stage]
            pool[view][stage] = {'candidate_id': best['candidate_id'] if best else None,
                'coverage_5px': s(best)['covered_fraction'] if best else 0 if primary['reference_length_px'] else None}
    source = {'annotation_sha256': reference['annotation_sha256'], 'quality_sha256': reference['quality_sha256'],
              'status': reference['status'], 'trusted_clipped_centerline': clipped,
              'reference_length_before_seed_exclusion_px': old._length(quality),
              'reference_length_after_seed_exclusion_px': old._length(clipped),
              'reference_length_removed_as_input_px': old._length(quality)-old._length(clipped),
              'actual_input_seed_xy': case['actual_input_seed_xy'], 'true_edge_endpoint': None}
    return {k: case[k] for k in ['sample_id', 'branch', 'method', 'mode', 'group_id', 'entry_available',
            'actual_seed_recorded', 'elapsed_seconds', 'runtime', 'files', 'response_finite_pixels', 'algorithm_diagnostics']} | {
        'selected_candidate_id': value.get('selected_candidate_id'), 'selection_status': 'selected' if selected else 'rejected',
        'candidate_count': len(candidates), 'selected': selected_score, 'candidate_scores': candidates,
        'first_candidate_diagnostic': candidates[0] if candidates else None,
        'pool_best_posthoc_geometry_only': pool, 'source_reference': source,
        'whole_arc_identity_success': None,
        'all_extracted_fragment_availability': fragment_availability(case,pieces,reference,exclusion),
        'pool_note': 'per-stage oracle diagnostic only; best stage pieces may belong to different candidates; never automatic identity selection'}


def aggregate(rows):
    result = {}
    for branch in BRANCHES:
        result[branch] = {}
        for method in METHODS:
            result[branch][method] = {}
            for mode in MODES:
                cases = [r for r in rows if (r['branch'], r['method'], r['mode']) == (branch, method, mode)]
                assert len(cases) == 8 and {r['sample_id'] for r in cases} == set(data_eval.SAMPLE_IDS)
                views = {}
                for view in ['all_geometry', 'supported_only']:
                    views[view] = {}
                    for stage in ['all', *old.STAGES.values()]:
                        ss = [r['selected'][view]['all'] if stage == 'all' else r['selected'][view]['stages'][stage] for r in cases]
                        length = sum(s['reference_length_px'] for s in ss)
                        covered = sum(s['covered_reference_length_px'] or 0 for s in ss)
                        finite = [s for s in ss if s['reference_to_prediction_mean_px'] is not None]
                        mass = sum(s['reference_length_px'] for s in finite)
                        views[view][stage] = {'reference_length_px': length, 'covered_length_px': covered,
                            'coverage_5px': covered/length if length else None,
                            'reference_to_prediction_mean_px_available_only': sum(s['reference_length_px']*s['reference_to_prediction_mean_px'] for s in finite)/mass if mass else None,
                            'distance_available_reference_length_px': mass,
                            'longest_uncovered_run_px': max((s['longest_uncovered_run_px'] or 0 for s in ss), default=0),
                            'images_with_reference': sum(s['reference_length_px'] > 0 for s in ss)}
                totals = {s: sum(r['selected']['claimed_evidence_lengths']['by_status_px'][s] for r in cases) for s in STATES}
                noninput = sum(v for k, v in totals.items() if k != 'entry_condition')
                times = [float(r['elapsed_seconds']) for r in cases if r['elapsed_seconds'] is not None]
                result[branch][method][mode] = {'images': 8, 'source_groups': 3,
                    'selected_images': sum(r['selection_status'] == 'selected' for r in cases),
                    'rejected_images': sum(r['selection_status'] == 'rejected' for r in cases),
                    'entry_unavailable_images': sum(not r['entry_available'] for r in cases),
                    'candidate_count_total': sum(r['candidate_count'] for r in cases), 'geometry': views,
                    'selected_claimed_lengths_px': totals,
                    'selected_bridge_fraction_of_noninput_length': totals['bridge']/noninput if noninput else None,
                    'timing': {'recorded_images': len(times), 'total_seconds': sum(times) if times else None, 'mean_seconds': float(np.mean(times)) if times else None,
                               'maximum_seconds': max(times) if times else None},
                    'complete_endpoint_identity_success_rate': None}
    return result


def freeze(run):
    file = run/'evaluation_protocol.json'
    payload = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': PROTOCOL,
               'source_run': str(SOURCE), 'source_index_sha256': sha(SOURCE/'data/index.json'),
               'fixed_visual_scope_sha256': sha(SOURCE/'evaluation_scope/scope.json'),
               'sample_ids': list(data_eval.SAMPLE_IDS), 'new_predictions_read': False}
    if not file.exists(): save(file, payload)
    else: assert read(file)['protocol'] == PROTOCOL, 'Frozen scoring protocol cannot change'
    return file


def run_evaluation(run):
    protocol_file = run/'evaluation_protocol.json'
    if not protocol_file.exists():
        raise FileNotFoundError('Freeze evaluation protocol before new formal predictions')
    protocol = read(protocol_file); assert protocol['protocol'] == PROTOCOL
    assert sha(SOURCE/'data/index.json') == protocol['source_index_sha256']
    index = {i['id']: i for i in read(SOURCE/'data/index.json')['items']}
    hashes = {}; cases = []
    for branch in BRANCHES:
        for method in METHODS:
            for mode in MODES:
                for sid in data_eval.SAMPLE_IDS:
                    cases.append(validate_saved(run, branch, method, mode, index[sid], hashes))
    # Every path and exact original float is saved and validated before GT opens.
    references = {sid: data_eval.load_evaluation_reference(sid) for sid in data_eval.SAMPLE_IDS}
    rows = [score_case(c, references[c['sample_id']]) for c in cases]
    for row in rows:
        for scored in [row['selected'], *row['candidate_scores']]:
            for stage in ['all', *old.STAGES.values()]:
                a=scored['all_geometry']['all'] if stage=='all' else scored['all_geometry']['stages'][stage]
                s=scored['supported_only']['all'] if stage=='all' else scored['supported_only']['stages'][stage]
                assert a['reference_length_px']==s['reference_length_px']
                assert (s['covered_reference_length_px'] or 0) <= (a['covered_reference_length_px'] or 0)+1e-6
    for branch in BRANCHES:
        for mode in MODES:
            for sid in data_eval.SAMPLE_IDS:
                pair = [r for r in rows if (r['branch'], r['mode'], r['sample_id']) == (branch, mode, sid)]
                assert pair[0]['source_reference'] == pair[1]['source_reference'], 'P0/P1 scoring denominator changed'
    output = run/'evaluation'; output.mkdir(exist_ok=False)
    for row in rows:
        save(output/row['branch']/row['method']/row['mode']/(row['sample_id']+'.json'), row)
    payload = {'status': 'completed', 'exit_code': 0, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': PROTOCOL, 'protocol_sha256': sha(protocol_file), 'source_sha256': sha(__file__),
        'command': sys.argv, 'formal_saved_pairs': len(cases), 'summary': aggregate(rows),
        'prediction_save_boundary': [{k: c[k] for k in ['sample_id', 'branch', 'method', 'mode', 'files']} for c in cases]}
    save(output/'summary.json', payload)
    print(json.dumps({'status': 'completed', 'cases': len(rows), 'output': str(output)}, ensure_ascii=False))


def checks():
    c = {'id': 'c0', 'points_xy': [[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]],
         'segment_status': ['entry_condition', 'heatmap_supported', 'bridge', 'image_supported', 'ambiguous']}
    assert segments_for(c, 'P1', SUPPORT) == [[[10., 0.], [20., 0.]], [[30., 0.], [40., 0.]]]
    assert lengths(c, 'P1')['bridge_fraction_of_noninput'] == .25
    pieces = old._reference_samples([[[10, 0], [50, 0]]], {}, lambda s, p: np.ones(len(p), np.uint8))
    supported = old._summarize_reference_rows(pieces, segments_for(c, 'P1', SUPPORT))
    allgeo = old._summarize_reference_rows(pieces, segments_for(c, 'P1', set(STATES)-{'entry_condition'}))
    assert supported['covered_fraction'] < allgeo['covered_fraction'] == 1.
    empty = old._summarize_reference_rows(pieces, [])
    assert empty['covered_fraction'] == 0 and empty['reference_to_prediction_mean_px'] is None
    print(json.dumps({'synthetic_contracts': 'passed', 'bridge_not_connected_in_support_view': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check: checks()
    elif args.freeze: print(freeze(args.run.resolve()))
    else: run_evaluation(args.run.resolve())
