"""Independent frozen heatmap evaluation. Never trains or reads H6 paths as truth.

All 48 requested prediction records must exist and match the inference freeze
before quality-clipped historical evaluation geometry is opened. Coordinates
and all profile distances are in original pixels; responses are never stretched.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import evaluate as old_eval
from scene import stage_at_points
from vision_geometry_lab.centerline_path_metrics_v6 import _sampling, clip_scoring_segments

ARMS = ('B', 'C', 'D')
MODES = ('CONDITIONAL', 'AUTO')
SCOPE = BASE / 'runs/run_20261030_002/evaluation_scope/scope.json'
SCOPE_SHA = '94e479fa8d050604cad7bae740d773f74fdd5a545da64da6e607ad6cb9d43bc5'
PARAMETERS = {
    'population': 'eight previously open evaluation images / three source groups; not blind',
    'response_semantics': 'center response strength, not calibrated identity probability',
    'stage_semantics': 'frozen visual position proxy; not physical contact',
    'seed_exclusion_radius_original_px': 5.,
    'response_thresholds': [.2, .5, .8],
    'profile_offsets_original_px': [-30., 30.],
    'profile_sampling_original_px': 1.,
    'peak_minimum_height': .2,
    'peak_minimum_prominence': .05,
    'peak_match_tolerance_original_px': 5.,
    'strong_peak_height': .5,
    'near_constant_std_warning': .01,
    'missing_auto': 'unavailable statistics, retained in 8-image denominator, never replaced by zero',
    'negative_scope': '16 frozen engineering background polygons and 2 inspected neighboring-band proxies only',
    'unlabelled': 'unknown; never whole-image false positives',
    'heatmap_profile_meaning': 'partial-reference geometric response diagnostic, not path identity accuracy',
    'no_H6_or_other_predicted_path_used_as_reference': True,
    'seed_ablation': {'modes':['zero_seed','moved_seed'], 'population':'8 CONDITIONAL images for each arm',
                      'far_distance_from_original_seed_original_px':256.,
                      'meaning':'input-condition sensitivity, not proof of correct target identity or calibrated rejection'},
}


def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def save_new(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)


def stats(values, weights=None):
    a = np.asarray(values, float).ravel()
    w = np.ones(len(a)) if weights is None else np.asarray(weights, float).ravel()
    if len(w) != len(a) or np.any(w < 0):
        raise ValueError('invalid weights')
    good = np.isfinite(a) & np.isfinite(w) & (w > 0)
    total = float(w[np.isfinite(w)].sum())
    a, w = a[good], w[good]
    out = dict(samples=int(good.sum()), total_weight=total, finite_weight=float(w.sum()),
               mean=None, std=None, min=None, max=None, p10=None, p50=None, p90=None, p95=None,
               fraction_lt_0_2=None, fraction_gt_0_5=None, fraction_gt_0_8=None)
    if not len(a):
        return out
    mean = float(np.average(a, weights=w))
    order = np.argsort(a)
    c = np.cumsum(w[order])
    out.update(mean=mean, std=float(np.sqrt(np.average((a-mean)**2, weights=w))),
               min=float(a.min()), max=float(a.max()))
    for q, name in ((.1, 'p10'), (.5, 'p50'), (.9, 'p90'), (.95, 'p95')):
        out[name] = float(np.interp(q*c[-1], c, a[order]))
    out.update(fraction_lt_0_2=float(w[a < .2].sum()/w.sum()),
               fraction_gt_0_5=float(w[a > .5].sum()/w.sum()),
               fraction_gt_0_8=float(w[a > .8].sum()/w.sum()))
    return out


def sample(response, points, bbox=(0, 1050, 4096, 3000), scale=.5):
    p = np.asarray(points, np.float32).reshape(-1, 2)
    if not len(p):
        return np.empty(0, np.float32)
    p = (p - np.asarray(bbox[:2], np.float32))*scale
    return cv2.remap(response, p[:, 0][None], p[:, 1][None], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))[0]


def polygon_stats(response, points):
    p = np.rint((np.asarray(points, float)-[0, 1050])*.5).astype(np.int32)
    mask = np.zeros(response.shape, np.uint8)
    cv2.fillPoly(mask, [p], 1)
    return stats(response[mask.astype(bool)])


def actual_seed(info):
    for key in ('seed_xy', 'free_seed_entry_only_xy', 'free_seed'):
        if key in info:
            return np.asarray(info[key], float).reshape(-1, 2)
    entry = info.get('entry', {})
    if isinstance(entry, dict):
        for key in ('seed_xy', 'free_seed', 'free_seed_entry_only_xy'):
            if key in entry:
                return np.asarray(entry[key], float).reshape(-1, 2)
    raise ValueError('inference must record actual original-coordinate seed')


def freeze_files(freeze, run):
    rows = freeze.get('files', freeze.get('artifacts'))
    if isinstance(rows, dict):
        rows = [{'path': p, 'sha256': h} for p, h in rows.items()]
    if not isinstance(rows, list) or not rows:
        raise ValueError('freeze needs nonempty files=[{path,sha256}]')
    found = {}
    for row in rows:
        p = Path(row['path'])
        if not p.is_absolute():
            p = run/p
        p = p.resolve()
        if sha(p) != row['sha256']:
            raise ValueError('frozen artifact changed: '+str(p))
        found[str(p)] = row['sha256']
    return found


def validate_predictions(run, protocol, freeze):
    files = freeze_files(freeze, run)
    if freeze.get('protocol_sha256') != sha(run/'protocol.json'):
        raise ValueError('protocol freeze mismatch')
    choice=read(run/'selection.json')
    if freeze.get('selection_sha256')!=sha(run/'selection.json') or freeze.get('models')!=choice['models']:
        raise ValueError('model selection freeze mismatch')
    if not freeze.get('fixed_choice_before_evaluation') or freeze.get('scored_before_freeze'):
        raise ValueError('evaluation must follow frozen training-side selection')
    for model in choice['models'].values():
        if sha(model['path'])!=model['sha256']:raise ValueError('frozen model changed')
    cases = []
    for arm in ARMS:
        for mode in MODES:
            for sid in protocol['eval_ids']:
                folder = run/'evaluation/predictions'/arm/mode/sid
                rp, ip = folder/'response.npz', folder/'inference.json'
                if str(rp.resolve()) not in files or str(ip.resolve()) not in files:
                    raise ValueError('prediction not covered by freeze: '+str(folder))
                info = read(ip)
                model=choice['models'][arm]
                if (info.get('variant')!=arm or info.get('model_sha256')!=model['sha256'] or
                    Path(info['model_path']).resolve()!=Path(model['path']).resolve() or
                    info.get('response_sha256')!=files[str(rp.resolve())]):
                    raise ValueError('prediction model/arm binding mismatch')
                with np.load(rp, allow_pickle=False) as z:
                    r = z['response']
                    assert r.shape == (975, 2048) and r.dtype == np.float32
                    f = np.isfinite(r)
                    if f.any():
                        assert f.all() and r.min() >= 0 and r.max() <= 1
                    else:
                        assert np.isnan(r).all() and mode == 'AUTO'
                if bool(info.get('available'))!=bool(f.any()):raise ValueError('availability metadata mismatch')
                if info.get('labels_read') is not False:raise ValueError('inference must be annotation-free')
                if not info.get('mapping','').startswith('model_xy=(original_xy-[0,1050])*.5'):
                    raise ValueError('half-resolution mapping must be explicit')
                seed = actual_seed(info)
                if f.any() and len(seed) < 2:
                    raise ValueError('available prediction without short entrance seed')
                if not f.any() and len(seed):
                    raise ValueError('missing AUTO may not use known fallback seed')
                if list(info.get('bbox', [0, 1050, 4096, 3000])) != [0, 1050, 4096, 3000]:
                    raise ValueError('wrong coordinate mapping')
                cases.append(dict(arm=arm, mode=mode, id=sid, response_path=rp,
                                  inference_path=ip, info=info, seed=seed,
                                  available=bool(f.any()), response_sha256=files[str(rp.resolve())]))
    for mode in MODES:
        for sid in protocol['eval_ids']:
            group = [c for c in cases if c['mode'] == mode and c['id'] == sid]
            if any(c['available'] != group[0]['available'] or
                   not np.array_equal(c['seed'], group[0]['seed']) for c in group):
                raise ValueError('arms differ in input seed or availability')
    # Gate ablations as well, before any evaluation reference is read.
    for case in cases:
        case['ablations']={}
        if case['mode']!='CONDITIONAL':continue
        for name in ('zero_seed','moved_seed'):
            folder=case['response_path'].parent/'ablations'/name
            rp,ip=folder/'response.npz',folder/'inference.json'
            if str(rp.resolve()) not in files or str(ip.resolve()) not in files:
                raise ValueError('ablation not covered by freeze: '+str(folder))
            info=read(ip);seed=actual_seed(info)
            if info.get('image_sha256')!=case['info'].get('image_sha256'):
                raise ValueError('seed ablation changed source image')
            if name=='zero_seed' and len(seed):raise ValueError('zero seed is nonempty')
            if name=='moved_seed' and (len(seed)<2 or np.array_equal(seed,case['seed'])):
                raise ValueError('moved seed missing or unchanged')
            with np.load(rp,allow_pickle=False) as z:
                r=z['response']
                assert r.shape==(975,2048) and r.dtype==np.float32
                assert np.isfinite(r).all() and r.min()>=0 and r.max()<=1
            case['ablations'][name]={'response_path':rp,'info':info,'seed_xy':seed.tolist(),
                                      'response_sha256':files[str(rp.resolve())]}
    return cases, files


def peak_positions(v, offsets):
    """Deterministic local maxima with plateau collapsing and .05 prominence.

    Constant or uniformly high profiles have no prominent peak. This is a
    response-structure diagnostic; neither peak height nor prominence is truth.
    """
    v = np.asarray(v, float)
    if not np.isfinite(v).all():
        return []
    out, k = [], 1
    while k < len(v)-1:
        end = k
        while end+1 < len(v) and abs(v[end+1]-v[k]) <= 1e-7:
            end += 1
        if end < len(v)-1 and v[k] > v[k-1] and v[end] > v[end+1] and v[k] >= .2:
            prominence = v[k]-max(float(v[:k+1].min()), float(v[end:].min()))
            if prominence >= .05:
                mid = (k+end)//2
                out.append(dict(offset_original_px=float(offsets[mid]), height=float(v[mid]),
                                prominence=float(prominence)))
        k = end+1
    return out


def profiles(response, pieces, scene):
    offsets = np.arange(-30., 31.)
    rows = []
    for piece_id, piece in enumerate(_sampling(pieces)):
        points, tangents = piece['points'], piece['tangents']
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        xy = points[:, None, :] + offsets[None, :, None]*normals[:, None, :]
        values = sample(response, xy.reshape(-1, 2)).reshape(len(points), len(offsets))
        stage = stage_at_points(scene, points)
        for i, v in enumerate(values):
            peaks = peak_positions(v, offsets)
            strongest = max(peaks, key=lambda x: x['height'], default=None)
            close = [x for x in peaks if abs(x['offset_original_px']) <= 5. and x['height'] >= .5]
            rows.append(dict(point_xy=points[i].tolist(), piece_id=piece_id,
                             weight_original_px=float(piece['weights'][i]), stage=int(stage[i]),
                             center_response=float(v[30]) if np.isfinite(v[30]) else None,
                             profile_available=bool(np.isfinite(v).all()),
                             peaks=peaks, peak_count=len(peaks), strong_peak_within_5px=bool(close),
                             strongest_peak_offset_original_px=strongest['offset_original_px'] if strongest else None,
                             high_response_width_within_60px=float(np.sum((v[1:] > .5)*.5+(v[:-1] > .5)*.5)) if np.isfinite(v).all() else None))
    return rows


def reference_stats(rows, stage=None):
    rr = [r for r in rows if stage is None or r['stage'] == stage]
    weights = np.asarray([r['weight_original_px'] for r in rr])
    values = np.asarray([np.nan if r['center_response'] is None else r['center_response'] for r in rr])
    out = stats(values, weights)
    good = np.isfinite(values)
    denom = sum(r['weight_original_px'] for r in rr if r['profile_available'])
    out['reference_length_original_px'] = float(weights.sum())
    out['finite_profile_reference_length_original_px'] = float(denom)
    out['strong_peak_within_5px_response_coverage'] = (sum(r['weight_original_px'] for r in rr
        if r['profile_available'] and r['strong_peak_within_5px'])/denom if denom else None)
    out['multiple_prominent_peaks_fraction'] = (sum(r['weight_original_px'] for r in rr
        if r['profile_available'] and r['peak_count'] > 1)/denom if denom else None)
    out['no_prominent_peak_fraction'] = (sum(r['weight_original_px'] for r in rr
        if r['profile_available'] and r['peak_count'] == 0)/denom if denom else None)
    out['strongest_peak_abs_offset_original_px'] = stats([
        np.nan if r['strongest_peak_offset_original_px'] is None else abs(r['strongest_peak_offset_original_px']) for r in rr], weights)
    out['high_response_width_within_60px_original_px'] = stats([
        np.nan if r['high_response_width_within_60px'] is None else r['high_response_width_within_60px'] for r in rr], weights)
    runs, length, start, last = [], 0., None, None
    for r in rr:
        low = r['center_response'] is not None and r['center_response'] < .2
        continuous = last is None or (r['piece_id'] == last['piece_id'] and
                      np.linalg.norm(np.asarray(r['point_xy'])-last['point_xy']) <= 5.)
        if not low or not continuous:
            if length:
                runs.append(dict(length_original_px=length, from_xy=start, to_xy=last['point_xy']))
            length, start = 0., None
        if low:
            start = r['point_xy'] if start is None else start
            length += r['weight_original_px']
        last = r
    if length:
        runs.append(dict(length_original_px=length, from_xy=start, to_xy=last['point_xy']))
    out['longest_center_response_below_0_2'] = max(runs, key=lambda x:x['length_original_px'], default=None)
    return out


def lighting_association(response, gray):
    # These bins are fixed raw intensity, never selected from model responses.
    result = {'meaning': 'association only; true rope and illumination are confounded, not a causal lighting test',
              'raw_gray_bins': []}
    for lo, hi in ((0., .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.000001)):
        mask = (gray >= lo) & (gray < hi)
        result['raw_gray_bins'].append({'range_0_1': [lo, min(hi,1.)], 'response': stats(response[mask])})
    f = np.isfinite(response)
    result['pearson_raw_gray_response'] = (float(np.corrcoef(gray[f], response[f])[0, 1])
        if f.any() and gray[f].std() > 1e-8 and response[f].std() > 1e-8 else None)
    return result


def boundary_stats(response):
    out = {}
    for name, start, end in (('top64_original_rows',0,32),('bottom64_original_rows',943,975)):
        a=response[start:end];finite=np.isfinite(a);counts=finite.sum(1)
        means=np.divide(np.nansum(a,axis=1),counts,out=np.full(end-start,np.nan),where=counts>0)
        k=int(np.nanargmax(means)) if np.isfinite(means).any() else None
        out[name]={'maximum_row_mean':float(means[k]) if k is not None else None,
                   'maximum_row_center_y_original':float(1050+2*(start+k)) if k is not None else None}
    return out


def ablation_score(case, response, profile_rows, scope):
    if not case['ablations']:return None
    seedmask=np.ones(response.shape,np.uint8)
    p=np.rint((case['seed']-[0,1050])*.5).astype(np.int32)
    cv2.polylines(seedmask,[p],False,0,1)
    distance=cv2.distanceTransform(seedmask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    far=distance>128.
    points=np.asarray([r['point_xy'] for r in profile_rows],float)
    weights=np.asarray([r['weight_original_px'] for r in profile_rows])
    phases=np.asarray([r['stage'] for r in profile_rows])
    center=sample(response,points)
    farref=sample(distance,points)>128.
    out={}
    for name,a in case['ablations'].items():
        with np.load(a['response_path'],allow_pickle=False) as z:r=z['response']
        delta=r-response;cv=sample(r,points);diff=cv-center
        out[name]={'response_sha256':a['response_sha256'],'actual_seed_xy':a['seed_xy'],
            'search_domain_mean_abs_change':float(np.abs(delta).mean()),
            'far_domain_mean_abs_change':float(np.abs(delta[far]).mean()),
            'far_domain_mean_signed_change':float(delta[far].mean()),
            'target_reference_response':stats(cv,weights),
            'target_reference_signed_change':stats(diff,weights),
            'target_reference_absolute_change':stats(np.abs(diff),weights),
            'far_target_reference_absolute_change':stats(np.abs(diff[farref]),weights[farref]),
            'stage_signed_change':{label:stats(diff[phases==s],weights[phases==s]) for s,label in old_eval.STAGES.items()},
            'background_regions':[{'id':b['id'],'response':polygon_stats(r,b['polygon_xy']),
                                   'signed_change':polygon_stats(delta,b['polygon_xy'])} for b in scope['background_regions']],
            'interpretation':'zero/moved input sensitivity only; unknown alternate target is not asserted to be negative'}
    return out


def case_score(case, scope, item):
    if case['info'].get('image_sha256') != item['image_sha256'] or item['image_sha256'] != scope['image_sha256']:
        raise ValueError('inference/source/reference image hash mismatch')
    with np.load(case['response_path'], allow_pickle=False) as z:
        response=z['response']
    scene=read(scope['common_scene'])
    if sha(scope['common_scene']) != scope['common_scene_sha256']:
        raise ValueError('frozen scene changed')
    segments=scope['quality_clipped_center_segments_xy']
    seed=case['seed']
    polygons=old_eval._entry_polygons(seed,5.) if len(seed)>1 else []
    clipped=clip_scoring_segments(segments,ignore_polygons=polygons)
    rows=profiles(response,clipped,scene)
    center={'all':reference_stats(rows),'stages':{name:reference_stats(rows,s) for s,name in old_eval.STAGES.items()}}
    pairs=[]
    for p in scope['target_neighbor_pairs']:
        t=clip_scoring_segments([p['target_points_xy']],ignore_polygons=polygons)
        n=clip_scoring_segments([p['neighbor_points_xy']],ignore_polygons=polygons)
        def ls(segments):
            pieces=_sampling(segments)
            return stats(np.concatenate([sample(response,q['points']) for q in pieces]) if pieces else [],
                         np.concatenate([q['weights'] for q in pieces]) if pieces else [])
        ts,ns=ls(t),ls(n)
        pairs.append({'id':p['id'],'target':ts,'neighbor':ns,
                      'target_minus_neighbor':ts['mean']-ns['mean'] if ts['mean'] is not None and ns['mean'] is not None else None,
                      'meaning':'frozen engineering adjacent-band contrast proxy; not identity accuracy'})
    with np.load(item['cache'],allow_pickle=False) as z:gray=z['image']
    roi=stats(response)
    out={'id':case['id'],'arm':case['arm'],'mode':case['mode'],'source_group':item['source_group'],
         'available':case['available'],'response_sha256':case['response_sha256'],
         'image_sha256':item['image_sha256'],'seed_xy':seed.tolist(),
         'reference_length_before_seed_exclusion_original_px':old_eval._length(segments),
         'reference_length_after_seed_exclusion_original_px':old_eval._length(clipped),
         'reference_removed_as_input_original_px':old_eval._length(segments)-old_eval._length(clipped),
         'target_center':center,'target_neighbor_pairs':pairs,
         'background_regions':[{'id':b['id'],'statistics':polygon_stats(response,b['polygon_xy'])} for b in scope['background_regions']],
         'search_domain_response':roi,'near_constant_warning':roi['std'] is not None and roi['std']<.01,
         'lighting_association':lighting_association(response,gray),'boundary_rows':boundary_stats(response),
         'seed_ablation':ablation_score(case,response,rows,scope),
         'profile_rows':rows,'complete_identity_accuracy':None,'true_endpoint_accuracy':None}
    return out


def mean_values(v):
    v=[x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def aggregate(rows):
    out={}
    for arm in ARMS:
        out[arm]={}
        for mode in MODES:
            rr=[r for r in rows if r['arm']==arm and r['mode']==mode]
            group={g:[r for r in rr if r['source_group']==g] for g in sorted({r['source_group'] for r in rr})}
            centers={}
            for key in ('all',*old_eval.STAGES.values()):
                vals=[r['target_center']['all' if key=='all' else 'stages'] if key=='all' else r['target_center']['stages'][key] for r in rr]
                centers[key]={k:mean_values([v[k] for v in vals]) for k in
                    ('mean','fraction_lt_0_2','fraction_gt_0_5','fraction_gt_0_8','strong_peak_within_5px_response_coverage','multiple_prominent_peaks_fraction','no_prominent_peak_fraction')}
                centers[key]['available_images']=sum(v['mean'] is not None for v in vals)
                centers[key]['total_reference_length_original_px']=sum(v['reference_length_original_px'] for v in vals)
            bg=[b['statistics'] for r in rr for b in r['background_regions']]
            out[arm][mode]={'images':len(rr),'source_groups':len(group),'available_images':sum(r['available'] for r in rr),
                'missing_entry_images':sum(not r['available'] for r in rr),'center_image_macro':centers,
                'background_region_macro':{k:mean_values([b[k] for b in bg]) for k in ('mean','p90','p95','fraction_gt_0_5','fraction_gt_0_8')},
                'background_available_regions':sum(b['mean'] is not None for b in bg),
                'target_neighbor_delta_two_proxy_macro':mean_values([p['target_minus_neighbor'] for r in rr for p in r['target_neighbor_pairs']]),
                'source_group_macro_target_mean':mean_values([mean_values([r['target_center']['all']['mean'] for r in gg]) for gg in group.values()]),
                'near_constant_images':sum(r['near_constant_warning'] for r in rr)}
            if mode=='CONDITIONAL':
                out[arm][mode]['seed_ablation_image_macro']={name:{
                    key:mean_values([r['seed_ablation'][name][key] for r in rr])
                    for key in ('search_domain_mean_abs_change','far_domain_mean_abs_change','far_domain_mean_signed_change')}
                    for name in ('zero_seed','moved_seed')}
    return out


def checks():
    o=np.arange(-30.,31.)
    assert not peak_positions(np.ones(61)*.8,o)
    p=peak_positions(np.exp(-.5*((o-9)/3)**2),o)
    assert len(p)==1 and p[0]['offset_original_px']==9.
    p=peak_positions(np.exp(-.5*((o-12)/2)**2)+np.exp(-.5*((o+12)/2)**2),o)
    assert len(p)==2
    assert stats([np.nan])['mean'] is None
    assert stats([0,1],[3,1])['mean']==.25
    a=np.arange(12,dtype=np.float32).reshape(3,4)
    assert sample(a,[[4,1052]])[0]==6
    return {'passed':True,'checks':6,'scope':'numerical response/profile checks only; no evaluation images or labels read'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=HERE);parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        print(json.dumps(checks()));return
    run=args.run.resolve();protocol=read(run/'protocol.json');freeze=read(run/'evaluation/freeze.json')
    cases,files=validate_predictions(run,protocol,freeze)  # Must finish before first scope read.
    if sha(SCOPE)!=SCOPE_SHA:raise ValueError('old independent scope changed')
    scope=read(SCOPE);scopes={r['sample_id']:r for r in scope['items']}
    assert set(scopes)==set(protocol['eval_ids']) and len(scopes)==8
    if sha(protocol['legacy_index']['path']) != protocol['legacy_index']['sha256']:
        raise ValueError('frozen data index changed')
    index=read(protocol['legacy_index']['path']);items={r['id']:r for r in index['items'] if r['split']=='eval'}
    rows=[]
    for case in cases:
        sid=case['id'];row=case_score(case,scopes[sid],items[sid]);rows.append(row)
        save_new(run/'evaluation/metrics'/case['arm']/case['mode']/(sid+'.json'),row)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'parameters':PARAMETERS,
             'protocol_sha256':sha(run/'protocol.json'),'prediction_freeze_sha256':sha(run/'evaluation/freeze.json'),
             'evaluation_code_sha256':sha(__file__),'frozen_reference_scope_sha256':SCOPE_SHA,
             'prediction_records':len(rows),'all_frozen_prediction_files_unchanged':all(sha(p)==h for p,h in files.items()),
             'aggregation':'center per-image arclength weighted, then image macro; additional source-group macro; background region macro',
             'results':aggregate(rows),'no_training_or_inference_changes':True,'exit_code':0}
    save_new(run/'evaluation/summary.json',summary)
    print(json.dumps({'exit_code':0,'prediction_records':len(rows),'summary':str(run/'evaluation/summary.json')}))


if __name__=='__main__':main()
