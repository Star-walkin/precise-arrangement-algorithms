"""Frozen archived-profile replay for V19 image-front candidates only.

No raw video decoding, camera access, truth labels, semantic inference or device
imports. Upstream integrity verification may hash original video files.
The timing covers only profile quality and the candidate state machine; it is
not camera-to-command latency or full vision inference speed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from vision_geometry_lab import v18_gradient_units_trial as upstream
from vision_geometry_lab import v19_profile_evidence as evidence
from vision_geometry_lab.v19_front_candidates import FrontTracker, RULES as FRONT_RULES


LAB = Path(__file__).resolve().parent
BASE = LAB/'outputs/v19_front_candidates'
UPSTREAM_BASE = upstream.BASE
READY_S = 5.
FRAME_FIELDS = ('normal_offset', 'normal_residual', 'normal_gradient',
                'normal_peak_ratio', 'normal_profiles', 'normal_clipped_fraction',
                'pose_valid')
RULES = dict(ready_s=READY_S, running_state=None,
    profile=dict(profile_size=evidence.PROFILE_SIZE,
        profile_origin=evidence.PROFILE_ORIGIN,
        minimum_startup_pairs=evidence.MINIMUM_STARTUP_PAIRS,
        minimum_startup_reliable=evidence.MINIMUM_STARTUP_RELIABLE,
        minimum_peak_ratio=evidence.MINIMUM_PEAK_RATIO,
        plateau_inner_px=evidence.PLATEAU_INNER_PX,
        plateau_outer_px=evidence.PLATEAU_OUTER_PX,
        minimum_plateau_contrast_gray=evidence.MINIMUM_PLATEAU_CONTRAST_GRAY,
        minimum_baseline_contrast_fraction=evidence.MINIMUM_BASELINE_CONTRAST_FRACTION,
        maximum_clipped_fraction=evidence.MAXIMUM_CLIPPED_FRACTION,
        noise_sigma='max(1,1.4826*MAD_of_complete_startup_plateau_offsets)',
        threshold_px='max(3,3*noise_sigma)',
        reference='offset_minus_V19_complete_plateau_startup_median',
        archive_residual='finite_required_difference_diagnostic_only'),
    front=deepcopy(FRONT_RULES),
    experiment='posthoc_development_image_candidates_not_blind_physical_event_validation',
    startup='0<=time<=5 fixed; do not update state machine until time>5',
    timing_scope='profile_evidence_plus_FrontTracker_only_not_full_pipeline',
    truth_input=False, physical_new_layer_confirmed=False,
    mechanical_revolution_confirmed=False, hardware_access=False,
    deployment_allowed=False)
STOP_RULE = dict(
    after_complete_replay='No threshold tuning in this round; retain every failure and ambiguity.',
    deployment_disallowed_and_cause_audit_required_if=[
        'no_image_candidate', 'incorrectly_early_candidate',
        'multiple_ambiguous_candidates',
        'first_report_later_than_manual_first_revolution_end_in_independent_scoring'],
    physical_t1='A side-contour image candidate must never automatically equal physical new-layer start.',
    manual_references='Scoring only after frozen replay; never an inference input.')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def strict_json(value):
    """JSON-safe tree: explicitly unknown nonfinite numbers become null."""
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, np.generic):
        return strict_json(value.item())
    if isinstance(value, dict):
        return {str(k):strict_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str,int,bool)):
        return value
    raise TypeError('unsupported JSON value '+type(value).__name__)


def _write(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(strict_json(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def _integer_indices(values, name, nonempty=True):
    a=np.asarray(values)
    if a.ndim==1 and not len(a) and not nonempty:
        return np.array([],dtype=np.int64)
    if (a.ndim!=1 or (nonempty and not len(a)) or a.dtype.kind not in 'iu'
            or any(int(x)<0 or int(x)>np.iinfo(np.int64).max for x in a)
            or any(int(y)<=int(x) for x,y in zip(a[:-1],a[1:]))):
        raise ValueError(name+' must be nonnegative strictly increasing integer indices')
    return a.astype(np.int64, copy=True)


def validate_arrays(arrays):
    times=np.asarray(arrays['times_s'],float)
    rows=np.asarray(arrays['rows_y'],float)
    if (times.ndim!=1 or not len(times) or not np.isfinite(times).all()
            or np.any(times<0.) or (len(times)>1 and not np.all(np.diff(times)>0.))):
        raise ValueError('times_s must be finite nonnegative strictly increasing')
    if (rows.ndim!=1 or not len(rows) or not np.isfinite(rows).all()
            or (len(rows)>1 and not np.all(np.diff(rows)>0.))):
        raise ValueError('rows_y must be finite strictly increasing')
    shape=(len(times),len(rows))
    for key in FRAME_FIELDS:
        expected=(len(times),) if key=='pose_valid' else (*shape,131) if key=='normal_profiles' else shape
        a=np.asarray(arrays[key])
        if a.shape!=expected or a.dtype.kind not in 'biuf':
            raise ValueError(key+' has invalid shape or nonnumeric dtype')
    ids=_integer_indices(arrays.get('frame_indices',np.arange(len(times))), 'frame_indices')
    if len(ids)!=len(times):
        raise ValueError('frame_indices length mismatch')
    return times, rows, ids


def validate_source(index, arrays):
    """Validate complete upstream coordinates before any candidate replay."""
    if index.get('output_status')!='complete_observations_only_no_events':
        raise ValueError('upstream observation status is not complete')
    video=index['video']
    for key in ('fps','frame_count','width','height'):
        value=video[key]
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float))
                or not math.isfinite(value) or value<=0.):
            raise ValueError('invalid video '+key)
    if any(int(video[k])!=video[k] for k in ('frame_count','width','height')):
        raise ValueError('noninteger video dimensions or frame count')
    times,rows,ids=validate_arrays(arrays)
    records=index['records']
    if len(records)!=len(times):
        raise ValueError('upstream record count mismatch')
    record_ids=_integer_indices([r['frame_index'] for r in records], 'record frame indices')
    if record_ids[-1]>=video['frame_count']:
        raise ValueError('source frame index outside video')
    if not np.array_equal(record_ids,ids):
        raise ValueError('record/array frame indices mismatch')
    if 'indices' in video and not np.array_equal(_integer_indices(video['indices'],'video indices'),ids):
        raise ValueError('video indices mismatch')
    record_times=np.asarray([r['time_s'] for r in records],float)
    if (not np.isfinite(record_times).all() or not np.allclose(record_times,times,rtol=0.,atol=1e-9)
            or not np.allclose(times,ids/float(video['fps']),rtol=0.,atol=1e-9)):
        raise ValueError('timestamps disagree with records or frame index/fps')
    for record in records:
        digest=record.get('frame_sha256')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid source frame digest')
    return times,rows,ids


def _load_source(upstream_base, side, verify_files=True):
    upstream_base=Path(upstream_base)
    ip=upstream_base/side/'index.json'
    mp=upstream_base/side/'measurement.npz'
    index=_read(ip)
    if verify_files:
        if index['measurement_sha256']!=sha(mp):
            raise ValueError('upstream measurement hash mismatch')
        if index['protocol_sha256']!=sha(upstream_base/'protocol.json'):
            raise ValueError('upstream protocol hash mismatch')
    with np.load(mp,allow_pickle=False) as z:
        arrays={key:z[key].copy() for key in ('times_s','rows_y',*FRAME_FIELDS)}
    arrays['frame_indices']=np.asarray([r['frame_index'] for r in index['records']])
    validate_source(index,arrays)
    return index,arrays


def _source_paths():
    return [Path(__file__), LAB/'v19_profile_evidence.py', LAB/'v19_front_candidates.py']


def _dependency_paths(upstream_base):
    base=Path(upstream_base)
    return [base/'protocol.json',*[base/side/name for side in ('left','right')
                                  for name in ('index.json','measurement.npz')]]


def freeze(base=BASE, upstream_base=UPSTREAM_BASE):
    base,upstream_base=Path(base),Path(upstream_base)
    if base.exists():
        raise FileExistsError('fresh V19 output directory required')
    upstream.verify(upstream_base)
    videos={}
    for side in ('left','right'):
        index,_=_load_source(upstream_base,side)
        videos[side]=index['video']
    protocol=dict(schema='v19_side_front_protocol_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rules=deepcopy(RULES), stop_rule=deepcopy(STOP_RULE),
        upstream_base=str(upstream_base), videos=videos,
        source_hashes={str(p):sha(p) for p in _source_paths()},
        dependencies={str(p):sha(p) for p in _dependency_paths(upstream_base)},
        no_truth_files_read=True,
        limitation='New profile support and temporal candidate rules after V18b; not a one-parameter or blind experiment')
    _write(base/'protocol.json',protocol)
    return protocol


def verify(base=BASE):
    base=Path(base)
    p=_read(base/'protocol.json')
    if p.get('schema')!='v19_side_front_protocol_1' or p['rules']!=RULES:
        raise ValueError('V19 protocol rules changed')
    if p.get('stop_rule')!=STOP_RULE:
        raise ValueError('V19 stop rule changed')
    up=Path(p['upstream_base'])
    upstream.verify(up)
    if set(p['source_hashes'])!={str(x) for x in _source_paths()}:
        raise ValueError('V19 source inventory changed')
    if set(p['dependencies'])!={str(x) for x in _dependency_paths(up)}:
        raise ValueError('upstream dependency inventory changed')
    for path,digest in {**p['source_hashes'],**p['dependencies']}.items():
        if sha(path)!=digest:
            raise ValueError('frozen input or source changed '+path)
    for side in ('left','right'):
        index,_=_load_source(up,side)
        if index['video']!=p['videos'][side]:
            raise ValueError('upstream video metadata changed')
    return p


def _warmup_tracker(now,rows):
    # No tracker method is called, and no future startup-derived threshold is
    # exposed in warmup records. Thus any short input prefix has the same rows.
    states=[dict(index=j,row_y=float(y),phase='warmup',armed=False,quality=False,
        displacement_px=None,high_threshold_px=None,low_threshold_px=None,
        low_start_s=None,low_count=0,last_low_s=None,first_high_s=None,
        high_count=0,confirmed_at_s=None,last_direct_s=None) for j,y in enumerate(rows)]
    return dict(time_s=float(now),state='warmup',reason='startup_model_not_available',
        row_states=states,episode_locked=False,running_state='unknown',new_candidates=[],
        material_identity_confirmed=False,new_layer_confirmed=False,
        mechanical_revolution_confirmed=False,actuator_command=None)


def _candidate_aliases(candidate, frame_index):
    out=deepcopy(candidate)
    out.update(estimated_onset_s=out['estimated_onset'],
        first_available_s=out['first_available'], confirmation_delay_s=out['delay'],
        rows_y=[row['row_y'] for row in out['rows']],
        last_low_first_high_interval_s=list(out['lastlow_firsthigh_interval']),
        first_available_frame_index=int(frame_index))
    return out


def replay_arrays(arrays, frame_indices=None):
    """Replay selected source frame IDs in order; never fill skipped frames.

    Startup model always uses the available 0..5s observations, not selected
    later frames. Warmup never advances the tracker. Deterministic records and
    candidates are separate from nondeterministic processing timings so prefix
    invariance can be tested without conflating wall-clock execution cost.
    """
    times,rows,ids=validate_arrays(arrays)
    if frame_indices is None:
        positions=np.arange(len(times))
    else:
        wanted=_integer_indices(frame_indices,'requested frame indices',nonempty=False)
        lookup={int(value):i for i,value in enumerate(ids)}
        if any(int(i) not in lookup for i in wanted):
            raise ValueError('requested frame index absent from source')
        positions=np.asarray([lookup[int(i)] for i in wanted],dtype=np.int64)
    model=evidence.startup_profile_model(times,arrays,ready_s=READY_S)
    thresholds=np.where(model['supported'],model['threshold_px'],np.nan)
    tracker=FrontTracker(rows,thresholds)
    records=[];candidates=[];timings=[];reasons=Counter()
    valid_cells=0;after_frames=0
    for i in positions:
        now=float(times[i]);source_id=int(ids[i])
        if now<=READY_S:
            quality=np.zeros(len(rows),bool)
            why=np.full(len(rows),'startup_warmup',object)
            residual=np.full(len(rows),np.nan)
            snapshot=_warmup_tracker(now,rows)
            elapsed=0.
        else:
            frame={key:np.asarray(arrays[key])[i] for key in FRAME_FIELDS}
            frame['time_s']=now
            before=time.perf_counter()
            quality_result=evidence.profile_evidence(frame,model)
            quality=quality_result['quality_mask']
            why=quality_result['reason']
            residual=quality_result['profile_residual']
            snapshot=tracker.update(now,residual,quality,running=None)
            elapsed=(time.perf_counter()-before)*1000.
            snapshot['new_candidates']=[_candidate_aliases(c,source_id) for c in snapshot['new_candidates']]
            candidates.extend(snapshot['new_candidates'])
            after_frames+=1;valid_cells+=int(quality.sum())
        reasons.update(map(str,why))
        records.append(strict_json(dict(time_s=now,frame_index=source_id,
            quality_mask=quality,quality_reason=why,profile_residual=residual,tracker=snapshot)))
        timings.append(elapsed)
    return strict_json(dict(startup_model=model,records=records,candidates=candidates,
        quality_counts=dict(frames=len(positions),rows=len(rows),after_startup_frames=after_frames,
            after_startup_cells=after_frames*len(rows),profile_consistent_cells=valid_cells,
            unknown_after_startup_cells=after_frames*len(rows)-valid_cells,
            reason_counts=dict(reasons)),
        timings_ms=timings,timing_scope=RULES['timing_scope'],
        running_state='unknown',material_identity_confirmed=False,
        physical_new_layer_confirmed=False,new_layer_confirmed=False,
        mechanical_revolution_confirmed=False,hardware_access=False,
        deployment_allowed=False,actuator_command=None))


def run(base=BASE,session='left'):
    if session not in ('left','right'):
        raise ValueError('session must be left or right')
    base=Path(base);p=verify(base)
    dest=base/session
    if dest.exists():
        raise FileExistsError('do not overwrite a prior V19 run')
    index,arrays=_load_source(p['upstream_base'],session)
    dest.mkdir()
    _write(dest/'started.json',dict(status='started_not_proof_of_completion',session=session))
    result=replay_arrays(arrays)
    # A complete output is published only if every frozen dependency still
    # matches after replay. A failed check leaves only the start marker.
    verify(base)
    up=Path(p['upstream_base'])
    result.update(status='complete',protocol_sha256=sha(base/'protocol.json'),
        source_measurement_sha256=sha(up/session/'measurement.npz'),
        source_index_sha256=sha(up/session/'index.json'),session=session,video=index['video'])
    _write(dest/'predictions.json',result)
    print('COMPLETE V19',session,'frames',len(result['records']),
          'image candidates',len(result['candidates']),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','verify','run'))
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--upstream-base',type=Path,default=UPSTREAM_BASE)
    parser.add_argument('--session',choices=('left','right'),default='left')
    args=parser.parse_args()
    if args.stage=='freeze':
        freeze(args.base,args.upstream_base)
    elif args.stage=='verify':
        verify(args.base);print('V19 frozen sources and archived inputs verified')
    else:
        run(args.base,args.session)


if __name__=='__main__':
    main()
