"""Posthoc fixed-denominator audit of archived V19 profile candidates only.

No events are emitted, no model/threshold is fitted, and no raw video or truth
file is opened. A common-support subset uses future coverage and is explicitly
diagnostic only; it cannot be an online selection or material-identity claim.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


LAB=Path(__file__).resolve().parent
SOURCE=LAB/'outputs/v19_front_candidates'
BASE=LAB/'outputs/v20_material_audit/support_audit'
RULES=dict(schema='v20_support_denominator_rules_1',source='archived_V19_predictions_only',
    start_s=0.,end_s=60.,bin_width_s=5.,startup_ready_s=5.,
    bins='[0,5),[5,10),...,[55,60]; last bin includes 60.0',
    original_row_count=128,spatial_partitions=[[0,32],[32,64],[64,96],[96,128]],
    quality='archived_quality_mask_and_finite_profile_residual; never state-machine latched high',
    high='direct quality true and profile_residual >= original per-row V19 high threshold',
    threshold='archived threshold_px, with unchanged tracker minimum 3 pixels',
    primary_denominator='same frozen startup supported row indices in every bin',
    unknown='not low; high/all slots is an observed-high lower bound, not a physical high fraction',
    common_support_fraction=.5,
    common_all_bins='startup supported rows with >=50% direct quality in every 0..60 bin',
    common_poststartup_bins='startup supported rows with >=50% direct quality in every bin starting at >=5s',
    common_support_uses_future=True,common_support_online_usable=False,
    event_times_read=False,events_implemented=False,new_thresholds_fitted=False,
    material_identity_confirmed=False,physical_new_layer_confirmed=False,hardware_access=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def ratio(count,total):
    return None if total==0 else float(count/total)


def _booleans(values,name,shape):
    a=np.asarray(values)
    if a.shape!=shape or a.dtype.kind!='b':
        raise ValueError(name+' must be explicit bool values of expected shape')
    return a


def unpack(prediction):
    if prediction.get('status')!='complete':
        raise ValueError('complete archived predictions are required')
    records=prediction['records'];model=prediction['startup_model']
    nrows=RULES['original_row_count'];nframes=len(records)
    if nframes!=601:
        raise ValueError('this fixed audit requires full 0..60 s at 10 Hz (601 frames)')
    times=np.asarray([r['time_s'] for r in records],float)
    indices=np.asarray([r['frame_index'] for r in records])
    fps=prediction['video']['fps']
    if (isinstance(fps,bool) or not isinstance(fps,(int,float)) or not np.isfinite(fps)
            or fps!=10. or not np.array_equal(indices,np.arange(nframes))
            or not np.array_equal(times,np.arange(nframes)/10.)):
        raise ValueError('archived source coordinates are not the fixed full 10 Hz window')
    base=_booleans(model['supported'],'startup supported',(nrows,))
    threshold=np.asarray(model['threshold_px'],float)
    if threshold.shape!=(nrows,) or not np.isfinite(threshold[base]).all() or np.any(threshold[base]<=0.):
        raise ValueError('startup thresholds malformed')
    threshold=np.maximum(threshold,3.)
    quality=_booleans([r['quality_mask'] for r in records],'quality_mask',(nframes,nrows))
    residual=np.asarray([r['profile_residual'] for r in records],float)
    if residual.shape!=(nframes,nrows):
        raise ValueError('profile residual shape mismatch')
    if np.any(quality & ~np.isfinite(residual)) or np.any(quality[:,~base]):
        raise ValueError('direct quality cannot validate unknown residual or unsupported startup row')
    if np.any(quality[times<=RULES['startup_ready_s']]):
        raise ValueError('startup rows must remain warmup, not direct replay observations')
    rows=np.asarray([s['row_y'] for s in records[0]['tracker']['row_states']],float)
    if rows.shape!=(nrows,) or not np.isfinite(rows).all() or not np.all(np.diff(rows)>0.):
        raise ValueError('source row coordinates malformed')
    for r in records:
        current=np.asarray([s['row_y'] for s in r['tracker']['row_states']],float)
        if not np.array_equal(current,rows):
            raise ValueError('original row coordinates changed during replay')
    for key in ('material_identity_confirmed','physical_new_layer_confirmed','hardware_access'):
        if prediction.get(key) is not False:
            raise ValueError('unsupported physical/deployment claim in archived predictions')
    return times,rows,base,threshold,quality,residual


def fixed_bins(times):
    result=[]
    for start in range(0,60,5):
        stop=start+5
        chosen=(times>=start)&((times<stop) if stop<60 else (times<=stop))
        result.append((start,stop,chosen))
    return result


def aggregate(select,row_mask,quality,high,residual):
    q=quality[np.ix_(select,row_mask)]
    h=high[np.ix_(select,row_mask)]
    d=residual[np.ix_(select,row_mask)]
    valid=int(q.sum());slots=int(q.size);n_high=int(h.sum())
    values=d[q]
    return dict(rows=int(row_mask.sum()),frames=int(select.sum()),all_slots=slots,
        direct_quality_slots=valid,unknown_slots=slots-valid,observed_high_slots=n_high,
        quality_coverage=ratio(valid,slots),high_given_direct_quality=ratio(n_high,valid),
        observed_high_per_all_slots=ratio(n_high,slots),
        median_direct_residual_px=float(np.median(values)) if values.size else None)


def audit_prediction(prediction):
    times,rows,base,threshold,q,resid=unpack(prediction)
    high=q & (resid>=threshold[None,:])
    bins=fixed_bins(times)
    coverage=np.asarray([q[mask].mean(axis=0) for _,_,mask in bins])
    common_all=base & np.all(coverage>=RULES['common_support_fraction'],axis=0)
    after_bins=np.asarray([start>=RULES['startup_ready_s'] for start,_,_ in bins])
    common_after=base & np.all(coverage[after_bins]>=RULES['common_support_fraction'],axis=0)
    masks=dict(all_original_rows=np.ones(len(rows),bool),startup_supported_rows=base,
        common_all_12_bins_posthoc=common_all,common_5_to_60_bins_posthoc=common_after)
    bucket_records=[]
    for start,stop,chosen in bins:
        perrow=[]
        for j,y in enumerate(rows):
            selected=np.zeros(len(rows),bool);selected[j]=True
            metrics=aggregate(chosen,selected,q,high,resid)
            metrics.update(original_row_index=j,row_y=float(y),
                startup_supported=bool(base[j]),
                high_threshold_px=float(threshold[j]) if np.isfinite(threshold[j]) else None)
            perrow.append(metrics)
        scopes={name:aggregate(chosen,mask,q,high,resid) for name,mask in masks.items()}
        spatial=[]
        for first,last in RULES['spatial_partitions']:
            fixed=np.zeros(len(rows),bool);fixed[first:last]=True
            spatial.append(dict(original_index_range=[first,last],
                range_interpretation='first inclusive, last exclusive',
                all_original=aggregate(chosen,fixed,q,high,resid),
                startup_supported=aggregate(chosen,fixed&base,q,high,resid),
                common_poststartup=aggregate(chosen,fixed&common_after,q,high,resid)))
        bucket_records.append(dict(start_s=start,stop_s=stop,last_endpoint_inclusive=stop==60,
            contains_startup_warmup=bool(np.any(chosen & (times<=5.))),
            frames=int(chosen.sum()),scopes=scopes,spatial_partitions=spatial,per_original_row=perrow))
    startup=prediction['startup_model']
    return dict(session=prediction.get('session'),frames=len(times),original_rows=len(rows),
        scope='candidate-profile denominator audit, not material or event truth',
        startup_support=dict(available_after_s=startup.get('ready_s'),
            supported_count=int(base.sum()),supported_original_indices=np.flatnonzero(base).tolist(),
            supported_rows_y=rows[base].tolist(),
            denominator_is_same_row_set_in_every_bucket=True,
            warmup_quality_false_by_design=True,
            profile_offset_count=startup.get('profile_offset_count'),
            reliable_profile_count=startup.get('reliable_profile_count')),
        posthoc_common_support=dict(uses_future=True,online_usable=False,
            minimum_per_bucket_direct_quality_fraction=.5,
            all_12_buckets_count=int(common_all.sum()),
            all_12_buckets_original_indices=np.flatnonzero(common_all).tolist(),
            poststartup_11_buckets_count=int(common_after.sum()),
            poststartup_11_buckets_original_indices=np.flatnonzero(common_after).tolist(),
            why_two_definitions='The 0..5 startup bucket is warmup and necessarily makes the all-bucket intersection empty.'),
        all_video_scopes={name:aggregate(np.ones(len(times),bool),mask,q,high,resid)
                          for name,mask in masks.items()},
        fixed_5s_buckets=bucket_records,
        interpretation_limits=[
            'High/all slots is an observation lower bound; unknown slots are not evidence of low position.',
            'Even common rows have different missing timestamps; common support controls row composition, not all selection bias.',
            'No record is labelled normal production or true new-layer growth; no manual event files are read.',
            'Residuals measure a selected image-profile edge, not necessarily the same material surface.',
            'Persistent bucket differences do not supply event timing or mechanical revolution phase.'])


def freeze(base=BASE,source=SOURCE):
    base,source=Path(base),Path(source)
    if base.exists():
        raise FileExistsError('fresh support audit directory required')
    paths=[source/'protocol.json',source/'left/predictions.json',source/'right/predictions.json']
    write(base/'protocol.json',dict(schema='v20_support_audit_protocol_1',rules=RULES,
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(source),
        input_hashes={str(p):sha(p) for p in paths},source_sha256=sha(__file__),
        rule_status='fixed before reading candidate statistics in this audit',
        posthoc_diagnostic_not_new_event_validation=True))


def verify(base=BASE):
    p=read(Path(base)/'protocol.json')
    if p['rules']!=RULES or p['source_sha256']!=sha(__file__):
        raise ValueError('audit rules or code changed')
    expected={str(Path(p['source'])/name) for name in (
        'protocol.json','left/predictions.json','right/predictions.json')}
    if set(p['input_hashes'])!=expected:
        raise ValueError('input inventory changed')
    for path,digest in p['input_hashes'].items():
        if sha(path)!=digest:
            raise ValueError('archived prediction changed')
    return p


def run(base=BASE):
    base=Path(base);p=verify(base)
    if (base/'audit.json').exists():
        raise FileExistsError('do not overwrite prior audit')
    source=Path(p['source']);result={}
    for side in ('left','right'):
        prediction=read(source/side/'predictions.json')
        if prediction.get('session')!=side or prediction.get('protocol_sha256')!=sha(source/'protocol.json'):
            raise ValueError('prediction session/protocol provenance mismatch')
        result[side]=audit_prediction(prediction)
    verify(base)
    write(base/'audit.json',dict(schema='v20_support_audit_1',status='complete',
        protocol_sha256=sha(base/'protocol.json'),input_hashes=p['input_hashes'],rules=RULES,
        source_sha256=p['source_sha256'],sessions=result,
        raw_images_read=False,truth_files_read=False,events_implemented=False,
        material_identity_confirmed=False,hardware_access=False))
    print('COMPLETE archived support audit',base,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','verify','run'))
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--source',type=Path,default=SOURCE)
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args.base,args.source)
    elif args.stage=='verify':verify(args.base);print('Support audit inputs verified')
    else:run(args.base)


if __name__=='__main__':main()
