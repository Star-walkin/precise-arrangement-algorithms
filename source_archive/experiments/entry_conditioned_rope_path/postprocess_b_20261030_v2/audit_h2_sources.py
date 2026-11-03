"""Read-only allowlisted source/cached-response audit; no labels or detectors."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib,math,sys,time
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
H1=HERE.parent/'postprocess_b_20261030_v1'
DATA=ROOT/'数据资产/05_派生与实验结果/camera_condition_images_20261009_v2'
A7=ROOT/'数据资产/05_派生与实验结果/camera_condition_free_straight_20261030_v7/all_001'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def match(seed,c):
    p=np.asarray(seed['xy'],float);d=p[-1]-p[0];d/=np.linalg.norm(d)
    axis=np.asarray(c['direction_xy'],float);axis/=np.linalg.norm(axis)
    angle=math.degrees(math.acos(float(np.clip(abs(d@axis),0,1))))
    offset=float(np.median(abs((p-np.asarray(c['origin_xy']))@np.array([-axis[1],axis[0]]))))
    threshold=max(6.,2*c['width_px'])
    return dict(candidate_id=c['candidate_id'],compatible=bool(angle<=6 and offset<=threshold),undirected_angle_deg=angle,perpendicular_distance_px=offset,threshold_px=threshold)
def main():
    tick=time.perf_counter();proto=read(H1/'protocol_inputs.json');jobs={j['id']:j for j in proto['jobs']}
    rows=list(csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig')))
    assert len(rows)==400 and len(jobs)==292 and sum(len(j['seeds']) for j in jobs.values())==299
    statuses=Counter();status_without_old=Counter();sources=[];new_images=[];unmatched=[];new_axes=[];saved_conflicts=[]
    for r in rows:
        assert '20260925' not in r['source_file']+r['output_file']
        assert not any(x in r['dataset_use_status'].lower() for x in ('reserved','heldout'))
        sid=Path(r['output_file']).stem;file=A7/'记录'/(r['camera_role']+'_camera')/(sid+'.json');v=read(file)['result']
        accepted=[c for c in v['candidates'] if c['accepted']];statuses[v['status']]+=1
        if sid not in jobs:
            status_without_old[v['status']]+=1
            if accepted:new_images.append(sid)
        saved=read(H1/'all_001/记录'/(sid+'.json')) if sid in jobs else None
        if saved:
            for e in saved['entries']:
                if e['automatic_entry_conflict']:saved_conflicts.append({'id':sid,'entry_index':e['entry_index'],'seed_id':e['seed']['entry_id']})
        matched=[False]*len(accepted)
        for k,s in enumerate(jobs.get(sid,{}).get('seeds',[])):
            checks=[match(s,c) for c in accepted]
            matched=[a or b['compatible'] for a,b in zip(matched,checks)]
            if accepted and not any(x['compatible'] for x in checks):
                unmatched.append({'id':sid,'entry_index':k,'seed_id':s['entry_id'],'accepted_axis_comparison':checks,'H1_conflict_flag':saved['entries'][k]['automatic_entry_conflict']})
        for c,ok in zip(accepted,matched):
            if not ok:new_axes.append({'id':sid,'candidate_id':c['candidate_id'],'origin_xy':c['origin_xy'],'direction_xy':c['direction_xy'],'width_px':c['width_px'],'image_already_has_old_response':sid in jobs})
        sources.append({'id':sid,'camera_role':r['camera_role'],'source_group':r['source_group'],'status':v['status'],'accepted_axis_count':len(accepted),'old_H1_condition_count':len(jobs.get(sid,{}).get('seeds',[])),'A7_path':str(file),'A7_sha256':sha(file),'original_image_path':str(DATA/r['output_file']),'original_image_sha256_from_manifest':r['output_sha256']})
    expected={(r['id'],r['entry_index']):r for r in read(H1/'numeric_response_audit.json')['checks']}
    floats=[]
    for j in proto['jobs']:
        inference=read(H1/'responses'/j['id']/'inference.json')
        assert inference['model_sha256']==proto['model_B']['sha256'] and inference['image_sha256']==j['image_sha256']
        for k,s in enumerate(j['seeds']):
            meta=inference['entry_conditions'][k];assert meta['seed']==s
            path=H1/'responses'/j['id']/f'entry_{k:02d}.npz';digest=sha(path)
            assert digest==meta['response_sha256']==expected[(j['id'],k)]['response_sha256']
            with np.load(path,allow_pickle=False) as z:
                assert z.files==['response'];a=z['response'];assert a.shape==(975,2048) and a.dtype==np.float32
                assert np.isfinite(a).all() and 0<=a.min()<=a.max()<=1
                bounds=[float(a.min()),float(a.max())]
            floats.append({'id':j['id'],'entry_index':k,'path':str(path),'sha256':digest,'range':bounds,'exact_seed_preserved':True})
    actual=list((H1/'responses').glob('*/entry_*.npz'));assert len(actual)==len(floats)==299
    pilot=list(proto['pilot_ids']);pilot[pilot.index('dual_20260924_111143__pair0075__right')]='R_run_20260923_01__f001274__right'
    reasons=['left bright-band premature stop','left weak/transition context','left run-stop lighting/context','left static comparison','left entry-to-surface bend','left ruler clear reference',
             'tool conflict and two legacy entries','guide-rod occlusion and recovered straight section','multiple old entries merged by A7 straight axis','visible straight gap beyond guide rod','right attached-side curved continuation','accepted A7 axis absent from legacy B seed; direction unresolved']
    byid={r['id']:r for r in sources}
    payload={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','source_sha256':sha(__file__),'command':sys.argv,
        'scope':'read-only metadata and original B cache; no S1 labels, reserved data, image decoding, training or devices',
        'manifest':{'path':str(DATA/'manifest.csv'),'sha256':sha(DATA/'manifest.csv'),'images':400,'camera_counts':dict(Counter(r['camera_role'] for r in rows))},
        'H1':{'formal_run':str(H1/'all_001'),'protocol_sha256':sha(H1/'protocol_inputs.json'),'source_images':400,'eligible_images':292,'entry_conditions':299,'old_unavailable_images':108,'float_cache_complete':True,'float_count':299,'all_arrays_finite_float32_975x2048':True,'all_cached_file_hashes_match_frozen_numeric_audit':True,'model_B':proto['model_B'],'model_sha_matches':sha(proto['model_B']['path'])==proto['model_B']['sha256'],'floats':floats},
        'A7':{'formal_run':str(A7),'status_counts':dict(statuses),'accepted_axes':sum(r['accepted_axis_count'] for r in sources),'accepted_axes_per_detected_image':1,'old_no_entry_108_status_counts':dict(status_without_old),'accepted_new_images_outside_H1_292':new_images,'accepted_axes_without_compatible_old_seed':new_axes,'old_seed_conditions_incompatible_with_accepted_A7':unmatched,'saved_H1_conflicts':saved_conflicts,'match_rule':'unchanged H1 undirected angle<=6deg and perpendicular median<=max(6px,2*A7width); geometric compatibility not identity truth'},
        'schemas':{'H1_float':'responses/<id>/entry_00.npz: response float32[975,2048], native crop half-resolution; first index0; seed and image/model hashes in inference.json',
            'H1_path':'all_001/记录/<id>.json entries[].candidates points_xy original xy; f3_tail_review ONLY crop-half xy, original=2*xy+[0,1050]',
            'A7':'all_001/记录/<role>_camera/<id>.json result.candidates accepted/plausible; center_xy,left_xy,right_xy original coords aligned with visible; pieces are [start,end) indices, not coordinates',
            'A7_direction':'unoriented axis; full center_xy extends beyond accepted measured pieces; use visible/pieces, do not fill long gaps or treat axis sign as winding direction',
            'F3':'observed merged minus inferred; boundaries are not centers; only adjacent supported boundaries form conservative center candidates; saved pitch is pixel proxy, not physical diameter',
            'coordinate_mapping':'A7/original -> crop_half=(xy-[0,1050])/2; fullhalf F3 arrays crop [525:]; model response is not fullhalf canvas'},
        'pilot12':[{'id':sid,'camera_role':byid[sid]['camera_role'],'source_group':byid[sid]['source_group'],'reason':why,'scope':'open development diagnostic, not independent test'} for sid,why in zip(pilot,reasons)],
        'full_scan_unknown_sentinel':'L_run_stop_run_20260922_01__f000976__left; A7 AMBIGUOUS, no old B entry; do not force a positive',
        'freeze_checkpoints':['400 original manifest IDs and camera200/200 exact; no silent subset denominator','old299 cached B floats+seed/hash immutable; any new seed/direction gets separate frozen-checkpoint forward','new accepted A7 axis Rrun1274 has no old matching condition; its two directions must stay separate and uncertain','6 old seed conditions conflict with accepted A7, whereas old H1 flag only5; retain failure records','A7 measured pieces only, gaps remain unknown; no SAM2 mask exclusion of free rope','F3 observed/inferred sources distinguished; boundary-to-center conversion and scoring parameters frozen before all400','maximum candidates and sampling budgets; preserve direction/identity ambiguity and missing paths','fixed0..1 heatmap; selected entry explicit; line/ribbon display not true body-mask validation','old12 development selection modified onlyRrun1274; full400 pictures not independent accuracy denominator'],
        'all400_sources':sources,'elapsed_seconds':time.perf_counter()-tick}
    target=HERE/'h2_source_audit.json'
    with target.open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':'passed','old_float_files':len(floats),'new_A7_images':len(new_images),'new_A7_axes_without_seed':len(new_axes),'old_seed_conflicts':len(unmatched),'saved_H1_conflicts':len(saved_conflicts),'audit':str(target)},ensure_ascii=False))
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
