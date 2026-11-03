"""Rerun frozen H6 geometry from new raw responses; never rasterize paths as heat."""
import argparse,concurrent.futures,gzip,json,sys,time,traceback
from pathlib import Path
import cv2,numpy as np
from common import HERE,H6,load,dump,sha
from legacy_h6 import get
H=get()
def process(job,responses,out,variant):
    tick=time.time();priors=H.load_surface_priors(dict(job,seeds=job['original_seeds']))
    original=priors['original_half'];gray=cv2.cvtColor(original[525:],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    entries=[];heats=[];tracksets=[]
    for k,(seed,row) in enumerate(zip(job['seeds'],responses)):
        rp=Path(row['response_file']);assert sha(rp)==row['response_sha256'] and row['entry_index']==k
        assert row['seed_xy']==seed['xy']
        with np.load(rp,allow_pickle=False) as z:heat=z['response'].copy()
        assert heat.shape==(975,2048) and np.isfinite(heat).all() and 0<=heat.min()<=heat.max()<=1
        tracks=H.build_tracks(priors,heat,gray);full_free=H.describe_free(priors,seed,heat)
        free,projection=H.prepare_task_free(full_free,tracks['tracks'])
        linked=H.connect(gray,heat,free,tracks['tracks'],[],priors['surface_half'][525:])
        for candidate in linked['candidates']:candidate['free_axis_id']=free.get('free_axis_id','axis_00')
        linked['alternative_axes_used_for_connection']=False
        linked['retained_axis_count']=int(bool(free['points_xy']))+len(free.get('alternative_free_axes',[]))
        entries.append(dict(entry_index=k,seed=seed,response_file=str(rp),response_sha256=sha(rp),
            free=free,full_free_observation=full_free,free_projection=projection,connection=linked,
            track_diagnostics=tracks['diagnostics'],candidate_track_geometry={t['track_id']:t['points_xy'] for t in tracks['tracks'] if t['track_id'] in {c['track_id'] for c in linked['candidates']}}))
        heats.append(heat);tracksets.append(tracks)
    H.task_direction_feasibility(entries,priors['sam2_surface_half'][525:])
    ranked=[(c['rank_score'],k,j,c) for k,e in enumerate(entries) for j,c in enumerate(e['connection']['candidates'])]
    ranked.sort(key=lambda x:(entries[x[1]]['task_direction']['selection_allowed'],x[0],x[3]['metrics'].get('free',{}).get('quality',0)),reverse=True)
    best=ranked[0] if ranked else None;selected=best[1] if best else 0;shown=ranked[:3]
    multi=np.rint(original.astype(np.float32)*.65).astype(np.uint8)
    for track in tracksets[selected]['tracks']:
        points=np.asarray(track['points_xy']);inferred=np.asarray(track['f3_pair_inferred'],bool)
        for j in range(1,len(points)):
            weak=bool(inferred[j] or inferred[j-1])
            if not weak or j%5<3:H.poly(multi,points[j-1:j+1],(110,110,110) if weak else (170,170,170),1)
    for n,(_,ei,ci,c) in reversed(list(enumerate(shown))):H.poly(multi,c['points_xy'],H.COLORS[n],3)
    for ei,e in enumerate(entries):
        H.poly(multi,(np.asarray(e['seed']['xy'])-[0,1050])*.5,(255,255,255),3)
        xy=tuple(np.rint(np.asarray(e['seed']['xy'][0])*.5).astype(int));cv2.putText(multi,'E'+str(ei+1),xy,cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    ambiguity=entries[selected]['connection'].get('ambiguous',True) or bool(job['seeds'][selected].get('direction_ambiguous',False))
    if len(ranked)>1 and ranked[0][1]!=ranked[1][1] and ranked[0][0]-ranked[1][0]<H.PARAMETERS['ambiguity_margin']:ambiguity=True
    final=H.draw_final(original,best[3],ambiguity) if best else original.copy()
    quality='候选接续待确认' if best else '没有接续候选'
    if best and ambiguity:quality='接续有歧义'
    if best and best[3]['kind']=='free_fragment_no_resolved_connection':quality='自由段保留；完整接续未解决'
    details='；'.join(f'{n+1}:E{ei+1} 分={score:.3f}'+(' 弱证据未采用' if c.get('continuation_eligible') is False else '') for n,(score,ei,ci,c) in enumerate(shown)) or '不补造路径'
    panels=[H.label(original,'1 原图',[job['id'],'同一开放数据；原自动短入口；无远端路径输入']),
        H.label(H.heat_panel(original,heats[selected]),f'2 新热力图 {variant} | E{selected+1}',['网络原始响应0–1；灰区未计算；固定色标','没有把后处理路径画回热图；强响应不是身份概率']),
        H.label(multi,'3 冻结H6中心轨迹与接续候选',[details,'灰实线/虚线：F3原支持/推断；贴面锁同轨并拒绝跨轨']),
        H.label(final,'4 路径结果 | '+quality,['绿：原图+热图；黄：热图；浅蓝：A7；青/紫：接续；橙虚线：弱证据','后处理参数与H6一致；新响应未经身份校准'])]
    dest=out/'四栏检测结果'/(job['id']+'.jpg');H.save(dest,np.concatenate(panels,axis=1))
    with np.load(job['response_files'][selected],allow_pickle=False) as z:old=z['response']
    compare=np.concatenate([panels[0],H.label(H.heat_panel(original,old),f'2 原B热力图 | E{selected+1}',['冻结step750；0–1固定色标','与右栏使用同一个入口条件']),panels[1]],axis=1)
    compare_path=out/'原图_旧B_新热力图'/(job['id']+'.jpg');H.save(compare_path,compare)
    record=dict(version='IR-H7.0.0',id=job['id'],heatmap_variant=variant,source_image=job['image_path'],source_image_sha256=job['image_sha256'],
        coordinate_system='crop_half_xy; original=2*xy+[0,1050]',entries=entries,selected_entry=selected,selected_candidate=best[2] if best else None,
        displayed_candidates=[dict(entry=k,candidate=j,score=s,task_direction_allowed=entries[k]['task_direction']['selection_allowed']) for s,k,j,c in shown],
        surface_tracks=tracksets[selected],input_dependency_sha256=priors['input_dependency_sha256'],response_entry_conditions_merged=False,
        complete_identity_verified=False,postprocess_parameters_changed=False,labels_read=False,devices=False,
        old_B_comparison=dict(entry_index=selected,path=job['response_files'][selected],sha256=sha(job['response_files'][selected])),elapsed_seconds=time.time()-tick)
    with gzip.open(out/'记录'/(job['id']+'.json.gz'),'wt',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False)
    return dict(id=job['id'],entry_count=len(entries),selected_entry=selected,kind=best[3]['kind'] if best else None,
        length_original_px=float(2*np.linalg.norm(np.diff(best[3]['points_xy'],axis=0),axis=1).sum()) if best else 0,
        ambiguous=ambiguity,termination=best[3]['termination'] if best else None,output=str(dest),sha256=sha(dest),comparison=str(compare_path),comparison_sha256=sha(compare_path),seconds=time.time()-tick)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--subset',choices=['pilot','all'],required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    sys.stdout.reconfigure(encoding='utf8');cv2.setNumThreads(1);protocol=load(HERE/'protocol.json');responses=load(HERE/'batch_inference/manifest.json')
    choice=load(HERE/'selection.json');variant=choice['selected_variant'];assert responses['variant']==variant
    assert all(sha(p)==s for p,s in choice['source_hashes'].items()),'source changed after model selection'
    lookup={r['id']:r['entries'] for r in responses['rows']};jobs=protocol['jobs']
    if a.subset=='pilot':jobs=[j for j in jobs if j['id'] in protocol['gallery_fixed_diagnostic_ids']]
    assert all(j['id'] in lookup for j in jobs)
    out=HERE/'postprocess'/a.run;out.mkdir(parents=True,exist_ok=False);(out/'记录').mkdir()
    sources={str(HERE/n):sha(HERE/n) for n in ['common.py','legacy_h6.py','postprocess_heatmap.py']}
    dump(out/'run_start.json',dict(command=sys.argv,planned=len(jobs),input_ids=[j['id'] for j in jobs],sources=sources,
        frozen_H6=load(H6/'freeze_final.json'),protocol_sha256=sha(HERE/'protocol.json'),selection_sha256=sha(HERE/'selection.json'),
        response_manifest_sha256=sha(HERE/'batch_inference/manifest.json'),parameters=H.PARAMETERS,topology_parameters=H.TOPOLOGY_PARAMETERS,
        variant=variant,new_training_during_postprocess=False))
    rows=[];failures=[];tick=time.time()
    with (out/'progress.jsonl').open('x',encoding='utf8') as f,concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        pending={ex.submit(process,j,lookup[j['id']],out,variant):j['id'] for j in jobs}
        for future in concurrent.futures.as_completed(pending):
            try:r=future.result();rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            except Exception:failures.append(dict(id=pending[future],traceback=traceback.format_exc()))
            if (len(rows)+len(failures))%12==0 or len(rows)+len(failures)==len(jobs):print(json.dumps(dict(done=len(rows),total=len(jobs),failures=len(failures),seconds=round(time.time()-tick,1))),flush=True)
    dump(out/'summary.json',dict(completed=len(rows),planned=len(jobs),rows=rows,failures=failures,exit_code=int(bool(failures)),seconds=time.time()-tick,
        source_unchanged=all(sha(p)==s for p,s in sources.items()),identity_accuracy_not_measured=True))
    sys.exit(bool(failures))
if __name__=='__main__':main()
