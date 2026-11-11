"""H6 gallery: source F3 centers and topology-constrained continuation."""
from pathlib import Path
import argparse,concurrent.futures,gzip,json,sys,time,traceback,platform,shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import HERE,H1,H2,H3,H4,H5,load,dump,sha,inputs
from surface_tracks import load_surface_priors,build_tracks,PARAMETERS as SURFACE_PARAMETERS
from topology import prepare_task_free,PARAMETERS as TOPOLOGY_PARAMETERS
from heat_free import describe_free,PARAMETERS as HEAT_PARAMETERS
from connect_paths import connect,PARAMETERS

COLORS=[(40,235,55),(255,220,30),(220,70,240)]
def save(p,im):
    p.parent.mkdir(parents=True,exist_ok=True)
    cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(p))
def label(im,title,lines):
    h,w=im.shape[:2];board=Image.new('RGB',(w,h+155),(246,248,251));board.paste(Image.fromarray(im[:,:,::-1]),(0,155));d=ImageDraw.Draw(board)
    f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',31);s=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    d.text((20,8),title,font=f,fill=(20,30,40))
    for i,l in enumerate(lines):d.text((20,58+40*i),l,font=s,fill=(50,65,80))
    return np.asarray(board)[:,:,::-1].copy()
def poly(im,points,color,thickness=2,dashed=False):
    p=np.rint(np.asarray(points)+[0,525]).astype(np.int32)
    if not dashed:cv2.polylines(im,[p],False,color,thickness,cv2.LINE_AA)
    else:
        for k in range(1,len(p)):
            if k%5<3:cv2.line(im,tuple(p[k-1]),tuple(p[k]),color,thickness,cv2.LINE_AA)
def draw_final(original,candidate,ambiguous):
    out=original.copy();p=np.asarray(candidate['points_xy'])+[0,525]
    for k in range(1,len(p)):
        s=candidate['supported'][k] and candidate['supported'][k-1]
        stage=candidate['stage'][k]
        color=(40,230,45) if stage=='free' else (255,210,35) if stage=='transition' else (230,70,230)
        heat_only=candidate['point_status'][k]=='heatmap_supported' and candidate['point_status'][k-1] in ('heatmap_supported','joint_heat_image_supported','image_supported')
        raw_a7=candidate['point_status'][k]=='raw_a7_supported' and candidate['point_status'][k-1] in ('raw_a7_supported','image_supported','joint_heat_image_supported')
        if raw_a7:color=(255,180,60)
        elif heat_only:color=(20,220,250)
        elif not s:color=(0,155,255)
        if s or heat_only or raw_a7 or k%5<3:cv2.line(out,tuple(np.rint(p[k-1]).astype(int)),tuple(np.rint(p[k]).astype(int)),color,3,cv2.LINE_AA)
    cv2.circle(out,tuple(np.rint(p[-1]).astype(int)),7,(255,255,255),2,cv2.LINE_AA)
    return out
def heat_panel(original,response):
    out=np.full_like(original,35);out[525:]=cv2.applyColorMap(np.rint(response*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    bar=cv2.applyColorMap(np.linspace(255,0,420).astype(np.uint8)[:,None],cv2.COLORMAP_TURBO)
    out[65:485,1850:1890]=np.repeat(bar,40,axis=1)
    for y,s in [(80,'1.0'),(280,'0.5'),(480,'0.0')]:cv2.putText(out,s,(1900,y),cv2.FONT_HERSHEY_SIMPLEX,.85,(240,240,240),2)
    return out

def task_direction_feasibility(entries,surface_mask):
    """Observed outside-to-spool progression for paired A7 directions only.

    This is a task-direction constraint, not contact/identity confirmation.
    No line is extrapolated into the mask; no conditional maps are merged.
    """
    groups={}
    for ei,e in enumerate(entries):
        f=e['free'];p=np.asarray(f['points_xy'],float).reshape(-1,2)
        ev=dict(selection_allowed=True,basis='no_observed_opposite_direction_contrast',
            heat_inside_original_px=0.,longest_heat_inside_original_px=0.,joint_inside_original_px=0.,
            observed_outside_then_inside=False,target_identity_confirmed=False)
        if len(p)>1:
            inside=cv2.remap(surface_mask.astype(np.float32),p[:,0].astype(np.float32)[None],p[:,1].astype(np.float32)[None],cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).ravel()>.5
            post=np.asarray(f['point_status'])!='entry_condition';heat=np.asarray(f['protect_support'],bool)&post
            joint=np.asarray(f['supported'],bool)&post;ds=2*np.linalg.norm(np.diff(p,axis=0),axis=1)
            hi=inside&heat;ji=inside&joint;ho=~inside&heat
            changes=np.diff(np.r_[False,hi,False].astype(np.int8))
            runs=zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))
            longest=max((float(ds[a:b-1].sum()) for a,b in runs),default=0.)
            ev.update(heat_inside_original_px=float(ds[hi[:-1]&hi[1:]].sum()),
                joint_inside_original_px=float(ds[ji[:-1]&ji[1:]].sum()),longest_heat_inside_original_px=longest,
                observed_outside_then_inside=bool(ho.any() and hi.any() and np.flatnonzero(ho)[0]<np.flatnonzero(hi)[0]))
        e['task_direction']=ev
        aid=e['seed'].get('a7_candidate_id')
        if aid:groups.setdefault(aid,[]).append(ei)
    for aid,indices in groups.items():
        for good in indices:
            a=entries[good];ea=a['task_direction']
            if not (ea['observed_outside_then_inside'] and ea['longest_heat_inside_original_px']>=PARAMETERS['direction_reach_continuous_heat_original_px'] and ea['joint_inside_original_px']>=PARAMETERS['direction_reach_joint_original_px']):continue
            da=np.asarray(a['seed']['xy'][-1],float)-np.asarray(a['seed']['xy'][0],float);da/=max(np.linalg.norm(da),1e-9)
            for other in indices:
                if other==good:continue
                b=entries[other];eb=b['task_direction'];db=np.asarray(b['seed']['xy'][-1],float)-np.asarray(b['seed']['xy'][0],float);db/=max(np.linalg.norm(db),1e-9)
                if da@db<-.8 and eb['heat_inside_original_px']<PARAMETERS['opposite_direction_maximum_heat_inside_original_px']:
                    eb.update(selection_allowed=False,basis='opposite_A7_condition_has_observed_outside_to_SAM2_progression',preferred_entry=good,paired_a7_candidate_id=aid)
                    ea.update(basis='observed_outside_to_SAM2_progression_over_opposite_direction',paired_a7_candidate_id=aid)

def process(job,out):
    tick=time.time();adapter=dict(job,seeds=job['original_seeds']);priors=load_surface_priors(adapter)
    original=priors['original_half'];gray=cv2.cvtColor(original[525:],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    old=load(H1/'all_001/记录'/f'{job["id"]}.json');entries=[];heatmaps=[];tracksets=[]
    with gzip.open(H2/'all_001/记录'/f'{job["id"]}.json.gz','rt',encoding='utf-8') as f:previous=json.load(f)
    for k,(seed,rp) in enumerate(zip(job['seeds'],job['response_files'])):
        rp=Path(rp)
        with np.load(rp,allow_pickle=False) as z:heat=z['response'].copy()
        assert heat.shape==gray.shape and np.isfinite(heat).all() and 0<=heat.min()<=heat.max()<=1
        assert sha(rp)==previous['entries'][k]['response_sha256']
        assert np.array_equal(np.asarray(seed['xy']),np.asarray(previous['entries'][k]['seed']['xy']))
        # Reuse only the exact matching old conditional response and H1 trace.
        oi=seed.get('old_entry_index');h1paths=[]
        if oi is not None:
            assert np.allclose(seed['xy'],old['entries'][oi]['seed']['xy'])
            assert sha(rp)==old['entries'][oi]['response_sha256'];h1paths=old['entries'][oi]['candidates']
        tracks=build_tracks(priors,heat,gray);full_free=describe_free(priors,seed,heat)
        free,projection=prepare_task_free(full_free,tracks['tracks'])
        linked=connect(gray,heat,free,tracks['tracks'],h1paths,priors['surface_half'][525:])
        for candidate in linked['candidates']:candidate['free_axis_id']=free.get('free_axis_id','axis_00')
        linked['alternative_axes_used_for_connection']=False
        linked['retained_axis_count']=int(bool(free['points_xy']))+len(free.get('alternative_free_axes',[]))
        entries.append(dict(entry_index=k,seed=seed,response_file=str(rp),response_sha256=sha(rp),free=free,full_free_observation=full_free,free_projection=projection,connection=linked,track_diagnostics=tracks['diagnostics'],candidate_track_geometry={t['track_id']:t['points_xy'] for t in tracks['tracks'] if t['track_id'] in {c['track_id'] for c in linked['candidates']}}))
        heatmaps.append(heat);tracksets.append(tracks)
    task_direction_feasibility(entries,priors['sam2_surface_half'][525:])
    ranked=[]
    for k,e in enumerate(entries):
        for j,c in enumerate(e['connection']['candidates']):ranked.append((c['rank_score'],k,j,c))
    # Connections are eligible only if they retain the complete supported heat
    # ridge; do not force the endpoint onto a nearby gray lane.
    ranked.sort(key=lambda x:(entries[x[1]]['task_direction']['selection_allowed'],x[0],x[3]['metrics'].get('free',{}).get('quality',0)),reverse=True)
    best=ranked[0] if ranked else None;selected=best[1] if best else 0
    shown=[]
    for c in ranked:
        if len(shown)>=3:break
        # Entries are separate conditions, alternatives remain explicitly named.
        shown.append(c)
    multi=np.rint(original.astype(np.float32)*.65).astype(np.uint8)
    for track in tracksets[selected]['tracks']:
        tp=np.asarray(track['points_xy']);inferred=np.asarray(track['f3_pair_inferred'],bool)
        for j in range(1,len(tp)):
            weak=bool(inferred[j] or inferred[j-1])
            if not weak or j%5<3:poly(multi,tp[j-1:j+1],(110,110,110) if weak else (170,170,170),1)
    for n,(_,ei,ci,c) in reversed(list(enumerate(shown))):poly(multi,c['points_xy'],COLORS[n],3)
    for ei,e in enumerate(entries):
        p=(np.asarray(e['seed']['xy'])-[0,1050])*.5;poly(multi,p,(255,255,255),3)
        xy=tuple(np.rint(np.asarray(e['seed']['xy'][0])*.5).astype(int));cv2.putText(multi,'E'+str(ei+1),xy,cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    ambiguity=entries[selected]['connection'].get('ambiguous',True) or bool(job['seeds'][selected].get('direction_ambiguous',False))
    if len(ranked)>1 and ranked[0][1]!=ranked[1][1] and ranked[0][0]-ranked[1][0]<PARAMETERS['ambiguity_margin']:ambiguity=True
    final=draw_final(original,best[3],ambiguity) if best else original.copy()
    quality='候选接续待确认' if best else '未获得可评分接续候选'
    if best and ambiguity:quality='候选接续有歧义，保留最高评分方案'
    if best and best[3]['kind']=='free_fragment_no_resolved_connection':quality='保留有证据直段；未强制接入弱证据绳圈'
    cdetails='；'.join(f'{n+1}:E{ei+1} 分={s:.3f}'+(' 弱证据未采用' if c.get('continuation_eligible') is False else '')+(' 朝外仅诊断' if not entries[ei]['task_direction']['selection_allowed'] else '') for n,(s,ei,ci,c) in enumerate(shown)) or '没有受支持的自由段接续；不按固定间距补线'
    final_details='绿：原图+热图；黄：仅热图；浅蓝：A7+原图补段；青/紫：过渡/贴面；橙虚线：弱证据'
    panels=[label(original,'1 原图',[job['id'],'原图按50%显示；自动入口来自历史候选/A7，无人工路径输入']),
      label(heat_panel(original,heatmaps[selected]),f'2 方案B热力图 | 条件入口 E{selected+1}', ['响应强度固定0–1；灰区未计算；各入口热图分别推理','强响应不等于同根身份可信度']),
      label(multi,'3 F3来源中心轨迹与合法接续候选',[cdetails,'灰实线：原F3支持；灰虚线：源内推断；绿/黄/紫：合法几何候选']),
      label(final,'4 不跨轨、不跳轨的接续 | '+quality,[final_details,'贴面后锁定同一中心轨迹；自由投影相交需原图证据，不证明真实接触'])]
    gallery=out/'四栏检测结果'/(job['id']+'.jpg');save(gallery,np.concatenate(panels,axis=1))
    record=dict(version='IR-H6.0.0',id=job['id'],source_image=job['image_path'],source_image_sha256=job['image_sha256'],coordinate_system='crop_half_xy; original=2*xy+[0,1050]',entries=entries,selected_entry=selected,selected_candidate=best[2] if best else None,displayed_candidates=[dict(entry=k,candidate=j,score=s,task_direction_allowed=entries[k]['task_direction']['selection_allowed']) for s,k,j,c in shown],surface_tracks=tracksets[selected],input_dependency_sha256=priors['input_dependency_sha256'],response_entry_conditions_merged=False,complete_identity_verified=False,labels_read=False,device_connected=False,elapsed_seconds=time.time()-tick)
    with gzip.open(out/'记录'/(job['id']+'.json.gz'),'wt',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False)
    return dict(id=job['id'],entry_count=len(entries),selected_entry=selected,has_path=best is not None,ambiguous=ambiguity,score=best[0] if best else None,kind=best[3]['kind'] if best else None,length_original_px=float(2*np.linalg.norm(np.diff(best[3]['points_xy'],axis=0),axis=1).sum()) if best else 0,termination=best[3]['termination'] if best else None,track_count=len(tracksets[selected]['tracks']),candidate_pool=sum(e['connection'].get('candidate_pool_count',0) for e in entries),supported_fraction=float(np.mean(best[3]['supported'])) if best else 0,seconds=time.time()-tick,output=str(gallery),sha256=sha(gallery))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--subset',choices=['pilot','extra','all'],default='pilot');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8');cv2.setNumThreads(1);config=inputs();jobs=config['jobs']
    if a.subset=='pilot':jobs=[j for j in jobs if j['id'] in config['pilot_ids']]
    if a.subset=='extra':jobs=[j for j in jobs if j['id'] in config['extra_ids']]
    out=HERE/a.run;out.mkdir(exist_ok=False);(out/'记录').mkdir();(out/'source_snapshot').mkdir()
    sources={}
    for name in ['run_h6.py','common.py','connect_paths.py','heat_free.py','surface_tracks.py','topology.py']:
        p=HERE/name;sources[str(p)]=sha(p);shutil.copy2(p,out/'source_snapshot'/p.name)
    reused={str(p):sha(p) for p in [H2/'surface_tracks.py',H2/'free_candidates.py',H2/'connect_paths.py',H1/'priors.py']}
    dump(out/'run_start.json',dict(command=sys.argv,python=sys.executable,environment=platform.platform(),input_ids=[j['id'] for j in jobs],sources=sources,reused_sources=reused,input_protocol_sha256=sha(HERE/'inputs.json'),parameters=PARAMETERS,heat_parameters=HEAT_PARAMETERS,surface_parameters=SURFACE_PARAMETERS,topology_parameters=TOPOLOGY_PARAMETERS,planned=len(jobs),training=False,devices=False))
    rows=[];failures=[];t=time.time()
    with (out/'progress.jsonl').open('x',encoding='utf-8') as f,concurrent.futures.ThreadPoolExecutor(a.workers) as ex:
        pending={ex.submit(process,j,out):j['id'] for j in jobs}
        for future in concurrent.futures.as_completed(pending):
            try:r=future.result();rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            except Exception:
                r=dict(id=pending[future],error=traceback.format_exc());failures.append(r);dump(out/(r['id']+'__failure.json'),r)
            if (len(rows)+len(failures))%12==0 or len(rows)+len(failures)==len(jobs):print(json.dumps(dict(done=len(rows),failures=len(failures),total=len(jobs),seconds=round(time.time()-t,1))),flush=True)
    summary=dict(exit_code=int(bool(failures)),completed=len(rows),planned=len(jobs),failures=failures,rows=rows,elapsed_seconds=time.time()-t,source_unchanged=all(sha(p)==s for p,s in (sources|reused).items()),source_denominator=400,old_entry_unresolved_count=108,unresolved_not_absent_rope=True,identity_accuracy_not_measured=True)
    dump(out/'summary.json',summary);sys.exit(summary['exit_code'])
if __name__=='__main__':main()
