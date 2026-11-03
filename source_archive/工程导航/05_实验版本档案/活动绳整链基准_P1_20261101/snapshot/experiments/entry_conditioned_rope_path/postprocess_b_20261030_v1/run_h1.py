"""Offline stage images and six-panel strips. No HTML, training, or hardware."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,sys,time,traceback,platform,shutil
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import h1_detector as detector
from priors import load_priors,geometry_for_seed
from f3_tail import propose_f3_tails

HERE=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,im):
    p.parent.mkdir(parents=True,exist_ok=True)
    ext=p.suffix
    cv2.imencode(ext,im,[cv2.IMWRITE_JPEG_QUALITY,94] if ext=='.jpg' else [cv2.IMWRITE_PNG_COMPRESSION,3])[1].tofile(str(p))
def panel_label(im,title,detail):
    # Chinese headers are separate from the photo, and every displayed panel
    # has the same half-size coordinate system and scale.
    h,w=im.shape[:2];board=Image.new('RGB',(w,h+100),(245,247,250))
    board.paste(Image.fromarray(im[:,:,::-1]),(0,100));d=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',31);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    d.text((18,8),title,font=font,fill=(20,30,40));d.text((18,53),detail,font=small,fill=(60,70,80))
    return np.asarray(board)[:,:,::-1].copy()
def points_half(path):return np.array(path['points_xy'])*.5
def draw_review_tails(image,tails):
    out=image.copy()
    for tail in tails:
        p=np.asarray(tail['points_xy'],float)+[0,525]
        for j in range(1,len(p)):
            if (j//3)%2==0:
                cv2.line(out,tuple(np.rint(p[j-1]).astype(int)),tuple(np.rint(p[j]).astype(int)),(0,165,255),2,cv2.LINE_AA)
    return out
def draw_paths(original,paths,seed,final=False):
    out=original.copy();mask=np.zeros((original.shape[0]*2,original.shape[1]*2),np.uint8)
    certain=np.zeros_like(mask)
    if not final:
        for k,path in reversed(list(enumerate(paths))):
            p=np.rint(points_half(path)).astype(np.int32)
            cv2.polylines(out,[p],False,[(30,230,30),(255,210,40),(220,60,220)][k%3],2,cv2.LINE_AA)
    elif paths:
        path=paths[0];p=points_half(path);full=np.array(path['points_xy']);width=path['width_px']
        color=np.zeros_like(out);area=np.zeros(out.shape[:2],np.uint8)
        for j in range(1,len(p)):
            supported=path['supported'][j] and path['supported'][j-1]
            c=(30,220,30) if supported else (0,150,255)
            thick=max(2,int(round(width[j]*.5)))
            a,b=np.rint(p[[j-1,j]]).astype(int)
            cv2.line(color,a,b,c,thick,cv2.LINE_AA);cv2.line(area,a,b,255,thick)
            aa,bb=np.rint(full[[j-1,j]]).astype(int)
            cv2.line(mask,aa,bb,255,max(1,int(round(width[j]))))
            if supported:cv2.line(certain,aa,bb,255,max(1,int(round(width[j]))))
        on=area>0;out[on]=np.rint(out[on]*.65+color[on]*.35).astype(np.uint8)
        for j in range(1,len(p)):
            c=(30,235,30) if path['supported'][j] else (0,155,255)
            cv2.line(out,tuple(np.rint(p[j-1]).astype(int)),tuple(np.rint(p[j]).astype(int)),c,1,cv2.LINE_AA)
    cv2.polylines(out,[np.rint(np.array(seed)*.5).astype(np.int32)],False,(255,255,255),2,cv2.LINE_AA)
    return out,mask,certain

def process(job,out):
    started=time.time();priors=load_priors(job)
    original=priors['original_half'];gray=cv2.cvtColor(original[525:],cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    inference=load(HERE/'responses'/job['id']/'inference.json')
    entry_results=[];strips=[];files=[]
    for k,seed in enumerate(job['seeds']):
        rp=HERE/'responses'/job['id']/f'entry_{k:02d}.npz'
        with np.load(rp,allow_pickle=False) as data:response=data['response'].copy()
        geometry=geometry_for_seed(priors,seed)
        paths,baseline,field=detector.detect(gray,response,seed['xy'],geometry['width_px'],
          priors['f3_merged_half'][525:],priors['f3_inferred_half'][525:],priors['surface_half'][525:])
        # A different accepted A7 axis makes this seed an explicit competing
        # entry, e.g. a tool. Preserve its search for review but never color it
        # as supported rope merely because long background edges are strong.
        seed_conflict=bool(geometry.get('a7_axis_hint') is None and
            any(c.get('accepted') for c in priors['a7_candidates']))
        if seed_conflict:
            for path in paths:
                path['supported']=[False]*len(path['supported'])
                for evidence in path['evidence']:
                    if evidence is not None:
                        evidence['local_measurement_supported']=evidence['supported']
                        evidence['supported']=False
                        evidence['entry_conflict']=True
        tails=dict(candidates=[],diagnostics={'reason':'no_path_or_conflicting_entry'})
        if paths and not seed_conflict:
            local=(np.asarray(paths[0]['points_xy'])-[0,1050])*.5
            if len(local)>3:
                tangent=local[-1]-local[max(0,len(local)-6)]
                fields=dict(gray=gray,heat=response,merged=priors['f3_merged_half'][525:],
                   inferred=priors['f3_inferred_half'][525:],surface=priors['surface_half'][525:])
                tails=propose_f3_tails(fields,local[-16:],tangent,float(np.median(paths[0]['width_px'][-8:]))*.5)
        heat=np.full_like(original,38);heat[525:]=cv2.applyColorMap(np.rint(response*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        local=np.full_like(original,38);local[525:]=cv2.applyColorMap(np.rint(field*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        hints=original.copy();hints[priors['f3_merged_half']>0]=(255,220,20)
        cv2.polylines(hints,[np.rint(np.asarray(seed['xy'])*.5).astype(np.int32)],False,(255,255,255),3)
        if geometry.get('a7_axis_hint'):
            axis=geometry['a7_axis_hint']
            pts=axis.get('anchor_segment_xy')
            if pts is not None:cv2.polylines(hints,[np.rint(np.asarray(pts)*.5).astype(np.int32)],False,(30,240,30),3)
        multi,_,_=draw_paths(original,paths,seed['xy'])
        if baseline:
            cv2.polylines(multi,[np.rint(points_half(baseline[0])).astype(np.int32)],False,(170,170,170),1)
        final,mask,certain=draw_paths(original,paths,seed['xy'],True)
        multi=draw_review_tails(multi,tails['candidates'])
        final=draw_review_tails(final,tails['candidates'])
        entry_name=f'{job["id"]}__entry{k+1:02d}'
        stages=[('01_原图',original,'原图','原图缩小至50%展示；源文件不改'),
          ('02_B数值热图',heat,'方案B原始响应','固定0–1色标；灰区未计算；强度不是概率'),
          ('03_局部候选',local,'局部对比诊断','显示去低频亮暗结构；搜索另测方向相关的法向双边'),
          ('04_几何提示',hints,'自动入口与F3提示','白线：原始自动入口；青线：F3；不指定同根'),
          ('05_路径候选',multi,'方向连续的多路径搜索','绿/青/紫：候选；灰：热图评分基线，同几何和停止约束'),
          ('06_最终叠加',final,'活动绳位置候选'+('（入口与A7冲突）' if seed_conflict else ''),
             '绿：局部匹配；橙：待确认；橙虚线：F3竞争尾段，不入主掩膜')]
        panels=[]
        for dirname,im,title,detail in stages:
            p=out/'分步图像'/dirname/(entry_name+'.jpg');save(p,im);files.append(str(p.relative_to(out)))
            panels.append(panel_label(im,f'{title} | 入口{k+1}/{len(job["seeds"])}',detail))
        strips.append(np.concatenate(panels,axis=1))
        for name,arr in [('候选绳带',mask),('局部支持绳带',certain)]:
            p=out/'掩膜'/name/(entry_name+'.png');save(p,arr);files.append(str(p.relative_to(out)))
        entry_results.append(dict(entry_index=k,seed=seed,geometry=geometry,response_sha256=sha(rp),
          candidates=paths,heat_only_baseline=baseline,selected_candidate_index=0 if paths else None,
          f3_tail_review=tails,f3_tail_coordinate_system='half_size_model_crop_xy; original_xy=2*xy+[0,1050]',
          automatic_entry_conflict=seed_conflict,full_identity_confirmed=False,estimated_width_not_segmentation_truth=True))
    strip=out/'长条对照图'/f'{job["id"]}.jpg';save(strip,np.concatenate(strips,axis=0));files.append(str(strip.relative_to(out)))
    record=dict(version='IR-H1.0.0',id=job['id'],source_image=job['image_path'],
      source_image_sha256=job['image_sha256'],original_shape=[3000,4096],
      display_scale=.5,entry_count=len(entry_results),entries=entry_results,
      input_dependency_sha256=priors['input_dependency_sha256'],
      files=files,parameters=detector.PARAMETERS,elapsed_seconds=time.time()-started,
      labels_read=False,device_connected=False,complete_rope_identity_verified=False)
    rp=out/'记录'/f'{job["id"]}.json';dump(rp,record)
    return dict(id=job['id'],entries=len(entry_results),candidate_counts=[len(x['candidates']) for x in entry_results],
      top_lengths_px=[x['candidates'][0]['length_px'] if x['candidates'] else 0 for x in entry_results],
      strip=str(strip.relative_to(out)),seconds=record['elapsed_seconds'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--subset',choices=['pilot','all','extra'],default='pilot');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    cv2.setNumThreads(1);out=HERE/args.run;out.mkdir(exist_ok=False);(out/'记录').mkdir()
    (out/'source_snapshot').mkdir()
    sources={}
    for p in [HERE/'h1_detector.py',HERE/'straight_support.py',HERE/'f3_tail.py',HERE/'priors.py',Path(__file__)]:
        sources[str(p)]=sha(p);shutil.copy2(p,out/'source_snapshot'/p.name)
    meta=load(HERE/'protocol_inputs.json');jobs=load(HERE.parent/'batch_ab_20261030_001/jobs.json')['jobs']
    from prepare_inference import PILOT_IDS
    if args.subset=='pilot':jobs=[j for j in jobs if j['id'] in PILOT_IDS]
    elif args.subset=='extra':
        other=[j for j in jobs if j['id'] not in PILOT_IDS]
        jobs=sorted(other,key=lambda j:hashlib.sha256(('H1extra:'+j['id']).encode()).hexdigest())[:8]
    dump(out/'run_start.json',dict(command=sys.argv,python=sys.executable,environment=platform.platform(),
      sources=sources,parameters=detector.PARAMETERS,input_ids=[j['id'] for j in jobs],
      protocol_sha256=sha(HERE/'protocol_inputs.json'),planned=len(jobs),training=False,labels_read=False))
    started=time.time();rows=[];failures=[]
    with (out/'progress.jsonl').open('x',encoding='utf-8') as log,concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        pending={ex.submit(process,j,out):j['id'] for j in jobs}
        for f in concurrent.futures.as_completed(pending):
            try:row=f.result();rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            except Exception:
                error=dict(id=pending[f],error=traceback.format_exc());failures.append(error);dump(out/(pending[f]+'__failure.json'),error)
            if (len(rows)+len(failures))%10==0 or len(rows)+len(failures)==len(jobs):print(json.dumps(dict(done=len(rows),failures=len(failures),total=len(jobs),seconds=round(time.time()-started,1))),flush=True)
    receipt=dict(exit_code=1 if failures else 0,completed=len(rows),planned=len(jobs),failures=failures,
      entries=sum(r['entries'] for r in rows),rows=rows,seconds=time.time()-started,
      source_unchanged=all(sha(p)==v for p,v in sources.items()))
    dump(out/'summary.json',receipt);sys.exit(receipt['exit_code'])
if __name__=='__main__':main()
