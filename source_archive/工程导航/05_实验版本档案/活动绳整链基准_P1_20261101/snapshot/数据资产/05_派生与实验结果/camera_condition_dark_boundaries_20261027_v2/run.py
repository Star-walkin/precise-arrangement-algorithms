"""D2暗纹边界实验：固定ROI/增强图，保持所有旧基准源码不变。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import argparse,importlib.util,json,shutil,sys,time,traceback
from dataclasses import replace
import cv2
import numpy as np
from data_io import HERE,D1,C3,selection,inputs,sources,sha,read_json,dump,write_image
from dark_observation import propose_dark_points
from dark_curves import group_and_fit

def small(value):
    if isinstance(value,dict):return {k:small(v) for k,v in value.items() if not(isinstance(v,np.ndarray) and v.size>200000)}
    if isinstance(value,list):return [small(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value

def threshold_control(original,enhanced,roi,pitch):
    """简单对照：仅降低原B1的三个强度下限，归类/曲线结构保持旧样。"""
    spec=importlib.util.spec_from_file_location('d2_control',D1/'dark_c3.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    b1=mod._pure_module('b1_control',mod._B1_SOURCE);c3=mod._pure_module('c3_control',mod._C3_SOURCE)
    params=replace(b1.FullROIParameters(),min_weak_contrast=.4,min_strong_contrast=1.1,min_response=1.3)
    exclusion=np.zeros(roi.shape,np.uint8)
    _,lines,diag=b1.detect_full_roi(enhanced,roi,exclusion,pitch,params)
    lines,recheck=mod._recheck_original(original,roi,lines,pitch)
    observed=np.zeros(roi.shape,np.uint8)
    for line in lines:
        for x,y in line['observed_points_xy']:cv2.circle(observed,(int(x),int(y)),1,255,-1)
    observed[roi==0]=0
    fit=c3.fit_conic_family(lines,pitch,roi,exclusion)
    return {'observed_lines':lines,'observed_mask':observed,'fit':fit,'inferred_mask':np.zeros_like(roi),
            'diagnostics':{'control':'threshold_floor_only_half','b1':diag,'original_recheck':recheck}}

def draw(enhanced_roi,roi,result):
    canvas=enhanced_roi.copy()
    for curve in result.get('fit',{}).get('curves',[]):
        pts=np.asarray(curve.get('sampled_points_xy',[]),float)
        states=curve.get('point_states',['supported_fit']*len(pts))
        uncertain=curve.get('fit_to_observed_residual',{}).get('quality')=='unreliable'
        for i in range(len(pts)-1):
            if np.linalg.norm(pts[i+1]-pts[i])>35:continue
            # ROI存在活动绳缺口；两个合法端点不能授权在中间补直线。
            q=np.rint(np.linspace(pts[i],pts[i+1],max(2,int(np.linalg.norm(pts[i+1]-pts[i]))+1))).astype(int)
            if np.any(roi[q[:,1],q[:,0]]==0):continue
            inferred='infer' in str(states[i]) or 'gap' in str(states[i])
            if inferred and (i//3)%2:continue
            if uncertain and not inferred and (i//6)%2:continue
            color=(215,70,215) if inferred else ((40,115,255) if uncertain else (230,190,35))
            cv2.line(canvas,tuple(np.rint(pts[i]).astype(int)),tuple(np.rint(pts[i+1]).astype(int)),color,2,cv2.LINE_AA)
    for line in result.get('observed_lines',[]):
        for p in line.get('observed_points_xy',[])[::2]:
            cv2.drawMarker(canvas,tuple(np.rint(p[:2]).astype(int)),(0,230,255),cv2.MARKER_CROSS,5,1,cv2.LINE_AA)
    cv2.drawContours(canvas,cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,135,0),1)
    canvas[roi==0]=0
    return canvas

def process(task):
    row,out,variant,pilot=task;out=Path(out);cv2.setNumThreads(1);start=time.perf_counter()
    rel=Path(row['output_file']);role=rel.parent.name;stem=rel.stem
    record={k:row[k] for k in ['output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status']}
    record.update({'version':read_json(HERE/'protocol.json')['version'],'variant':variant,'source_png_sha256':row['output_sha256'],'pilot_selected':row['output_file'] in pilot})
    try:
        original,roi,enhanced,enhanced_roi,pitch,oldcase,deps=inputs(row)
        if variant=='threshold_only':
            result=threshold_control(original,enhanced,roi,pitch);proposal_diag={}
        elif variant=='anchored':
            from anchored_dark import detect_anchored
            c3_record=read_json(C3/'boundaries/records'/role/(stem+'.json'))
            result=detect_anchored(original,enhanced,roi,pitch,c3_record)
            proposal_diag=result.get('observation_diagnostics',{})
        else:
            proposals=propose_dark_points(original,enhanced,roi,pitch)
            result=group_and_fit(proposals,roi,pitch)
            proposal_diag=proposals.get('diagnostics',{})
        masks={'observed':result['observed_mask'],'smooth':result['fit']['smooth_mask'],
               'inferred':result.get('inferred_mask',np.zeros_like(roi))}
        for key,m in masks.items():
            if m.shape!=roi.shape or m.dtype!=np.uint8 or np.any((m!=0)&(m!=255)) or np.any(m[roi==0]):raise ValueError('Mask contract: '+key)
            write_image(out/'记录/masks'/key/role/rel.name,m)
        overlay=draw(enhanced_roi,roi,result);write_image(out/'图像结果'/role/rel.name,overlay)
        points=sum(len(line.get('observed_points_xy',[])) for line in result['observed_lines'])
        record.update({'status':'completed','pitch_px':pitch,'dark_roi_pixels':int(np.count_nonzero(roi)),
            'observed_point_count':points,'observed_pixels':int(np.count_nonzero(masks['observed'])),
            'curve_count':len(result['fit'].get('curves',[])),'smooth_pixels':int(np.count_nonzero(masks['smooth'])),
            'uncertain_fit_curve_count':sum(c.get('fit_to_observed_residual',{}).get('quality')=='unreliable' for c in result['fit'].get('curves',[])),
            'inferred_pixels':int(np.count_nonzero(masks['inferred'])),'input_dependency_sha256':deps,
            'old_d1_observed_point_count':oldcase['record']['c3_diagnostics']['original_rechecked_points'],
            'old_d1_curve_count':oldcase['record']['curve_count'],
            'observation_diagnostics':proposal_diag,'grouping_diagnostics':result.get('diagnostics',{}),
            'fit_diagnostics':result['fit'].get('diagnostics',{})})
        dump(out/'记录/cases'/role/(stem+'.json'),{'record':record,'observed_lines':small(result['observed_lines']),'fit':small(result['fit'])})
    except Exception as e:
        record.update({'status':'failed','error':repr(e),'traceback':traceback.format_exc()});dump(out/'记录/cases'/role/(stem+'.json'),{'record':record})
    return {**record,'elapsed_s':round(time.perf_counter()-start,3)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','all'],required=True);ap.add_argument('--variant',choices=['d2','threshold_only','anchored'],default='anchored');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    out=args.output.absolute()
    if out.parent!=HERE or out.exists():raise ValueError('Use new direct child run')
    rows,pilot=selection(args.mode);source=sources();protocol=read_json(HERE/'protocol.json')
    freeze_path=HERE/protocol.get('freeze_manifest','freeze_all.json')
    if any(sha(p)!=h for p,h in protocol['upstream_source_sha256'].items()):raise ValueError('Old baseline source changed')
    if args.mode=='all':
        frozen=read_json(freeze_path)
        if source!=frozen['source_sha256'] or args.variant!=frozen['variant']:raise ValueError('Final source freeze mismatch')
    out.mkdir();snapshots={}
    for i,p in enumerate(source):
        target=out/'source_snapshot'/(str(i).zfill(2)+'_'+Path(p).name);target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);snapshots[p]=target.relative_to(out).as_posix()
    dump(out/'run_context.json',{'source_sha256':source,'source_snapshot_paths':snapshots,'selection':[r['output_file'] for r in rows],
        'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'python':sys.version,'python_executable':sys.executable,
        'opencv':cv2.__version__,'numpy':np.__version__,'workers':args.workers,'mode':args.mode,'variant':args.variant,'version':protocol['version'],'freeze_manifest_path':str(freeze_path)})
    records=[]
    with (out/'run_log.jsonl').open('w',encoding='utf-8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i,r in enumerate(ex.map(process,[(r,str(out),args.variant,pilot) for r in rows]),1):
                records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                print(json.dumps({'i':i,'n':len(rows),'image':r['output_file'],'status':r['status'],'points':r.get('observed_point_count'),'curves':r.get('curve_count'),'error':r.get('error')},ensure_ascii=False),flush=True)
    failed=sum(r['status']!='completed' for r in records);same=source==sources()
    summary={'version':protocol['version'],'variant':args.variant,'planned':len(rows),'completed':len(rows)-failed,'failed':failed,'source_unchanged':same,
        'exit_code':int(bool(failed) or not same),'by_camera':{c:sum(r['camera_role']==c for r in records) for c in ['left','right']},
        'images_with_observations':sum(r.get('observed_point_count',0)>0 for r in records),'images_with_curves':sum(r.get('curve_count',0)>0 for r in records),
        'total_observed_points':sum(r.get('observed_point_count',0) for r in records),'total_curves':sum(r.get('curve_count',0) for r in records),
        'uncertain_fit_curves':sum(r.get('uncertain_fit_curve_count',0) for r in records),
        'old_d1_points':sum(r.get('old_d1_observed_point_count',0) for r in records),
        'finished_utc':datetime.now(timezone.utc).isoformat(),'accuracy':None,'html_generated':False,'equipment_access':False,'reserved_access':False}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    raise SystemExit(summary['exit_code'])

if __name__=='__main__':main()
