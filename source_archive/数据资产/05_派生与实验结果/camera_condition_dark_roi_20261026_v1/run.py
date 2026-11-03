"""按用户的三个步骤各输出一套图片；不生成HTML。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import argparse,json,sys,shutil,time,traceback
import cv2
import numpy as np
from data_io import HERE,selection,inputs,sources,dump,read_json,write_image
from dark_roi import process_dark_roi
from dark_c3 import detect_dark_c3

FOLDERS=['01_欠曝光ROI','02_对比度增强ROI','03_C3边界结果']

def small(value):
    if isinstance(value,dict):return {k:small(v) for k,v in value.items() if not(isinstance(v,np.ndarray) and v.size>200000)}
    if isinstance(value,list):return [small(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value

def draw(enhanced,roi,result):
    canvas=enhanced.copy()
    # 紫色为整组模型拟合，黄色十字为实际检测采样；都只显示在暗ROI内。
    for curve in result.get('fit',{}).get('curves',[]):
        pts=np.asarray(curve.get('sampled_points_xy',[]),float)
        if len(pts)<2:continue
        for i in range(len(pts)-1):
            if np.linalg.norm(pts[i+1]-pts[i])<=35:
                cv2.line(canvas,tuple(np.rint(pts[i]).astype(int)),tuple(np.rint(pts[i+1]).astype(int)),(220,75,220),2,cv2.LINE_AA)
    for line in result.get('observed_lines',[]):
        for p in line.get('observed_points_xy',[])[::3]:
            cv2.drawMarker(canvas,tuple(np.rint(p[:2]).astype(int)),(0,230,255),cv2.MARKER_CROSS,7,1,cv2.LINE_AA)
    contours=cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(canvas,contours,-1,(0,175,0),1)
    canvas[roi==0]=0
    return canvas

def process(task):
    row,out,pilot=task;out=Path(out);cv2.setNumThreads(1);start=time.perf_counter()
    rel=Path(row['output_file']);role=rel.parent.name;name=rel.name
    record={k:row[k] for k in ['output_file','camera_role','source_group','source_file','source_time_s','dataset_use_status']}
    record.update({'source_png_sha256':row['output_sha256'],'pilot_selected':row['output_file'] in pilot})
    try:
        image,surface,pitch,deps=inputs(row)
        dark,raw_roi,enh_roi,enh_full,diag=process_dark_roi(image,surface,pitch)
        result=detect_dark_c3(image,enh_full,dark,pitch)
        observed=result['observed_mask'];smooth=result['fit']['smooth_mask']
        for a in [dark,observed,smooth]:
            if a.shape!=surface.shape or a.dtype!=np.uint8 or np.any((a!=0)&(a!=255)):raise ValueError('Invalid binary mask')
        if np.any(dark[surface==0]) or np.any(observed[dark==0]) or np.any(smooth[dark==0]):raise ValueError('Mask outside allowed ROI')
        if not np.array_equal(raw_roi[dark>0],image[dark>0]) or np.any(raw_roi[dark==0]) or np.any(enh_roi[dark==0]):raise ValueError('ROI pixel contract')
        overlay=draw(enh_roi,dark,result)
        for folder,a in zip(FOLDERS,[raw_roi,enh_roi,overlay]):write_image(out/'图像结果'/folder/role/name,a)
        for folder,a in [('dark_roi',dark),('observed',observed),('smooth',smooth)]:write_image(out/'记录/masks'/folder/role/name,a)
        record.update({'status':'completed','pitch_px':pitch,'surface_pixels':int(np.count_nonzero(surface)),
            'dark_roi_pixels':int(np.count_nonzero(dark)),'observed_pixels':int(np.count_nonzero(observed)),
            'curve_count':len(result['fit'].get('curves',[])),'smooth_pixels':int(np.count_nonzero(smooth)),
            'roi_and_enhancement':diag,'c3_diagnostics':result.get('diagnostics',{}),'input_dependency_sha256':deps})
        dump(out/'记录/cases'/role/(rel.stem+'.json'),{'record':record,'observed_lines':small(result['observed_lines']),'fit':small(result['fit'])})
    except Exception as e:
        record.update({'status':'failed','error':repr(e),'traceback':traceback.format_exc()})
        dump(out/'记录/cases'/role/(rel.stem+'.json'),{'record':record})
    return {**record,'elapsed_s':round(time.perf_counter()-start,3)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','all'],required=True);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    out=args.output.absolute()
    if out.parent!=HERE or out.exists():raise ValueError('Use new direct child directory')
    rows,pilot=selection(args.mode);source=sources()
    if args.mode=='all' and source!=read_json(HERE/'freeze_all.json')['source_sha256']:raise ValueError('Frozen source changed')
    out.mkdir();snapshots={}
    for i,p in enumerate(source):
        target=out/'source_snapshot'/(str(i).zfill(2)+'_'+Path(p).name);target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target);snapshots[p]=target.relative_to(out).as_posix()
    dump(out/'run_context.json',{'source_sha256':source,'source_snapshot_paths':snapshots,'selection':[r['output_file'] for r in rows],
        'started_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'python':sys.version,'python_executable':sys.executable,
        'opencv':cv2.__version__,'numpy':np.__version__,'workers':args.workers,'mode':args.mode,'version':'DR-C3.1.0'})
    records=[]
    with (out/'run_log.jsonl').open('w',encoding='utf-8') as log:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i,r in enumerate(ex.map(process,[(r,str(out),pilot) for r in rows]),1):
                records.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
                print(json.dumps({'i':i,'n':len(rows),'image':r['output_file'],'status':r['status'],'dark':r.get('dark_roi_pixels'),'points':r.get('observed_pixels'),'curves':r.get('curve_count'),'error':r.get('error')},ensure_ascii=False),flush=True)
    failed=sum(r['status']!='completed' for r in records);same=source==sources()
    summary={'version':'DR-C3.1.0','planned':len(rows),'completed':len(rows)-failed,'failed':failed,'source_unchanged':same,
        'exit_code':int(bool(failed) or not same),'images_per_output_group':len(rows)-failed,'output_groups':FOLDERS,
        'by_camera':{c:sum(r['camera_role']==c for r in records) for c in ['left','right']},
        'nonempty_dark_roi':sum(r.get('dark_roi_pixels',0)>0 for r in records),
        'images_with_observations':sum(r.get('observed_pixels',0)>0 for r in records),
        'images_with_curves':sum(r.get('curve_count',0)>0 for r in records),
        'total_observed_pixels':sum(r.get('observed_pixels',0) for r in records),'total_curves':sum(r.get('curve_count',0) for r in records),
        'finished_utc':datetime.now(timezone.utc).isoformat(),'accuracy':None,'html_generated':False,'equipment_access':False,'reserved_access':False}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    raise SystemExit(summary['exit_code'])

if __name__=='__main__':main()
