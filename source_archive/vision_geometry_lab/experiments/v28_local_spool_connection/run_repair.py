"""Reproduce V28 foundational repair on the fixed open development images."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from vision_geometry_lab.experiments.v28_local_spool_connection import data_eval, pipeline
from vision_geometry_lab.experiments.v28_local_spool_connection import bands_repair, regions_repair, connections_repair
from vision_geometry_lab.experiments.v28_local_spool_connection.run_experiment import detect_input
from vision_geometry_lab.experiments.v28_local_spool_connection.check_repair import evaluate_saved_pool

DEBUG_IDS=[data_eval.SAMPLE_IDS[0],data_eval.SAMPLE_IDS[2]]

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save_image(path,image):
    ok,encoded=cv2.imencode('.png',image)
    if not ok:raise ValueError('encode failed')
    encoded.tofile(path)

def draw_path(canvas,points,support,inferred,unavailable,color=None,offset=(0,0),scale=1.):
    p=(np.asarray(points)-offset)*scale
    for j in range(1,len(p)):
        both=bool(support[j-1] and support[j])
        c=color if color is not None else ((40,230,60) if both else (210,80,210) if unavailable[j] else (0,160,255))
        if both or j%3==0:
            cv2.line(canvas,tuple(np.rint(p[j-1]).astype(int)),tuple(np.rint(p[j]).astype(int)),c,2,cv2.LINE_AA)

def render_bands(image,bands):
    canvas=cv2.resize(image,(2048,1500))
    for b in bands['structures']:
        draw_path(canvas,b['points_xy'],b['supported'],b['inferred'],b['unavailable'],scale=.5)
        for side in ('left_xy','right_xy'):
            pts=b[side]
            for i in range(1,len(pts)):
                if pts[i-1] is not None and pts[i] is not None and b['supported'][i-1] and b['supported'][i]:
                    cv2.line(canvas,tuple(np.rint(np.array(pts[i-1])*.5).astype(int)),
                             tuple(np.rint(np.array(pts[i])*.5).astype(int)),(255,200,40),1,cv2.LINE_AA)
    cv2.putText(canvas,'Whole-envelope bands: green body, blue edges; dotted = hypothesis',(20,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    return canvas

def render_candidates(image,pool,regions,evaluation=None):
    mask=regions['spool_mask']|regions['free_corridor']|regions['transition_mask']
    ys,xs=np.nonzero(mask)
    if not len(xs):return cv2.resize(image,(2048,1500))
    x0=max(0,int(xs.min())-50);x1=min(image.shape[1],int(xs.max())+50)
    y0=max(0,int(ys.min())-60);y1=min(image.shape[0],int(ys.max())+30)
    crop=image[y0:y1,x0:x1];targetw=1100;scale=targetw/crop.shape[1]
    height=int(crop.shape[0]*scale);base=cv2.resize(crop,(targetw,height))
    panels=[]
    metrics={r['id']:r for r in evaluation['candidates']} if evaluation else {}
    for c in pool['candidates']:
        panel=base.copy()
        draw_path(panel,c['points_xy'],c['supported'],c['inferred'],c['unavailable'],offset=(x0,y0),scale=scale)
        for side in ('left_xy','right_xy'):
            for p,ok in zip(c[side],c['supported']):
                if p is not None and ok:cv2.circle(panel,tuple(np.rint((np.array(p)-(x0,y0))*scale).astype(int)),1,(255,255,40),-1)
        cv2.rectangle(panel,(0,0),(targetw,68),(0,0,0),-1)
        m=metrics.get(c['id']);text=c['id']+' '+c['kind']
        cv2.putText(panel,text,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.60,(255,255,255),1,cv2.LINE_AA)
        if m:
            text=f"bend 5px={m['bend5px_coverage']:.3f}; strict={m['supported_only_bend5px_coverage']:.3f}; background={m['explicit_background_fraction']:.3f}"
            cv2.putText(panel,text,(12,50),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
        panels.append(panel)
    if not panels:
        cv2.putText(base,'NO ENTRY / NO CANDIDATE',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,100,255),2)
        return base
    while len(panels)%2:panels.append(base.copy())
    return np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)])

def render_bend(image,pool,ev):
    # Visualization only after pool persistence and evaluation. The reference
    # never supplies the search crop or any graph node.
    ref=data_eval.load_evaluation_reference(ev['sample_id'])
    rankings={g:dict(best_id=None,selected_id=None,status='UNKNOWN',reason='no_ranking') for g in ('G0','G1','G2')}
    complete=data_eval.evaluate_reference(ref,pool,rankings,pool['input_mode'])
    segments=complete['trusted_bend_segments']
    if not segments:return None
    p=np.concatenate([np.asarray(s) for s in segments]);lo=np.maximum(0,p.min(axis=0)-100).astype(int)
    hi=np.minimum([image.shape[1],image.shape[0]],p.max(axis=0)+100).astype(int)
    base=image[lo[1]:hi[1],lo[0]:hi[0]].copy();scale=min(2.,1000/max(1,base.shape[1]))
    base=cv2.resize(base,None,fx=scale,fy=scale)
    best=max(ev['candidates'],key=lambda r:r['bend5px_coverage']) if ev['candidates'] else None
    for s in segments:
        pts=np.rint((np.asarray(s)-lo)*scale).astype(int)
        cv2.polylines(base,[pts],False,(255,255,255),2,cv2.LINE_AA)
    if best:
        c=next(c for c in pool['candidates'] if c['id']==best['id'])
        draw_path(base,c['points_xy'],c['supported'],c['inferred'],c['unavailable'],offset=lo,scale=scale)
    return base

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--samples',choices=('debug','all'),default='debug')
    ap.add_argument('--mode',choices=('AUTO','CONDITIONAL'),default='AUTO')
    ap.add_argument('--reuse-debug',type=Path)
    ap.add_argument('--reuse-band-folder',default='bands_v2')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(2)
    items=data_eval.list_inputs()
    if args.samples=='debug':items=[i for i in items if i['sample_id'] in DEBUG_IDS]
    write(out/'input_manifest.json',items)
    write(out/'config.json',dict(bands=bands_repair.DEFAULT_CONFIG,connection=connections_repair.DEFAULT_CONFIG,
        legacy_evaluation=data_eval.EVALUATION_PROTOCOL,scoring='disabled_until_debug_bend_coverage_and_background_gate'))
    hashes={}
    for name in ('regions_repair.py','bands_repair.py','connections_repair.py','run_repair.py','check_repair.py'):
        src=Path(__file__).parent/name;dest=out/'source_snapshot'/name;dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(src.read_bytes());hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    write(out/'source_hashes.json',hashes)
    folders=[]
    for item in items:
        start=time.perf_counter();sid=item['sample_id'];folder=out/args.mode/sid;folder.mkdir(parents=True)
        image=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR)
        if args.reuse_debug:
            old=args.reuse_debug
            entry=read(old/'regions'/sid/'fresh_auto_entry.json')
            data=np.load(old/'regions'/sid/'regions.npz');regions={k:data[k] for k in data.files}
            regions['summary']=read(old/'regions'/sid/'regions.json')
            band_folder=args.reuse_band_folder if (old/args.reuse_band_folder/sid/'bands.json').exists() else 'bands'
            bands=read(old/band_folder/sid/'bands.json')
            sources=dict(entry=str(old/'regions'/sid/'fresh_auto_entry.json'),regions=str(old/'regions'/sid/'regions.npz'),
                bands=str(old/band_folder/sid/'bands.json'),bands_sha256=data_eval.sha256(old/band_folder/sid/'bands.json'),
                extraction_config=bands['config'],extraction_iteration=read(old/band_folder/sid/'iteration.json') if (old/band_folder/sid/'iteration.json').exists() else None)
            write(folder/'reused_input_provenance.json',sources)
        else:
            entry=detect_input(image,sid,args.mode)
            regions=regions_repair.detect_regions(image,entry['free_seed'],entry['width_px'],camera=item['source']['camera'],arc=entry['arc'])
            if entry['available']:
                direction=pipeline.unit(np.asarray(entry['free_seed'][-1])-entry['free_seed'][0])
                bands=bands_repair.extract_local_bands(image,entry['free_seed'][-1],direction,entry['width_px'],regions=regions)
            else:bands=dict(structures=[],sections=[],diagnostics=dict(reason='ENTRY_UNAVAILABLE'))
        write(folder/'entry.json',entry);write(folder/'regions.json',regions['summary'])
        np.savez_compressed(folder/'regions.npz',**{k:v for k,v in regions.items() if isinstance(v,np.ndarray)})
        write(folder/'bands.json',bands)
        pool=connections_repair.generate_pool(image,entry['free_seed'],entry['width_px'],bands,regions) if entry['available'] else dict(
            candidates=[],image_shape_hw=list(image.shape[:2]),origin_xy=[0.,0.],direction_xy=[1.,0.],normal_xy=[0.,1.],width_px=1.,
            free_seed=None,input_uses_labels=False,geometry_completion_is_evidence=False)
        pool['input_mode']=args.mode;pool['automatic_entry_success']=entry['available']
        write(folder/'pool.json',pool)
        save_image(folder/'original.png',cv2.resize(image,(2048,1500)))
        save_image(folder/'regions.png',regions_repair.render_regions(image,regions))
        save_image(folder/'bands.png',render_bands(image,bands))
        save_image(folder/'candidates_before_evaluation.png',render_candidates(image,pool,regions))
        folders.append((item,folder));print(json.dumps(dict(sample_id=sid,entry=entry['available'],bands=len(bands['structures']),
            candidates=len(pool['candidates']),forward=pool.get('diagnostics',{}).get('connection_sections_reached'),seconds=round(time.perf_counter()-start,1))),flush=True)
    # All image-only predictions are saved before opening evaluation references.
    rows=[]
    for item,folder in folders:
        data=np.load(folder/'regions.npz');regions={k:data[k] for k in data.files}
        ev=evaluate_saved_pool(item['sample_id'],folder/'pool.json',regions)
        write(folder/'evaluation.json',ev);rows.append(ev)
        image=cv2.imdecode(np.fromfile(item['image_path'],np.uint8),cv2.IMREAD_COLOR);pool=read(folder/'pool.json')
        save_image(folder/'candidate_diagnostics.png',render_candidates(image,pool,regions,ev))
        bend=render_bend(image,pool,ev)
        if bend is not None:save_image(folder/'best_bend_evaluation_only.png',bend)
        print(json.dumps(dict(sample_id=item['sample_id'],best5px=ev['best_bend5px_coverage'],strict5px=ev['best_supported_only_bend5px_coverage'],
            max_forward=ev['forward_reach_px'],background=max([r['explicit_background_fraction'] for r in ev['candidates']] or [0.]))),flush=True)
    write(out/'results.json',rows)
    write(out/'complete.json',dict(predictions_saved_before_labels=True,source_hashes=hashes,mode=args.mode,samples=args.samples,
        scoring_resumed=False,physical_identity_confirmed=False))

if __name__=='__main__':main()
