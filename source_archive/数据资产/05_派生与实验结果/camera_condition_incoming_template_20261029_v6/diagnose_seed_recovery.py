"""Fixed open-image recovery diagnostic; prediction saved before viewing."""
from pathlib import Path
import importlib.util,sys,time,shutil,json
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;A5=HERE.parent/'camera_condition_incoming_photometric_20261029_v5'
sys.path.insert(0,str(A5))
import whole_io as io
from free_rope import refine_and_extend
sys.path.insert(0,str(HERE))
from free_seed_recovery import recover_free_seeds

def compact(x):
    if isinstance(x,np.ndarray) and x.ndim>=2:return {'array_shape':list(x.shape),'dtype':str(x.dtype)}
    if isinstance(x,dict):return {k:compact(v) for k,v in x.items()}
    if isinstance(x,list):return [compact(v) for v in x]
    return x

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',default='seed_probe_001');p.add_argument('--mode',choices=['single','fixed4'],default='single');args=p.parse_args()
    names=['dual_20260923_101643__pair0375__right']
    if args.mode=='fixed4':names+=['L_run_20260923_01__f000015__left','ruler_left_05','ruler_right_05']
    out=HERE/args.run;out.mkdir(exist_ok=False);shutil.copy2(HERE/'free_seed_recovery.py',out/'source_snapshot.py');cv2.setNumThreads(1);rows=io.selection('all');results=[]
    for name in names:
        row=next(r for r in rows if Path(r['output_file']).stem==name);rel=Path(row['output_file']);original=io.DATA/rel;surface=io.C4/'roi/surface_masks'/rel.parent.name/rel.name;c3=io.C3/'boundaries/records'/rel.parent.name/(name+'.json')
        assert io.sha(original)==row['output_sha256'];io.checked(surface);io.checked(c3)
        data={'original':io.read_image(original),'surface':io.read_image(surface,0),'pitch':float(io.read_json(c3)['record']['pitch_px'])};start=time.time();r=recover_free_seeds(data);free=[]
        for c in r['candidates']:
            measured=refine_and_extend({**data,'a2_record':{'free_segment':c['seed']}});free.append({'candidate_id':c['candidate_id'],'free':compact(measured)})
        io.dump(out/(name+'.json'),{'input':row['output_file'],'input_sha256':row['output_sha256'],'surface_sha256':io.sha(surface),'generated':r,'revalidated':free,'seconds':time.time()-start})
        overlay=data['original'].copy();colors=[(40,230,80),(255,100,60),(30,210,255),(240,90,220),(230,230,30),(200,100,200)]
        for i,c in enumerate(r['candidates']):
            q=np.rint(c['seed']['observed_segment_xy']).astype(np.int32);cv2.line(overlay,tuple(q[0]),tuple(q[1]),colors[i%len(colors)],max(2,int(c['seed']['width_px'])));cv2.putText(overlay,c['candidate_id']+' '+free[i]['free']['status'],tuple(q.mean(axis=0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.8,colors[i%len(colors)],2)
        io.write_image(out/(name+'.jpg'),cv2.resize(overlay,(2048,1500)))
        rec={'input':row['output_file'],'proposals':len(r['candidates']),'revalidated_observed':sum(v['free']['status']=='observed' for v in free),'seconds':time.time()-start,'reject_reasons':[v['free']['diagnostics'].get('reason') for v in free]};results.append(rec);print(json.dumps(rec,ensure_ascii=False),flush=True)
    io.dump(out/'summary.json',{'cases':results,'source_sha256':io.sha(HERE/'free_seed_recovery.py'),'exit_code':0,'scope':'Fixed explicitly open development images; no true identity scoring or production source modifications.'});return 0
if __name__=='__main__':raise SystemExit(main())
