import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='1'
import argparse,concurrent.futures,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2
from PIL import Image
from common import *
from engine import Engine,transform_image,original_abc
from render_batch import panel

CASES=[('dual_20260923_113029__left',i,True) for i in [0,200,500]]+[
    ('dual_20260924_111500__left',i,True) for i in [0,150,400,500]]+[
    ('dual_20260923_113305__right',i,True) for i in [250,300,500]]+[
    ('dual_20260923_113745__left',370,True),('dual_20260923_101517__right',200,True),
    ('dual_20260923_101517__left',200,False),('dual_20260924_111500__right',200,False),
    ('dual_20260923_113305__left',300,False)]

def run_case(args):
    revision,id,start,expected,n=args
    item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id)
    out=ROOT/'experiments'/revision/f'{id}__{start:06d}';out.mkdir(parents=True,exist_ok=False)
    engine=Engine();cap=cv2.VideoCapture(item['source']);cap.set(1,start);rows=[];tick=time.time()
    for i in range(start,start+n):
        ok,raw=cap.read();assert ok;im=transform_image(raw,item['camera']);r=engine.step(im,i/item['fps'],i)
        if r['algorithm_errors']:raise RuntimeError(str(r['algorithm_errors']))
        r.update(frame_index=i,time_s=i/item['fps'],source_pixel_sha256=pixelsha(raw),analysis_pixel_sha256=pixelsha(im),
            original_line_abc=original_abc(r['track']['current_line_abc'],im.shape[1],item['camera']))
        save_record(out,i,r)
        if i==start+n-1:Image.fromarray(panel(raw,r,item)).save(out/'last.jpg',quality=94)
        rows.append(dict(frame=i,track=r['track']['valid'],boundary=r['boundary']['fit_valid'],
            errors=r['algorithm_errors'],track_reason=r['track']['diagnostics']['accept_reason'],
            boundary_reasons=r['boundary']['reasons'],abc=r['original_line_abc']))
    cap.release();result=dict(id=id,start=start,frames=n,expected_visible_rope_in_review=expected,
        tracking=sum(r['track'] for r in rows),boundary=sum(r['boundary'] for r in rows),
        elapsed_s=time.time()-tick,rows=rows,folder=str(out))
    dump(out/'summary.json',result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--revision',required=True);ap.add_argument('--frames',type=int,default=6)
    ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();results=[]
    import shutil
    snap=ROOT/'experiments'/a.revision/'code_snapshot';snap.mkdir(parents=True,exist_ok=False)
    for p in ['tracker_sg3.py','boundary_bg8.py','camera_cg2.py','engine.py','probe_cases.py']:
        shutil.copy2(ROOT/p,snap/p)
    dump(snap/'hashes.json',{p.name:sha(p) for p in snap.glob('*.py')})
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(run_case,(a.revision,*c,a.frames)) for c in CASES]
        for f in concurrent.futures.as_completed(fs):
            r=f.result();results.append(r);dump(ROOT/'experiments'/a.revision/'summary.json',results)
            print(r['id'],r['start'],'track',r['tracking'],'boundary',r['boundary'],flush=True)
if __name__=='__main__':main()
