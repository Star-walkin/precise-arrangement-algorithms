"""Inspect every previously rejected color-fragment frame, before promotion."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2,json,time
from diagnose import ROOT,OLD,read
from region_bg5_refined import segment
from pilot import review

def main():
    cv2.setNumThreads(1);out=ROOT/'refinement_probe';out.mkdir(exist_ok=True)
    run=OLD/'transfer_runs/dual_20260923_113745';cap=cv2.VideoCapture(read(run/'binding.json')['source'])
    ids=[r['frame'] for r in read(ROOT/'color_diagnostic.json')];rows=[];start=time.perf_counter()
    for i in ids:
        cap.set(1,i);ok,im=cap.read();assert ok
        old=read(run/'bg1/frames'/f'{i:06d}.json')['region'];new=segment(im);r=new['record']
        rows.append(dict(frame=i,valid=r['valid'],reasons=r['reasons'],edge=r['edge_y'],fragment_association=r.get('yellow_fragment_association')))
        if i in [12,146,280,404,406,583]:review(im,new,old,f'BG5.1 / {i/10:.1f}s',out/f'preview_{i:03d}.jpg')
        print(i,r['valid'],r['reasons'],flush=True)
    cap.release();(ROOT/'refinement_probe.json').write_text(json.dumps(dict(frames=len(rows),valid=sum(x['valid'] for x in rows),elapsed_s=time.perf_counter()-start,rows=rows),indent=2),encoding='utf8')

if __name__=='__main__':main()
