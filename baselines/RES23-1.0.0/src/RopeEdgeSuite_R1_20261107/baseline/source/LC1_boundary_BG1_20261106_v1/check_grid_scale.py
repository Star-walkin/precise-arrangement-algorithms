"""Same-frame original/half-grid check on fixed near-edge review frames."""
from pathlib import Path
import cv2,json,numpy as np
from run_full import ROOT,OLD,SOURCE,dump,sha
from gated_rope import analyze
if __name__=='__main__':
    cv2.setNumThreads(1);out=ROOT/'grid_scale_check';out.mkdir(exist_ok=True)
    c=cv2.VideoCapture(str(SOURCE));rows=[]
    for i in [127,200,240]:
        c.set(1,i);ok,a=c.read();assert ok
        sr=json.loads((OLD/f'frame_{i:06d}.json').read_text(encoding='utf8'))
        full,_=analyze(a,sr,processing_scale=1.)
        half=json.loads((ROOT/'full_002/frames'/f'{i:06d}.json').read_text(encoding='utf8'))
        dump(out/f'{i:03d}_original_grid.json',full)
        f=full['F3']['continuous_pitch_distance'] if full['count_valid'] else None
        h=half['F3']['continuous_pitch_distance'] if half['count_valid'] else None
        rec=dict(frame=i,original_grid_count_valid=full['count_valid'],half_grid_count_valid=half['count_valid'],
                 original_N=f,half_N=h,difference=None if f is None or h is None else h-f,
                 original_reasons=full['reasons'],half_reasons=half['reasons'])
        rows.append(rec);print(rec,flush=True)
    c.release();dump(out/'summary.json',dict(records=rows,scope='grid sensitivity, not accuracy against human ground truth'))
