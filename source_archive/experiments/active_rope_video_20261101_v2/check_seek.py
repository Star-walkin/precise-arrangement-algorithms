"""Verify every chunk warm-start seek against sequential source decoding."""
import sys,json,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.append('D:/precise_arrangement/vision_prototype/.venv/Lib/site-packages')
import cv2,numpy as np
HERE=Path(__file__).resolve().parent
def check(v):
    cv2.setNumThreads(1);fps=v['nominal_fps'];n=v['frame_count_metadata'];size=round(20*fps)
    starts=sorted(set(max(0,i-int(np.ceil(fps))) for i in range(0,n,size)))
    last=max(starts);cap=cv2.VideoCapture(v['canonical_path']);references={}
    for i in range(last+1):
        if not cap.grab():raise RuntimeError('Sequential decode failed')
        if i in starts:
            ok,img=cap.retrieve();assert ok;references[i]=hashlib.sha256(img.tobytes()).hexdigest()
    cap.release();cap=cv2.VideoCapture(v['canonical_path']);rows=[]
    for i in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();assert ok
        h=hashlib.sha256(img.tobytes()).hexdigest();same=h==references[i]
        rows.append(dict(frame=i,pixel_identical=same,sha256=h));assert same,(v['video_id'],i)
    cap.release();return dict(video_id=v['video_id'],seeks=rows)
def main():
    t=time.perf_counter();v=json.loads((HERE.parent/'active_rope_video_20261101_v1/source_trace/source_trace.json').read_text(encoding='utf8'))['videos'];rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(check,x) for x in v]
        for f in as_completed(futures):
            r=f.result();rows.append(r);print(json.dumps(dict(video=r['video_id'],seeks=len(r['seeks']),passed=True)),flush=True)
    (HERE/'seek_verification.json').write_text(json.dumps(dict(status='passed',videos=16,chunk_starts=sum(len(r['seeks']) for r in rows),
        rows=rows,seconds=time.perf_counter()-t,hardware=False),ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
