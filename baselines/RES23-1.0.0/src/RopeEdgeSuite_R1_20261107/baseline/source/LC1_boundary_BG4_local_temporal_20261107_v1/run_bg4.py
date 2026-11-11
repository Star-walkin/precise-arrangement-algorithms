import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse,time,shutil,hashlib
from collections import deque,Counter as Tally
from concurrent.futures import ProcessPoolExecutor
import cv2
from bg4_common import *
from local_observation import analyze,PARAMETERS as LOCAL_PARAMETERS
from temporal_count import Counter,PARAMETERS as TEMPORAL_PARAMETERS
from camera_reference import CameraContinuity

def worker(i,pixelsha,motion):
    cv2.setNumThreads(1);return analyze(i,pixelsha,motion)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='run_current_001');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--limit',type=int,default=601);a=ap.parse_args()
    out=ROOT/a.name
    if out.exists():raise FileExistsError(out)
    out.mkdir();(out/'frames').mkdir();(out/'snapshot').mkdir()
    deps=list(ROOT.glob('*.py'))+[BG3/'global_points.py',BG3/'inputs.py']
    frozen={str(p):sha(p) for p in deps}
    for p in ROOT.glob('*.py'):shutil.copy2(p,out/'snapshot'/p.name)
    assert sha(SOURCE)==SOURCE_SHA
    dump(out/'binding.json',dict(source=str(SOURCE),source_sha256=SOURCE_SHA,code_sha256=frozen,
        fixed_d_original_px=D,reused_current_pixel_bound_BG3_cache=True,baseline_read_only=True,
        local_parameters=LOCAL_PARAMETERS,temporal_parameters=TEMPORAL_PARAMETERS,absolute_accuracy_validated=False))
    cap=cv2.VideoCapture(str(SOURCE));camera=CameraContinuity();counter=Counter();pending=deque();rows=[];start=time.perf_counter()
    def consume():
        r=pending.popleft().result();r['temporal']=counter.step(r);dump(out/'frames'/f"{r['frame_index']:06d}.json",r)
        rows.append(dict(frame=r['frame_index'],state=r['temporal']['state'],display_N=r['temporal']['display_N']))
        if len(rows)%20==0 or len(rows)==a.limit:
            progress=dict(processed=len(rows),planned=a.limit,states=dict(Tally(q['state'] for q in rows)),
                displayed=sum(q['display_N'] is not None for q in rows),elapsed_s=time.perf_counter()-start)
            dump(out/'progress.json',progress);print(progress,flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i in range(a.limit):
            ok,image=cap.read();assert ok
            motion=camera.step(image);pixelsha=hashlib.sha256(image.tobytes()).hexdigest()
            pending.append(pool.submit(worker,i,pixelsha,motion))
            if len(pending)>=a.workers+2:consume()
        while pending:consume()
    cap.release();assert all(sha(p)==h for p,h in frozen.items())
    dump(out/'receipt.json',dict(status='completed',frames=len(rows),rows=rows,elapsed_s=time.perf_counter()-start,
        source_unchanged=sha(SOURCE)==SOURCE_SHA,code_unchanged=True))
    print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
