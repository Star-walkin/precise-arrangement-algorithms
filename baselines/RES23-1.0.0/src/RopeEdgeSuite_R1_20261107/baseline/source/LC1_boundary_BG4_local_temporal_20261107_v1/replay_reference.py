"""Recheck actual video background, preserve spatial evidence, replay causal state."""
import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2,time,hashlib,shutil,inspect
from collections import Counter as Tally
from bg4_common import *
from camera_reference import CameraContinuity
from temporal_count import Counter

def main():
    run=ROOT/'full_002';assert not run.exists();run.mkdir();(run/'frames').mkdir();(run/'snapshot').mkdir()
    oldrun=ROOT/'full_001';oldbinding=read(oldrun/'binding.json');assert read(oldrun/'receipt.json')['status']=='completed'
    # Spatial output was produced with frozen code; verify it before reuse.
    for name in ('bg4_common.py','local_observation.py','temporal_count.py'):
        assert sha(ROOT/name)==oldbinding['code_sha256'][str(ROOT/name)]
    assert sha(SOURCE)==SOURCE_SHA
    files=list(ROOT.glob('*.py'))+[BG3/'global_points.py',BG3/'inputs.py',Path(inspect.getfile(SurfaceObserver))]
    frozen={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,run/'snapshot'/p.name)
    dump(run/'binding.json',dict(source=str(SOURCE),source_sha256=SOURCE_SHA,code_sha256=frozen,
        spatial_source=str(oldrun),spatial_binding_sha256=sha(oldrun/'binding.json'),fixed_d_original_px=D,
        replay_scope='new fixed-reference camera registration and causal temporal replay; spatial records unchanged',
        absolute_accuracy_validated=False))
    cap=cv2.VideoCapture(str(SOURCE));cv2.setNumThreads(1);camera=CameraContinuity();counter=Counter();rows=[];tick=time.perf_counter()
    for i in range(601):
        source=oldrun/'frames'/f'{i:06d}.json';r=read(source);ok,a=cap.read()
        assert ok and hashlib.sha256(a.tobytes()).hexdigest()==r['source_pixel_sha256']
        r['prior_spatial_record_sha256']=sha(source);r['camera_motion']=camera.step(a);r['temporal']=counter.step(r)
        dump(run/'frames'/f'{i:06d}.json',r)
        rows.append(dict(frame=i,state=r['temporal']['state'],display_N=r['temporal']['display_N']))
        if i%50==0:
            prog=dict(processed=i+1,planned=601,states=dict(Tally(q['state'] for q in rows)),elapsed_s=time.perf_counter()-tick)
            print(prog,flush=True);dump(run/'progress.json',prog)
    cap.release();assert all(sha(p)==h for p,h in frozen.items())
    dump(run/'receipt.json',dict(status='completed',frames=601,elapsed_s=time.perf_counter()-tick,rows=rows,
        source_unchanged=sha(SOURCE)==SOURCE_SHA,code_unchanged=True))
    dump(ROOT/'delivery'/'superseded_notice.json',dict(status='partial_video_superseded',reason='Accumulated optical-flow drift caused false camera-change alarms. Use delivery_final after fixed-reference replay.',
        retained_for_audit=True))
if __name__=='__main__':main()
