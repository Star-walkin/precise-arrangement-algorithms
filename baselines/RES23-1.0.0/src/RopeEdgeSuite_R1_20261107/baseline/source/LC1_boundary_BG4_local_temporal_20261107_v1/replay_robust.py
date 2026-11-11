"""Final causal replay: spatial and absolute camera evidence unchanged."""
import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import time,shutil,inspect
from collections import Counter as Tally
from bg4_common import *
from temporal_robust import Counter

def main():
    src=ROOT/'full_002';run=ROOT/'full_003';assert not run.exists()
    assert read(src/'receipt.json')['status']=='completed'
    run.mkdir();(run/'frames').mkdir();(run/'snapshot').mkdir()
    files=list(ROOT.glob('*.py'))+[BG3/'global_points.py',BG3/'inputs.py',Path(inspect.getfile(SurfaceObserver))]
    frozen={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,run/'snapshot'/p.name)
    assert sha(SOURCE)==SOURCE_SHA
    dump(run/'binding.json',dict(source=str(SOURCE),source_sha256=SOURCE_SHA,code_sha256=frozen,
        spatial_and_camera_source=str(src),parent_binding_sha256=sha(src/'binding.json'),fixed_d_original_px=D,
        replay_scope='causal filter only; spatial and fixed-reference camera evidence unchanged',absolute_accuracy_validated=False))
    counter=Counter();rows=[];tick=time.perf_counter()
    compact=[]
    for i in range(601):
        p=src/'frames'/f'{i:06d}.json';r=read(p);r['prior_camera_record_sha256']=sha(p)
        r['temporal']=counter.step(r);dump(run/'frames'/f'{i:06d}.json',r)
        keys=('frame_index','nominal_time_s','region','s1','intersection_y_px','camera_motion','observation','temporal','baseline_display_N','baseline_count_valid')
        compact.append({k:r[k] for k in keys})
        # Drop long display-only arrays from compact validation input.
        for obj,key in ((compact[-1]['region'],'top_curve_xy'),(compact[-1]['region'],'top_observed_xy'),(compact[-1]['s1'],'line_points_xy')):obj.pop(key,None)
        rows.append(dict(frame=i,state=r['temporal']['state'],display_N=r['temporal']['display_N']))
        if i%100==0:print(dict(processed=i+1,states=dict(Tally(q['state'] for q in rows))),flush=True)
    assert all(sha(p)==h for p,h in frozen.items())
    dump(run/'compact_validation.json',compact)
    dump(run/'receipt.json',dict(status='completed',frames=601,rows=rows,elapsed_s=time.perf_counter()-tick,
        source_unchanged=sha(SOURCE)==SOURCE_SHA,code_unchanged=True))
    print('FINAL REPLAY COMPLETE',flush=True)
if __name__=='__main__':main()
