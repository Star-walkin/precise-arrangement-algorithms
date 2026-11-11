import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import argparse,json,time,shutil,cv2
from inputs import ROOT,BASE,SOURCE,sha,dump,load_or_extract,core
from pipeline import analyze

def worker(i,image,out):
    cv2.setNumThreads(1);t=time.perf_counter()
    old=json.loads((BASE/'frames'/f'{i:06d}.json').read_text(encoding='utf8'))
    data=load_or_extract(i,image,old,ROOT/'cache')
    r=analyze(image,old,data);r['elapsed_seconds']=time.perf_counter()-t
    from pathlib import Path
    dump(Path(out)/'frames'/f'{i:06d}.json',r)
    b=r['BG3']
    return dict(frame=i,region_valid=r['region_gate_passed'],count_valid=r['count_valid'],display_N=r['display_N'],
        classes=len(b['curves']) if b else 0,points=b['diagnostics']['raw_point_count'] if b else 0,
        assigned=b['diagnostics']['classified_point_count'] if b else 0,reasons=r['reasons'],seconds=r['elapsed_seconds'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='pilot_001');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--frames',default='0,60,121,127,163,200,240,270,300,450,600');ap.add_argument('--full',action='store_true')
    a=ap.parse_args();ids=list(range(601)) if a.full else [int(x) for x in a.frames.split(',')]
    out=ROOT/a.name
    if out.exists():raise FileExistsError(out)
    out.mkdir();(out/'frames').mkdir();(out/'snapshot').mkdir()
    files=list(ROOT.glob('*.py'))+core.source_dependencies()+[core.__file__]
    frozen={str(p):sha(p) for p in files}
    for p in ROOT.glob('*.py'):shutil.copy2(p,out/'snapshot'/p.name)
    expected='ead815ac73239dec6fcbd141ffb2873fb2ac575dfb79347a036ee877b65255f8'
    assert sha(SOURCE)==expected
    dump(out/'binding.json',dict(source=str(SOURCE),source_sha256=expected,code_sha256=frozen,
        frame_indices=ids,source_BG1=str(BASE),fixed_d_original_px=2238.7,processing_scale=.5,
        yellow_points='fresh or verified same-pixel raw-point cache',old_fitted_curves_used=False,absolute_accuracy_validated=False))
    cap=cv2.VideoCapture(str(SOURCE));pending=deque();rows=[];t=time.perf_counter()
    def consume():
        rows.append(pending.popleft().result())
        prog=dict(processed=len(rows),planned=len(ids),region_passed=sum(r['region_valid'] for r in rows),
            count_valid=sum(r['count_valid'] for r in rows),displayed=sum(r['display_N'] is not None for r in rows),
            classes=sum(r['classes'] for r in rows),elapsed_s=time.perf_counter()-t)
        dump(out/'progress.json',prog)
        if len(rows)%10==0 or len(ids)<20:print(prog,flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i in ids:
            if not a.full:cap.set(1,i)
            ok,image=cap.read();assert ok
            pending.append(pool.submit(worker,i,image,str(out)))
            if len(pending)>=a.workers+1:consume()
        while pending:consume()
    cap.release();assert all(sha(p)==h for p,h in frozen.items())
    dump(out/'receipt.json',dict(status='completed',completed=len(rows),frame_rows=rows,
        elapsed_s=time.perf_counter()-t,source_unchanged=sha(SOURCE)==expected,code_unchanged=True))
    print('COMPLETED',out,flush=True)
if __name__=='__main__':main()
