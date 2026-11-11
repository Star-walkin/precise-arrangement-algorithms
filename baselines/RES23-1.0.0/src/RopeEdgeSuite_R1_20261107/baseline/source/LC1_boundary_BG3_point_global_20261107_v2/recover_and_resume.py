"""Repair only damaged raw caches from source pixels; resume unchanged frozen models."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,time,zipfile,hashlib,shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np
import inputs
from inputs import ROOT,BASE,SOURCE,sha,dump
from run_bg3 import worker

def read(p):return json.loads(p.read_text(encoding='utf8'))
def row(r):
    b=r['BG3']
    return dict(frame=r['frame_index'],region_valid=r['region_gate_passed'],count_valid=r['count_valid'],display_N=r['display_N'],
        classes=len(b['curves']) if b else 0,points=b['diagnostics']['raw_point_count'] if b else 0,
        assigned=b['diagnostics']['classified_point_count'] if b else 0,reasons=r['reasons'],seconds=r['elapsed_seconds'])

def main():
    cv2.setNumThreads(1);run=ROOT/'full_001';binding=read(run/'binding.json')
    assert not (run/'receipt.json').exists()
    assert sha(SOURCE)==binding['source_sha256']
    for p,h in binding['code_sha256'].items():assert sha(p)==h,p
    start=time.perf_counter();prior=read(run/'progress.json')['elapsed_s'];damaged=[]
    # Existing legacy files are read-only; leave damaged files in their original directory.
    for path in inputs.LEGACY_CACHE.glob('*.npz'):
        try:
            with zipfile.ZipFile(path) as f:
                bad=f.testzip()
                if bad:raise zipfile.BadZipFile(bad)
        except (zipfile.BadZipFile,EOFError) as e:
            damaged.append(dict(path=str(path),sha256=sha(path),frame=int(path.stem),error=repr(e)))
    cap=cv2.VideoCapture(str(SOURCE))
    legacy=inputs.LEGACY_CACHE;inputs.LEGACY_CACHE=ROOT/'__force_fresh_extraction__'
    assert not inputs.LEGACY_CACHE.exists()
    for problem in damaged:
        i=problem['frame'];target=ROOT/'cache'/f'{i:06d}.npz'
        assert not target.exists(),'Do not replace a current cache without inspecting it'
        cap.set(1,i);ok,image=cap.read();assert ok
        old=read(BASE/'frames'/f'{i:06d}.json');data=inputs.extract(image,old)
        assert data is not None
        tmp=target.with_suffix('.repair.npz');np.savez_compressed(tmp,**data);tmp.replace(target)
        problem.update(rebuilt_path=str(target),rebuilt_sha256=sha(target),
            source_pixel_sha256=old['source_pixel_sha256'],same_raw_points_verified=True)
        print('Rebuilt damaged legacy cache from source frame',i,flush=True)
    inputs.LEGACY_CACHE=legacy;cap.release()
    dump(run/'cache_recovery.json',dict(damaged_files=damaged,legacy_files_modified=False,
        recovery_script_sha256=sha(__file__),algorithm_sources_unchanged=True))
    cap=cv2.VideoCapture(str(SOURCE));pending=deque();rows=[];existing=[]
    def consume():rows.append(pending.popleft().result())
    def progress():
        dump(run/'progress.json',dict(processed=len(rows),planned=601,
            region_passed=sum(r['region_valid'] for r in rows),count_valid=sum(r['count_valid'] for r in rows),
            displayed=sum(r['display_N'] is not None for r in rows),classes=sum(r['classes'] for r in rows),
            elapsed_s=prior+time.perf_counter()-start,resumed=True))
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i in binding['frame_indices']:
            ok,image=cap.read();assert ok
            path=run/'frames'/f'{i:06d}.json'
            if path.exists():
                r=read(path);assert hashlib.sha256(image.tobytes()).hexdigest()==r['source_pixel_sha256']
                assert r['frame_index']==i and r['reused_BG1_record_sha256']==sha(BASE/'frames'/f'{i:06d}.json')
                existing.append(dict(frame=i,record_sha256=sha(path)));rows.append(row(r))
            else:
                pending.append(pool.submit(worker,i,image,str(run)))
                if len(pending)>=7:consume()
            if i%20==0:progress();print('Resume reached source frame',i,'completed',len(rows),flush=True)
        while pending:consume();progress()
    cap.release();rows.sort(key=lambda r:r['frame']);assert [r['frame'] for r in rows]==list(range(601))
    assert all(sha(p)==h for p,h in binding['code_sha256'].items())
    assert sha(SOURCE)==binding['source_sha256']
    for item in existing:assert sha(run/'frames'/f"{item['frame']:06d}.json")==item['record_sha256']
    shutil.copy2(__file__,run/'snapshot'/'recover_and_resume.py')
    dump(run/'receipt.json',dict(status='completed',completed=601,frame_rows=rows,
        elapsed_s=prior+time.perf_counter()-start,source_unchanged=True,code_unchanged=True,
        recovery_script_sha256=sha(__file__),resumed=True,preserved_completed_records=existing,
        cache_recovery_sha256=sha(run/'cache_recovery.json'),runtime_excludes_interruption_wait=True))
    print('RESUME COMPLETED',flush=True)
if __name__=='__main__':main()
