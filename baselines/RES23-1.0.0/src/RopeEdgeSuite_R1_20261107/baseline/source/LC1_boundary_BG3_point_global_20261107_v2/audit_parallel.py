"""Bounded parallel execution of the same per-frame audit."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time,traceback
from collections import Counter,deque
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np
from audit_results import audit_frame,read,wait,summary
from inputs import ROOT,BASE,sha,dump
BG2_RECEIPT=ROOT.parent/'LC1_boundary_BG2_20261106_v1'/'full_001'/'receipt.json'
BG2_ROWS={r['frame']:r for r in read(BG2_RECEIPT)['frame_rows']}

def worker(i):
    cv2.setNumThreads(1);run=ROOT/'full_001';r=wait(run/'frames'/f'{i:06d}.json')
    old=read(BASE/'frames'/f'{i:06d}.json');data=None;counts=Counter();issue=None;error=None
    if r['region_gate_passed']:
        with np.load(ROOT/'cache'/f'{i:06d}.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    try:error=audit_frame(r,old,data,counts)
    except Exception as e:issue=dict(frame=i,error=repr(e),traceback=traceback.format_exc())
    bg2=BG2_ROWS[i]
    rejected=Counter()
    if r['BG3']:
        for c in r['BG3']['diagnostics']['rejections']:rejected.update(c['reasons'])
    return dict(frame=i,counts=dict(counts),q90=error,issue=issue,gate=r['region_gate_passed'],
        valid=r['count_valid'],display=r['display_N'],BG1_display=old['display_N'],
        BG2_display=bg2['display_N'],BG2_valid=bg2['count_valid'],reasons=r['reasons'],rejections=dict(rejected))

def main():
    run=ROOT/'full_001';out=ROOT/'delivery';binding=read(run/'binding.json')
    auditor_sha=sha(ROOT/'audit_results.py');driver_sha=sha(__file__)
    rows=[];issues=[];counts=Counter();reasons=Counter();rejections=Counter();q90=[];pending=deque();start=time.perf_counter()
    def consume():
        r=pending.popleft().result();rows.append(r);counts.update(r['counts']);reasons.update(r['reasons']);rejections.update(r['rejections'])
        if r['q90'] is not None:q90.append(r['q90'])
        if r['issue']:issues.append(r['issue']);print('AUDIT ISSUE',r['issue'],flush=True)
        if len(rows)%20==0:
            dump(run/'audit_progress.json',dict(audited=len(rows),issues=issues,elapsed_s=time.perf_counter()-start))
            print('Audited',len(rows),'issues',len(issues),flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i in binding['frame_indices']:
            pending.append(pool.submit(worker,i))
            if len(pending)>=5:consume()
        while pending:consume()
    receipt=wait(run/'receipt.json');assert receipt['status']=='completed'
    for p,h in binding['code_sha256'].items():assert sha(p)==h,p
    assert sha(ROOT/'audit_results.py')==auditor_sha and sha(__file__)==driver_sha
    metrics=dict(frames=len(rows),region_passed=sum(r['gate'] for r in rows),region_blocked=sum(not r['gate'] for r in rows),
        new_count_valid=sum(r['valid'] for r in rows),new_displayed=sum(r['display'] is not None for r in rows),
        BG2_count_valid=sum(r['BG2_valid'] for r in rows),BG2_displayed=sum(r['BG2_display'] is not None for r in rows),
        counts=dict(counts),reasons=dict(reasons),whole_class_rejection_reasons=dict(rejections),
        algorithm_runtime_s=receipt['elapsed_s'],absolute_physical_accuracy=None,
        raw_point_fit_q90_pitch_median=float(np.median(q90)),raw_point_fit_q90_pitch_max=float(np.max(q90)),matched_jumps={})
    for label in ('BG1','BG2'):
        pairs=[]
        for a,b in zip(rows,rows[1:]):
            if not 121<=b['frame']<=270:continue
            vals=[a['display'],b['display'],a[label+'_display'],b[label+'_display']]
            if all(v is not None for v in vals):pairs.append(dict(frame=b['frame'],old=vals[3]-vals[2],new=vals[1]-vals[0]))
        metrics['matched_jumps'][label]=dict(window=[121,270],pairs=len(pairs),
            old=summary([p['old'] for p in pairs]),new=summary([p['new'] for p in pairs]),details=pairs)
    dump(out/'geometry_audit.json',dict(status='failed' if issues else 'passed',frames=len(rows),issues=issues,counters=dict(counts),
        elapsed_s=time.perf_counter()-start,code_hashes_unchanged=True,per_frame_auditor_sha256=auditor_sha,
        driver_sha256=driver_sha,BG2_comparison_receipt_sha256=sha(BG2_RECEIPT),
        notes='All curve samples, original masks, unique points/intersections and analytic order at 0.5 processing pixel tested.'))
    dump(out/'metrics.json',metrics);print('AUDIT COMPLETED issues',len(issues),flush=True)
    if issues:raise SystemExit(1)
if __name__=='__main__':main()
