"""Replay two earliest anchor inputs against frozen B numerical responses.

This audit alone loads the historical short seeds/responses for comparison.
FrameEngine video prediction never consumes these reference heatmaps.
"""
import json,time
from engine import FrameEngine,frozen_tools,js,sha,H,H6,HERE,cv2,np

def main():
    ids=['dual_20260923_101517__pair0015__right','L_run_20260923_01__f000015__left']
    jobs={j['id']:j for j in js(H6/'inputs.json')['jobs']}
    engine=FrameEngine.__new__(FrameEngine);engine.net=frozen_tools()[-1];engine.forward_total=0
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(1)
    rows=[]
    for ident in ids:
        job=jobs[ident];bgr=cv2.imdecode(np.fromfile(job['image_path'],np.uint8),cv2.IMREAD_COLOR)
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        gray=cv2.resize(gray[1050:],(2048,975),interpolation=cv2.INTER_AREA)
        t=time.perf_counter();actual,normalization=engine._forward(gray,job['seeds'][0])
        path=job['response_files'][0]
        with np.load(path,allow_pickle=False) as z:expected=z['response']
        error=np.abs(actual-expected)
        rows.append(dict(id=ident,input_sha256=sha(job['image_path']),response_path=path,response_sha256=sha(path),
            seed=job['seeds'][0],max_absolute_error=float(error.max()),mean_absolute_error=float(error.mean()),
            allclose_1e_5=bool(np.allclose(actual,expected,atol=1e-5,rtol=0)),
            seconds=time.perf_counter()-t,normalization=normalization))
    result=dict(engine_sha256=sha(HERE/'engine.py'),tolerance_absolute=1e-5,rows=rows,
        passed=all(r['allclose_1e_5'] for r in rows),historical_response_used_only_for_this_numeric_audit=True)
    (HERE/'B_replay_parity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=True),flush=True)
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
