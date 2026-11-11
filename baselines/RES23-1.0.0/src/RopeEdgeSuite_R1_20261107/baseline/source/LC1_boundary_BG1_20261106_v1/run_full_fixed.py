"""601 original frames, current BG1 before F3, bounded independent workers.

Existing S1 observations are reused only after exact decoded-pixel binding.
No temporally held mask/count and no reserved videos are accessed.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import deque,Counter
import cv2,numpy as np,json,time,hashlib,argparse,shutil
from inspect_regions import SOURCE
from gated_rope import analyze
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'LC1_fixed_d2238p7_full_20261106_v1/analysis_anchored/frames'
EXPECTED='ead815ac73239dec6fcbd141ffb2873fb2ac575dfb79347a036ee877b65255f8'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def dump(p,x):p.write_text(json.dumps(plain(x),ensure_ascii=False,allow_nan=False),encoding='utf8')
def work(i,image,outstr):
    cv2.setNumThreads(1);out=Path(outstr);old=OLD/f'frame_{i:06d}.json'
    sr=json.loads(old.read_text(encoding='utf8'))
    pixel=hashlib.sha256(image.tobytes()).hexdigest()
    assert pixel==sr['source_pixel_sha256']
    r,m=analyze(image,sr)
    r.update(frame_index=i,time_s=sr['time_s'],nominal_time_s=i/10,
             source_pixel_sha256=pixel,reused_S1_record_sha256=sha(old),source_video_sha256=EXPECTED)
    r['masks']={}
    for k in ['yellow','surface']:
        path=out/'masks'/f'{i:06d}_{k}.png';cv2.imencode('.png',m[k+'_mask'])[1].tofile(str(path))
        r['masks'][k]={'path':str(path.relative_to(out)),'sha256':sha(path)}
    dump(out/'frames'/f'{i:06d}.json',r)
    return dict(frame=i,region_valid=r['region']['valid'],F3_called=r['F3_called'],count_valid=r['count_valid'],
                display_N=r['display_N'],reasons=r['reasons'],seconds=r['elapsed_seconds'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--stop',type=int,default=601)
    ap.add_argument('--name',default='full_002');args=ap.parse_args()
    out=ROOT/args.name
    if out.exists():raise FileExistsError(out)
    assert sha(SOURCE)==EXPECTED
    out.mkdir();(out/'frames').mkdir();(out/'masks').mkdir();(out/'snapshot').mkdir()
    frozen={}
    for p in list(ROOT.glob('*.py'))+list((ROOT/'vendor').rglob('*.py')):
        frozen[str(p)]=sha(p);dest=out/'snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    dump(out/'binding.json',dict(source=str(SOURCE),sha256=EXPECTED,code_sha256=frozen,
         user_supplied_manifest_sha256=sha(ROOT/'vendor_manifest.json'),frames=args.stop,
         coordinates='original_4096x3000_unmirrored',d_px=2238.7,workers=args.workers,
         region_gate_required_before_F3=True,static_SAM_surface_used=False,full_visible_surface_F3=True,
         F3_processing_scale=.5,region_local_edge_refinement_scale=1.,
         historical_C3_F2_fusion_replayed=False,absolute_accuracy_validated=False))
    c=cv2.VideoCapture(str(SOURCE));assert int(c.get(cv2.CAP_PROP_FRAME_COUNT))==601
    start=time.perf_counter();rows=[];pending=deque()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        def consume():
            rows.append(pending.popleft().result())
            prog=dict(processed=len(rows),planned=args.stop,region_valid=sum(r['region_valid'] for r in rows),
                      F3_called=sum(r['F3_called'] for r in rows),count_valid=sum(r['count_valid'] for r in rows),
                      displayed=sum(r['display_N'] is not None for r in rows),elapsed_s=time.perf_counter()-start)
            dump(out/'progress.json',prog)
            if len(rows)%10==0:print(prog,flush=True)
        for i in range(args.stop):
            ok,a=c.read();assert ok
            pending.append(pool.submit(work,i,a,str(out)))
            if len(pending)>=args.workers+2:consume()
        while pending:consume()
    c.release();assert all(sha(p)==v for p,v in frozen.items())
    prog=json.loads((out/'progress.json').read_text(encoding='utf8'))
    dump(out/'receipt.json',dict(status='completed',**prog,frame_rows=rows,source_unchanged=sha(SOURCE)==EXPECTED,
                               gate_blocks=sum(not r['region_valid'] for r in rows),
                               region_failed_F3_calls=sum(r['F3_called'] and not r['region_valid'] for r in rows)))
    print('COMPLETED',prog,flush=True)
if __name__=='__main__':main()
