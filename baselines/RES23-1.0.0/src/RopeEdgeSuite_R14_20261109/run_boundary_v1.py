import time,gzip,json
import cv2
from common import *
from boundary_bg7 import Boundary

def main():
    cv2.setNumThreads(1);out=ROOT/'boundary_run';out.mkdir(exist_ok=False);(out/'frames').mkdir()
    bind=read(RUN12/'binding.json');source=bind['source']
    dump(out/'binding.json',dict(source=source,source_sha256=bind['source_sha256'],frames=600,
        boundary_code_sha256=sha(ROOT/'boundary_bg7.py'),runner_sha256=sha(__file__),
        reused_same_frame_SG2=True,reused_BG5_candidates_only=True,no_future_frames_used=True))
    cap=cv2.VideoCapture(source);b=Boundary();rows=[];start=time.perf_counter()
    for i in range(600):
        ok,im=cap.read();assert ok;r=record(i);assert pixelsha(im)==r['source_pixel_sha256']
        n=b.step(im,r['region_BG5'],r['S1_tracker_full'],i)
        n.update(source_pixel_sha256=pixelsha(im),source_record_sha256=sha(RUN12/'frames'/f'{i:06d}.json.gz'),
            camera_valid=r['camera_motion'].get('valid',True))
        if not n['camera_valid']:
            n['fit_valid']=False;n['count_boundary_valid']=False;n['reasons'].append('camera_motion_invalid')
            b=Boundary()
        p=out/'frames'/f'{i:06d}.json.gz';tmp=p.with_suffix('.part')
        with gzip.open(tmp,'wt',encoding='utf8') as f:json.dump(plain(n),f,ensure_ascii=False,allow_nan=False)
        tmp.replace(p)
        rows.append({k:n.get(k) for k in ['frame','fit_valid','count_boundary_valid','d_locally_observed','d_foreground_occluded','edge_y','edge_uncertainty_px','reasons']})
        if i%50==0 or i==599:
            v=dict(processed=i+1,fit=sum(q['fit_valid'] for q in rows),count_boundary=sum(q['count_boundary_valid'] for q in rows),elapsed_s=round(time.perf_counter()-start,1))
            print(v,flush=True);dump(out/'progress.json',v)
    cap.release();dump(out/'compact.json',rows)
    assert sha(source)==bind['source_sha256']
    dump(out/'receipt.json',dict(complete=True,frames=600,source_unchanged=True,elapsed_s=time.perf_counter()-start))
if __name__=='__main__':main()
