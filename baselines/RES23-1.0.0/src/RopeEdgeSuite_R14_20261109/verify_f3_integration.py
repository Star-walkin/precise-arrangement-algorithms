import sys,time,importlib.util
import cv2,numpy as np
from common import *
from roi_adapter import build_roi
def main():
    cv2.setNumThreads(1)
    spec=importlib.util.spec_from_file_location('res12_extract_fresh',R12/'run_full.py')
    oldmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmodule)
    ids=[141,261,336,370,380,388,461,581];out=ROOT/'F3_integration';out.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(read(RUN12/'binding.json')['source']);rows=[]
    for i in ids:
        tick=time.perf_counter();cap.set(1,i);ok,im=cap.read();assert ok;r=record(i);n=record(i,ROOT/'boundary_final')
        assert pixelsha(im)==n['source_pixel_sha256']==r['source_pixel_sha256']
        original=cv2.imdecode(np.fromfile(FRONT12/'regions'/f'{i:06d}.png',np.uint8),0)
        mask,meta=build_roi(original,n);points=[]
        if meta['spatial_roi_valid']:
            fresh=oldmodule.extract_current(im,mask,r['S1_tracker_full']);points=fresh['points']
            np.savez_compressed(out/f'points_{i:06d}.npz',points=points,source_pixel_sha256=pixelsha(im),mask_sha256=pixelsha(mask))
        cv2.imencode('.png',mask)[1].tofile(out/f'roi_{i:06d}.png')
        item=dict(frame=i,roi=meta,fresh_F3_called=meta['spatial_roi_valid'],points=len(points),
            source_pixel_sha256=pixelsha(im),mask_sha256=pixelsha(mask),elapsed_s=time.perf_counter()-tick)
        assert not (n.get('d_foreground_occluded') and meta['count_ready'])
        if not n['fit_valid']:assert len(points)==0 and not meta['spatial_roi_valid']
        rows.append(item);dump(out/'progress.json',rows);print(i,len(points),round(item['elapsed_s'],1),flush=True)
    cap.release();dump(ROOT/'F3_integration_receipt.json',dict(passed=True,frames=ids,rows=rows,
        fresh_original_01_F3_on_new_ROI=True,full_video_F3_recomputed=False,root_counter_rerun=False,
        adapter_sha256=sha(ROOT/'roi_adapter.py')))
if __name__=='__main__':main()
