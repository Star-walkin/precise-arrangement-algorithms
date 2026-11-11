import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import cv2,copy
from common import *
from tracker_sg5 import Fusion
OLD=PARENT/'RopeEdgeSuite_R15_20261109';STRICT=PARENT/'RopeEdgeSuite_Batch_RES14_20261109'
def get(id,i):
    item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id)
    cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,raw=cap.read();cap.release();assert ok
    im=cv2.flip(raw,1) if item['camera']=='right' else raw
    s=record(STRICT/'runs'/id,i);r=record(OLD/'runs'/id,i)
    assert pixelsha(im)==r['analysis_pixel_sha256']
    assert s['analysis_pixel_sha256']==r['analysis_pixel_sha256']
    return im,s['track'],r['track']
def main():
    cv2.setNumThreads(1);rows=[]
    id='dual_20260923_113745__left'
    for i in [370,397,400]:
        f=Fusion();im,s,r=get(id,i-1);f.step(im,s,r,(i-1)/10,i-1)
        im,s,r=get(id,i);o=f.step(im,s,r,i/10,i)
        assert o['line_abc']==s['line_abc'] and o['valid']
        bad=f.step(im,Fusion.empty('fault_injection_strict_unavailable'),r,i/10,i)
        assert not bad['valid'],bad
        rows.append(dict(id=id,frame=i,strict_correct_retained=True,injected_reflection_rejected=True,gate=bad['diagnostics']['SG5']))
    for id in ['dual_20260923_113029__left','dual_20260924_111500__left']:
        f=Fusion()
        for i in [1,150,200,400,500]:
            im,s,r=get(id,i);o=f.step(im,s,r,i/10,i)
            assert o['valid'],o
            rows.append(dict(id=id,frame=i,recovered=True,branch=o['diagnostics']['SG5']['branch']))
    id='dual_20260923_113305__right'
    for i in [499,540]:
        im,s,r=get(id,i);f=Fusion();o=f.step(im,s,r,i/10,i)
        assert not o['diagnostics']['SG5']['strict_check']['valid']
        assert o['valid'] and o['diagnostics']['SG5']['branch']=='verified_segmented_SG3',o
        rows.append(dict(id=id,frame=i,strict_texture_rejected=True,current_rope_recovered=True))
    dump(ROOT/'sg5_regression_test.json',dict(passed=True,cases=rows,
        note='Targeted current-image algorithm tests, not an accuracy estimate or full temporal evaluation'))
    print('SG5 regression checks passed:',len(rows),flush=True)
if __name__=='__main__':main()
