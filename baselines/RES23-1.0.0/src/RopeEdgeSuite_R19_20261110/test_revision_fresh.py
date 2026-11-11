import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import cv2,numpy as np
from common import *
from engine import Engine
def main():
    tests=[]
    for id,limit,runroot in [('dual_20260925_232941__right',9,ROOT),('dual_20260923_113029__left',12,PARENT/'RopeEdgeSuite_R18_20261110')]:
        item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id);cap=cv2.VideoCapture(item['source']);engine=Engine();checks=[]
        for i in range(limit):
            ok,raw=cap.read();assert ok;im=cv2.flip(raw,1) if item['camera']=='right' else raw
            fresh=engine.step(im,i/item['fps'],i);cached=record(runroot/'runs'/id,i)
            assert pixelsha(raw)==cached['source_pixel_sha256'];assert fresh['track']['valid']==cached['track']['valid']
            if fresh['track']['valid']:np.testing.assert_allclose(fresh['track']['line_abc'],cached['track']['line_abc'],atol=1e-8,rtol=0)
            for k in ['fit_valid','count_boundary_valid','reasons']:assert fresh['boundary'][k]==cached['boundary'][k],(id,i,k)
            for k in ['edge_y','edge_uncertainty_px']:
                a=fresh['boundary'][k];b=cached['boundary'][k]
                if a is None:assert b is None
                else:assert abs(a-b)<1e-8,(id,i,k,a,b)
            np.testing.assert_allclose(fresh['boundary']['top_curve_xy'],cached['boundary']['top_curve_xy'],atol=1e-8,rtol=0)
            checks.append(dict(frame=i,all_numeric_checks_equal=True,source_pixel_verified=True))
        cap.release();tests.append(dict(id=id,frames=checks));print('Revision fresh parity',id,limit,flush=True)
    dump(ROOT/'revision_fresh_parity.json',dict(tests=tests,absolute_tolerance=1e-8,engine_sha=sha(ROOT/'engine.py')))
if __name__=='__main__':main()
