import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import cv2,numpy as np
from common import *
from engine import Engine
def main():
    tests=[]
    for id in ['dual_20260923_113029__left','dual_20260924_111143__right']:
        item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id)
        cap=cv2.VideoCapture(item['source']);engine=Engine();checks=[]
        for i in range(12):
            ok,raw=cap.read();assert ok;im=cv2.flip(raw,1) if item['camera']=='right' else raw
            fresh=engine.step(im,i/item['fps'],i);cached=record(ROOT/'runs'/id,i)
            assert pixelsha(raw)==cached['source_pixel_sha256']
            assert fresh['track']['valid']==cached['track']['valid']
            assert fresh['track']['diagnostics']['SG5']['branch']==cached['track']['diagnostics']['SG5']['branch']
            if fresh['track']['valid']:
                np.testing.assert_allclose(fresh['track']['line_abc'],cached['track']['line_abc'],atol=1e-8,rtol=0)
            for key in ['fit_valid','count_boundary_valid','reasons']:
                assert fresh['boundary'][key]==cached['boundary'][key],(id,i,key)
            for key in ['edge_y','edge_uncertainty_px']:
                a=fresh['boundary'][key];b=cached['boundary'][key]
                if a is None:assert b is None
                else:assert abs(a-b)<1e-8,(id,i,key,a,b)
            np.testing.assert_allclose(fresh['boundary']['top_curve_xy'],cached['boundary']['top_curve_xy'],atol=1e-8,rtol=0)
            checks.append(dict(frame=i,pixel_hash_match=True,numeric_and_state_checks_equal=True))
        cap.release();tests.append(dict(id=id,checks=checks));print('Fresh/replay parity passed',id,flush=True)
    dump(ROOT/'fresh_replay_parity.json',dict(tests=tests,absolute_tolerance=1e-8,
        scope='24 frames across both cameras; not proof for every frame',engine_sha=sha(ROOT/'engine.py')))
if __name__=='__main__':main()
