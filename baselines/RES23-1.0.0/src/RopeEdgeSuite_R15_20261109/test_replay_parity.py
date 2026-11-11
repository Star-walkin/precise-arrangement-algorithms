"""Verify optimized replay equals a direct fresh pipeline on actual video frames."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import cv2,numpy as np
from common import *
from engine_final import Engine,transform_image

def main():
    tests=[]
    for id in ['dual_20260923_113029__left','dual_20260924_111143__right']:
        item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==id)
        cap=cv2.VideoCapture(item['source']);engine=Engine();checks=[]
        for i in range(12):
            ok,raw=cap.read();assert ok;im=transform_image(raw,item['camera']);fresh=engine.step(im,i/item['fps'],i)
            cached=record(ROOT/'runs_bg9'/id,i);assert not fresh['algorithm_errors']
            assert fresh['track']['valid']==cached['track']['valid']
            if fresh['track']['valid']:np.testing.assert_allclose(fresh['track']['current_line_abc'],cached['track']['current_line_abc'],atol=1e-8,rtol=0)
            for key in ['fit_valid','count_boundary_valid','reasons']:
                assert fresh['boundary'][key]==cached['boundary'][key],(id,i,key)
            for key in ['edge_y','edge_uncertainty_px']:
                a=fresh['boundary'][key];b=cached['boundary'][key]
                if a is None:assert b is None
                else:assert abs(a-b)<1e-8,(id,i,key,a,b)
            np.testing.assert_allclose(fresh['boundary']['top_curve_xy'],cached['boundary']['top_curve_xy'],atol=1e-8,rtol=0)
            checks.append(dict(frame=i,pixel_hash_match=pixelsha(raw)==cached['source_pixel_sha256'],all_numeric_and_state_checks_equal=True))
        cap.release();tests.append(dict(id=id,checks=checks));print('Exact fresh/replay parity',id,len(checks),flush=True)
    dump(ROOT/'replay_parity_test.json',dict(tests=tests,absolute_tolerance=1e-8,
        scope='fresh whole pipeline vs exact-frame stage replay; 24 frames across two cameras',
        engine_final_sha=sha(ROOT/'engine_final.py'),boundary_sha=sha(ROOT/'boundary_bg9.py')))
if __name__=='__main__':main()
