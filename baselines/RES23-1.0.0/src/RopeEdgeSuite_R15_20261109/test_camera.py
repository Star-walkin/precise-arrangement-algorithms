import cv2,numpy as np
from common import *
from camera_cg2 import CameraContinuity
cv2.setNumThreads(1)
item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']=='dual_20260923_101643__right')
cap=cv2.VideoCapture(item['source']);tracker=CameraContinuity();rows=[];first=None
for i in range(85):
    ok,im=cap.read();assert ok;im=cv2.flip(im,1)
    if first is None:first=im.copy()
    q=tracker.step(im);rows.append(dict(frame=i,**q))
cap.release()
test=CameraContinuity();test.step(first)
shifted=cv2.warpAffine(first,np.float32([[1,0,16],[0,1,0]]),(4096,3000))
movement=[test.step(shifted) for _ in range(3)]
assert movement[-1]['camera_changed'],'Known 16 px displacement must invalidate reference'
dump(ROOT/'experiments/camera_test.json',dict(rows=rows,synthetic_16px_translation=movement))
print('Confirmed',sum(q['confirmed'] for q in rows),'/',len(rows),'movement flagged',movement[-1]['camera_changed'])
print('Failures',[(r['frame'],r.get('inlier_ratio'),r.get('occupied_background_bins'),r.get('inlier_span_fraction'),r.get('inlier_residual_q95')) for r in rows if not r['confirmed']])
