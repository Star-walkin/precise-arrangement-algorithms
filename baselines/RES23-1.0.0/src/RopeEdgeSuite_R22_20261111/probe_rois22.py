from common22 import *
from extract_rotation22 import rois_from_boundary
import cv2
item=next(x for x in plan() if '111500' in x['id'])
cap=cv2.VideoCapture(item['source'])
for i in range(8):
    ok,raw=cap.read();assert ok
    r=frontend(item['id'],i)
    print(i,r['boundary'].get('fit_valid'))
    if r['boundary'].get('fit_valid'):
        small=cv2.resize(raw,(1024,750),interpolation=cv2.INTER_AREA)
        rois=rois_from_boundary(r,small)
        print('ROIS',[(q['box'],q['mask'].mean()) for q in rois])
        print('BOUNDARY',r['boundary']['normalized_x'],r['boundary']['fit_coefficients_ascending'])
        for q in rois:
            x0,y0,x1,y1=q['box'];cv2.rectangle(small,(x0,y0),(x1,y1),(0,0,255),2)
        cv2.imencode('.jpg',small)[1].tofile(ROOT/'inspection/rois.jpg')
        break
cap.release()
