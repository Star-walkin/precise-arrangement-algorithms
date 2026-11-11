from pathlib import Path
import cv2,json,numpy as np,time
from region_gate import segment
ROOT=Path(__file__).resolve().parent
SOURCE=Path(r'E:\study\program\济宁\双相机同步\dual_20260924_111500\left_DA9627065.avi')
def overlay(image,result):
    a=image.copy();y=result['yellow_mask']>0;m=result['surface_mask']>0
    a[y]=(a[y]*.70+np.array([0,180,255])*.30).astype(np.uint8)
    a[m]=(a[m]*.75+np.array([255,130,0])*.25).astype(np.uint8)
    r=result['record']
    for key,col in [('yellow_contours_xy',(0,170,255)),('surface_contours_xy',(255,140,0))]:
        for q in r.get(key,[]):cv2.polylines(a,[np.asarray(q,np.int32)],True,col,4)
    for x,y in r.get('top_observed_xy',[]):cv2.circle(a,(int(x),int(y)),4,(255,0,255),-1)
    cv2.line(a,(2239,1800),(2239,a.shape[0]-1),(0,255,255),3)
    if r['edge_y'] is not None:cv2.circle(a,(2239,round(r['edge_y'])),14,(255,255,255),4)
    return a
if __name__=='__main__':
    out=ROOT/'pilot_regions_v2';out.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(SOURCE))
    for i in [0,60,121,163,200,240,270,300,450,600]:
        cap.set(1,i);ok,a=cap.read();assert ok;t=time.perf_counter();r=segment(a)
        (out/f'{i:03d}.json').write_text(json.dumps(r['record'],ensure_ascii=False),encoding='utf-8')
        cv2.imencode('.jpg',cv2.resize(overlay(a,r),(1229,900)))[1].tofile(str(out/f'{i:03d}.jpg'))
        cv2.imencode('.png',r['surface_mask'])[1].tofile(str(out/f'{i:03d}_mask.png'))
        print(i,{k:r['record'].get(k) for k in ['valid','reasons','edge_y','top_support_fraction','top_largest_observation_gap_px','local_residual_MAD_px','surface_area_px']},round(time.perf_counter()-t,2),flush=True)
