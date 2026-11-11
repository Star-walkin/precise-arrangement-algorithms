"""Inspect remaining strong-reflection color-component failures; no mutation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2,numpy as np,json
from diagnose import ROOT,OLD,read

def main():
    cv2.setNumThreads(1);run=OLD/'transfer_runs/dual_20260923_113745';rows=read(run/'compact.json')
    ids=[x['frame'] for x in rows if x['region_reasons']==['insufficient_yellow_flange']]
    cap=cv2.VideoCapture(read(run/'binding.json')['source']);results=[]
    for i in ids:
        cap.set(1,i);ok,im=cap.read();assert ok
        sm=cv2.resize(im,None,fx=.25,fy=.25,interpolation=cv2.INTER_AREA);h,w=sm.shape[:2]
        b,g,r=cv2.split(sm.astype(np.float32));light=(b+g+r)/3;ch=((r+g)/2-b)/np.maximum(light,12)
        m=((ch>.45)&(g>.72*r)&(light>18)).astype(np.uint8)*255;m[:int(.5*h)]=0;m[:,int(.9*w):]=0
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        n,lab,stats,cent=cv2.connectedComponentsWithStats(m,8)
        kk=sorted(range(1,n),key=lambda k:stats[k,4],reverse=True)[:8]
        entry=dict(frame=i,minimum_area=.018*h*w,total_area=int((m>0).sum()),components=[dict(area=int(stats[k,4]),bbox=stats[k,:4].tolist(),center=cent[k].tolist()) for k in kk])
        results.append(entry)
    cap.release();(ROOT/'color_diagnostic.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    for r in results:print(r['frame'],'min',r['minimum_area'],'top areas',[x['area'] for x in r['components'][:4]],flush=True)

if __name__=='__main__':main()
