import cv2
from common import *
from tracker_sg4 import Fusion,anchor_of
from test_sg4 import get
def main():
    cv2.setNumThreads(1);rows=[]
    cases={'dual_20260923_113305__right':[1,150,250,400,450,499,540,580],
        'dual_20260923_113745__left':[20,150,300,370,397,400,500],
        'dual_20260923_101517__right':[1,93,248,400,580],
        'dual_20260924_111143__right':[1,150,300,382,514,580],
        'dual_20260923_101643__right':[1,150,300,411,447,480]}
    for id,ids in cases.items():
        for i in ids:
            im,s,r=get(id,i)
            if not s['valid']:continue
            f=Fusion();f.anchor_x=anchor_of(s);a=f.verify_fallback(im,s,i/10)
            f.last_strict=None;b=f.verify_fallback(im,r,i/10) if r['valid'] else None
            row=dict(id=id,frame=i,strict_check=a,relaxed_check=b);rows.append(row)
            print(id,i,'x',f.anchor_x,'strict',a['valid'],a['reasons'],'relaxed',b['valid'] if b else None,b['reasons'] if b else [],flush=True)
    dump(ROOT/'strict_gate_probe.json',rows)
if __name__=='__main__':main()
