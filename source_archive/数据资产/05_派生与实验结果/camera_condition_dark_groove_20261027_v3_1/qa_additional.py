"""冻结后固定额外12图：输入增强、旧D2、新D3原坐标对照，不挑结果。"""
from pathlib import Path
import sys
import cv2
import numpy as np
from data_io import HERE,D1,D2,old,read_image,write_image,dump,sha

run=HERE/sys.argv[1]
rows,qa=old.old.old.select('all')
chosen=[]
for camera in ['left','right']:
    names=sorted(r['output_file'] for r in rows if r['camera_role']==camera and r['output_file'] in qa)
    chosen.extend(names[i] for i in [1,5,8,11,15,17])
out=run/'记录/qa_additional'
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
entries=[]
for rel in chosen:
    p=Path(rel);role=p.parent.name
    mask=read_image(D1/'all_001/记录/masks/dark_roi'/role/p.name,0)
    images=[read_image(D1/'all_001/图像结果/02_对比度增强ROI'/role/p.name),
            read_image(D2/'all_002/图像结果'/role/p.name),read_image(run/'图像结果'/role/p.name)]
    n,_,stats,_=cv2.connectedComponentsWithStats(mask)
    for k,component in enumerate(sorted(range(1,n),key=lambda i:-stats[i,4])):
        x,y,w,h,area=map(int,stats[component])
        if area<10000:continue
        factor=min(1.,800/max(w,h));ww,hh=max(1,round(w*factor)),max(1,round(h*factor))
        panels=[]
        for im,title in zip(images,['D1 fixed enhancement','D2.1.1','D3 frozen']):
            tile=np.zeros((hh+30,ww,3),np.uint8)
            tile[30:]=cv2.resize(im[y:y+h,x:x+w],(ww,hh),interpolation=cv2.INTER_AREA)
            cv2.putText(tile,title,(4,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            panels.append(tile)
        target=out/(p.stem+f'__part{k}.jpg')
        write_image(target,np.concatenate(panels,axis=1))
        entries.append({'output_file':rel,'bbox_xyxy':[x,y,x+w,y+h],'comparison':str(target),'sha256':sha(target)})
dump(out/'index.json',{'selection':chosen,'rule':'each camera sorted QA19 indices 1,5,8,11,15,17; before full run; no further tuning','entries':entries})
print({'selected':len(chosen),'panels':len(entries),'directory':str(out)})
