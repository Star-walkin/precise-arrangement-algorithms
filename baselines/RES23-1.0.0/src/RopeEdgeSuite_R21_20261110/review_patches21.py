import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from common21 import *
import cv2
from PIL import Image,ImageDraw,ImageFont
cv2.setNumThreads(1);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
for id,indices in [('dual_20260924_111143__right',[30,60,90,120]),('dual_20260924_111500__left',[240,245,247,249,255,260,270,280,290])]:
    item=next(v for v in plan() if v['id']==id);cap=cv2.VideoCapture(item['source']);panels=[]
    for i in indices:
        path=ROOT/'measurements'/id/'frames'/f'{i:06d}.json.gz';row=gzread(path) if path.exists() else count20(id,i);m=row['measurement'];c=m.get('central') or {}
        if not c:continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,raw=cap.read();assert ok
        if item['camera']=='right':raw=cv2.flip(raw,1)
        d=m['d'];by=c['boundary']['y'];ay=c['active_y'];pitch=c['local_pitch_px']
        xa=max(0,int(d-220));xb=min(4096,int(d+300));ya=max(0,int(min(by,ay)-80));yb=min(3000,int(max(by,ay)+120))
        crop=raw[ya:yb,xa:xb];ann=crop.copy();cv2.line(ann,(round(d-xa),0),(round(d-xa),len(ann)-1),(255,0,255),1)
        rr=frontend(id,i);b=rr['boundary'];xx=np.arange(xa,xb);yy=np.polynomial.polynomial.polyval((xx-b['normalized_x']['center'])/b['normalized_x']['scale'],b['fit_coefficients_ascending'])
        cv2.polylines(ann,[np.rint(np.c_[xx-xa,yy-ya]).astype('int32')],False,(255,200,0),1)
        for cr in c['crossings']:
            y=cr['y'];
            if ya<=y<yb:cv2.circle(ann,(round(d-xa),round(y-ya)),3,(0,255,0),-1)
        cv2.circle(ann,(round(d-xa),round(ay-ya)),5,(0,0,255),2)
        pic=Image.new('RGB',(1040,330),'#121e25');draw=ImageDraw.Draw(pic)
        draw.text((5,4),f"{i/10:.1f}s N={m.get('N'):.2f} mode={m.get('measurement_mode','R20')} pitch={pitch:.1f}px",font=font,fill='white')
        for j,img in enumerate([crop,ann]):
            pi=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));pi.thumbnail((515,290));pic.paste(pi,(j*520,35))
        pic.save(ROOT/'inspection'/f'{id}_{i:06d}_detail.jpg');panels.append(pic)
        print(id,i,'count',c['count_observed_grooves'],'boundary',by,'active',ay,'nearest_crossings',[round(q['y']-by,2) for q in c['crossings'] if by-20<q['y']<by+100],flush=True)
    cap.release()
    for k in range(0,len(panels),3):
        sheet=Image.new('RGB',(1040,330*len(panels[k:k+3])))
        for j,panel in enumerate(panels[k:k+3]):sheet.paste(panel,(0,j*330))
        sheet.save(ROOT/'inspection'/f'{id}_detail_sheet{k//3}.jpg')
