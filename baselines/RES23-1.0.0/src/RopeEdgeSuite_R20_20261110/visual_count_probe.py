import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,cv2
from PIL import Image,ImageDraw
from common20 import *
from count20 import measure
from render20 import font

def one(id,i,d):
    item=next(x for x in plan() if x['id']==id);r=record(id,i)
    fr=read(ROOT/'f3_v2'/id/'records'/f'{i:06d}.json')
    with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
    with gzip.open(ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:g=json.load(f)
    with gzip.open(ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:geo=json.load(f)
    cal=dict(status='diagnostic_column_not_final_calibration',d=d,interval=[d-1,d+1]);m=measure(g['groups'],data,r['boundary'],geo['axis'],cal)
    cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,im=cap.read();cap.release();assert ok and pixelsha(im)==r['source_pixel_sha256']
    if item['camera']=='right':im=cv2.flip(im,1)
    c=m['central'];y0=c['boundary']['y'];y1=c['active_y'];xa=int(d)-240;xb=int(d)+240;ya=max(0,int(min(y0,y1))-60);yb=min(3000,int(max(y0,y1))+100)
    raw=im[ya:yb,xa:xb];ann=raw.copy();x=int(d)-xa
    cv2.line(ann,(x,0),(x,len(ann)-1),(230,50,230),1)
    for y,col in [(y0,(255,200,20)),(y1,(0,180,255))]:cv2.line(ann,(0,round(y)-ya),(len(ann[0])-1,round(y)-ya),col,2)
    accepted=[q for q in c['crossings'] if y0<q['y']<y1]
    for j,q in enumerate(accepted,1):
        y=round(q['y'])-ya;cv2.circle(ann,(x,y),3,(0,255,255),-1);cv2.putText(ann,str(j),(x+15,y+4),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1,cv2.LINE_AA)
    scale=min(2.,800/max(1,len(raw)));size=(round(raw.shape[1]*scale),round(raw.shape[0]*scale));canvas=Image.new('RGB',(size[0]*2,100+size[1]),'#111b25');dr=ImageDraw.Draw(canvas)
    dr.text((10,8),id+f'  frame {i}',font=font(22),fill='white')
    dr.text((10,40),f'诊断列 d={d:.0f}，非最终标定；观测绳沟 {len(accepted)}；N={m.get("N")}',font=font(20),fill='white')
    for k,p in enumerate([raw,ann]):canvas.paste(Image.fromarray(cv2.cvtColor(p,cv2.COLOR_BGR2RGB)).resize(size),(k*size[0],100))
    out=ROOT/'visual_count_probes';out.mkdir(exist_ok=True);dest=out/f'{id}_{i:06d}'
    canvas.save(str(dest)+'.jpg',quality=96);dump(str(dest)+'.json',m);print(dest,flush=True)

if __name__=='__main__':
    for id,i,d in [('dual_20260923_113745__left',50,2100),('dual_20260923_101517__right',140,2520),('dual_20260923_113305__right',80,2600),('dual_20260924_111143__right',80,2520)]:
        one(id,i,d)
