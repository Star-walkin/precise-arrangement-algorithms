"""Review BG5 current-frame evidence before rerunning the complete detector."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,time,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from region_bg5 import segment
from diagnose import ROOT,OLD,read

def review(im,new,old,title,path):
    a=im.copy();reg=new['record'];mask=new['surface_mask']>0
    a[mask]=(a[mask]*.82+np.array([40,130,30])*.18).astype(np.uint8)
    for xy in reg.get('top_curve_supported_segments',[]):
        cv2.polylines(a,[np.rint(xy).astype(np.int32)],False,(255,220,40),4,cv2.LINE_AA)
    for x,y in reg.get('top_observed_xy',[]):cv2.circle(a,(round(x),round(y)),3,(0,240,255),-1)
    for x,y in reg.get('outer_rim',{}).get('points_xy',[]):cv2.circle(a,(round(x),round(y)),4,(240,80,220),-1)
    cv2.line(a,(2239,1700),(2239,2999),(230,80,250),3)
    pic=Image.new('RGB',(1800,1050),'#14202a');draw=ImageDraw.Draw(pic);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    draw.text((20,10),title,fill='white',font=font)
    for j,img in enumerate([im,a]):
        crop=img[1750:3000,300:3300]
        pic.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((1800,750)),(0,60)) if j==1 else None
    # An unannotated inset allows direct image/curve comparison at the count origin.
    crop=im[1990:2390,1900:2550];pic.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((650,400)),(1140,620))
    lines=[f"旧版卷面: {old['valid']} / {', '.join(old['reasons'])}",
           f"BG5 卷面: {reg['valid']} / {', '.join(reg['reasons'])}",
           f"外缘在 d 处: {reg.get('outer_rim_status')}；物体确认: {reg.get('flange_presence',{}).get('valid')}",
           f"d 附近支撑: {reg.get('local_support_fraction')}；残差 MAD: {reg.get('local_residual_MAD_px')}"]
    for j,s in enumerate(lines):draw.text((15,825+j*45),s,fill='white',font=font)
    pic.save(path,quality=94)

def main():
    cv2.setNumThreads(1);out=ROOT/'pilot';out.mkdir(exist_ok=True);rows=[]
    for session,ids in [('dual_20260923_113745',[0,20,35,190,240,280,286,316,336,526,599]),('dual_20260924_111143',[0,300,600])]:
        run=OLD/'transfer_runs'/session;b=read(run/'binding.json');cap=cv2.VideoCapture(b['source'])
        for i in ids:
            cap.set(1,i);ok,im=cap.read();assert ok
            old=read(run/'bg1/frames'/f'{i:06d}.json');assert hashlib.sha256(im.tobytes()).hexdigest()==old['source_pixel_sha256']
            tick=time.perf_counter();new=segment(im);g=new['record']
            row=dict(session=session,frame=i,old=old['region']['valid'],valid=g['valid'],reasons=g['reasons'],presence=g.get('flange_presence'),
                     edge_old=old['region'].get('edge_y'),edge_new=g.get('edge_y'),elapsed_s=time.perf_counter()-tick)
            rows.append(row);print(json.dumps(row),flush=True)
            review(im,new,old['region'],f'{session} / {i/10:.1f}s',out/f'{session}_{i:03d}.jpg')
            (out/f'{session}_{i:03d}.json').write_text(json.dumps(g,ensure_ascii=False),encoding='utf8')
        cap.release()
    (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf8')

if __name__=='__main__':main()
