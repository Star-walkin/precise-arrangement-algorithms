import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2,numpy as np
from PIL import Image,ImageDraw
from common import *
from tracker_sg4 import line_angle_delta,line_y
from render import font,polyline,pic
OLD=PARENT/'RopeEdgeSuite_R15_20261109';STRICT=PARENT/'RopeEdgeSuite_Batch_RES14_20261109'
def main():
    cv2.setNumThreads(1);allrows=[];selected=[]
    for item in read(ROOT/'plan.json')['videos']:
        id=item['id'];rows=[]
        for p in sorted((OLD/'runs'/id/'frames').glob('*.json.gz')):
            i=int(p.name.split('.')[0]);a=record(STRICT/'runs'/id,i)['track'];b=record(OLD/'runs'/id,i)['track']
            if a['valid'] and b['valid']:
                delta=abs(line_y(a['line_abc'],1600)-line_y(b['line_abc'],1600));da=line_angle_delta(a['line_abc'],b['line_abc'])
                if delta>8 or da>1:rows.append(dict(id=id,frame=i,time_s=i/item['fps'],delta_y1600=delta,angle_delta=da))
        allrows+=rows
        picks=[]
        for r in sorted(rows,key=lambda r:-r['delta_y1600']):
            if all(abs(r['frame']-p['frame'])>=30 for p in picks):picks.append(r)
            if len(picks)==2:break
        for r in picks:
            i=r['frame'];cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,raw=cap.read();cap.release();assert ok
            im=cv2.flip(raw,1) if item['camera']=='right' else raw
            out=Image.new('RGB',(1200,1070),'#111b25');d=ImageDraw.Draw(out)
            d.text((15,8),f"{id} {i/10:.1f}s | SG2 / SG3 difference {r['delta_y1600']:.1f}px",font=font(24),fill='white')
            for k,(label,track) in enumerate([('Current raw image',None),('Strict SG2 current support',record(STRICT/'runs'/id,i)['track']),('Relaxed SG3 current support',record(OLD/'runs'/id,i)['track'])]):
                ann=im.copy()
                if track:
                    for pts in track['supported_segments_xy']:polyline(ann,pts,(60,255,60),5)
                d.text((15,46+k*340),label,font=font(20),fill='white')
                out.paste(pic(ann[1650:2300,0:2400],(1200,325)),(0,75+k*330))
            path=ROOT/'review'/f'{id}__branch_{i:06d}.jpg';path.parent.mkdir(exist_ok=True);out.save(path,quality=94)
            r['image']=str(path);selected.append(r)
    dump(ROOT/'branch_disagreements.json',dict(rows=allrows,review_selected=selected))
    print('Disagreements:',len(allrows),'review images:',len(selected),flush=True)
if __name__=='__main__':main()
