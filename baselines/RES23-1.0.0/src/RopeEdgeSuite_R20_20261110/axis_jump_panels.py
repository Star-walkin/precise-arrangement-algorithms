import cv2
from PIL import Image,ImageDraw
from common20 import *
from render20 import font,line

def main():
    out=ROOT/'jump_review';out.mkdir(exist_ok=True)
    for id,indices in [('dual_20260923_101517__right',[246,247]),('dual_20260923_101643__right',[358,359]),('dual_20260924_111143__right',[356,357,358])]:
        item=next(x for x in plan() if x['id']==id);panels=[]
        for i in indices:
            cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,im=cap.read();cap.release();assert ok;r=record(id,i);assert pixelsha(im)==r['source_pixel_sha256'];im=cv2.flip(im,1)
            with gzip.open(ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:g=json.load(f)
            a=g['axis'];ann=im.copy();abc=a['line_abc'];xs=np.arange(900,2800,4);ys=-(abc[0]*xs+abc[2])/abc[1]
            line(ann,np.c_[xs,ys],(0,150,255),3)
            for p in a['points_xy']:cv2.circle(ann,tuple(np.rint(p).astype(int)),2,(0,255,255),-1)
            for q in r['track'].get('supported_segments_xy',[]):line(ann,q,(255,255,20),2)
            yy=-(abc[0]*2500+abc[2])/abc[1];ya=int(np.clip(yy-430,1400,2140));yb=ya+860;xa,xb=1100,2900
            canvas=Image.new('RGB',(1000,1040),'#111b25');dr=ImageDraw.Draw(canvas)
            dr.text((10,5),f'{id} / {i}',font=font(23),fill='white');dr.text((10,38),f'y(2500)={yy:.1f}; 当前轴橙，支持点黄，SG5青',font=font(21),fill='white')
            for j,v in enumerate([im,ann]):canvas.paste(Image.fromarray(cv2.cvtColor(v[ya:yb,xa:xb],cv2.COLOR_BGR2RGB)).resize((1000,478)),(0,80+480*j))
            panels.append(canvas)
        combined=Image.new('RGB',(1000*len(panels),1040),'black')
        for j,v in enumerate(panels):combined.paste(v,(j*1000,0))
        combined.save(out/f'{id}.jpg',quality=96);print(id,flush=True)

if __name__=='__main__':main()
