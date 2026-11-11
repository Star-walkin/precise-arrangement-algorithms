import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from inputs import ROOT,SOURCE

def draw_models(image,curves,scale=.5):
    a=image.copy()
    for c in curves:
        xy=np.rint(np.asarray(c['xy'])/scale).astype(np.int32)
        states=np.asarray(c['point_states'])=='observed';edge=states[:-1]&states[1:]
        starts=np.r_[0,np.flatnonzero(edge[1:]!=edge[:-1])+1];ends=np.r_[starts[1:],len(edge)]
        for lo,hi in zip(starts,ends):cv2.polylines(a,[xy[lo:hi+1]],False,(45,245,45) if edge[lo] else (0,150,255),2)
    return a

def comparison(image,r,points,out):
    classified=image.copy();labels=np.asarray(r['diagnostics']['point_membership'])
    palette=[(255,100,80),(30,220,255),(255,80,255),(80,255,110),(255,220,50),(80,120,255)]
    for p,k in zip(points,labels):
        if k>=0:cv2.circle(classified,tuple(np.rint(p[1:3]*2).astype(int)),2,palette[k%len(palette)],-1)
    fit=draw_models(image,r['curves'])
    for a in (classified,fit):cv2.line(a,(2239,2000),(2239,2999),(0,255,255),2)
    x0,y0,x1,y1=1750,2260,2470,2780;w=x1-x0;h=y1-y0
    canvas=Image.new('RGB',(w*3+40,h+160),'#f1f4f8');dr=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    for k,(a,title) in enumerate(zip((image,classified,fit),('原图','原始黄点的全局归类（不同类分色）','每类一条连续模型：绿=图像支持 / 橙=补全'))):
        xx=10+k*(w+10);dr.text((xx,16),title,font=small,fill='#172435')
        canvas.paste(Image.fromarray(cv2.cvtColor(a[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)),(xx,60))
    dg=r['diagnostics']
    dr.text((20,h+77),f"BG3：{dg['raw_point_count']} 个原始点，{dg['classified_point_count']} 点归入 {dg['class_count']} 类；每类只产生一条曲线，内部不按缺点拆段。",font=font,fill='#172435')
    dr.text((20,h+116),'橙线是模型插值，不能自动作为可靠计数证据；仅在有效黄点跨度内建模，未向两端无限外推。',font=small,fill='#172435')
    canvas.save(out)

if __name__=='__main__':
    r=json.loads((ROOT/'pilot240_geometry.json').read_text(encoding='utf8'))
    points=np.load(ROOT/'cache/000240.npz')['points']
    cap=cv2.VideoCapture(str(SOURCE));cap.set(1,240);ok,a=cap.read();cap.release();assert ok
    out=ROOT/'delivery';out.mkdir(exist_ok=True);comparison(a,r,points,out/'pilot240_point_classes.png')
