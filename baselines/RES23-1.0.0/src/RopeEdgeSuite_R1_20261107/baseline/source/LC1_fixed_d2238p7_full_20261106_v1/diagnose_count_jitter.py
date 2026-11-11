"""Read-only diagnostic plot of the completed trial, with no inference or fitting."""
from pathlib import Path
import csv,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'analysis_anchored/measurements.csv').open(encoding='utf-8-sig')))
r=[v for v in rows if 121<=int(v['frame_index'])<=241]
t=np.array([float(v['source_nominal_time_s']) for v in r])
valid=np.array([v['F3_count_valid']=='True' for v in r])
n=np.array([float(v['F3_continuous_N']) if v['F3_continuous_N'] else np.nan for v in r])
active=np.array([float(v['intersection_y_px']) for v in r]);edge=np.array([float(v['edge_y_px']) for v in r])
pitch=np.array([float(v['local_pitch_px']) for v in r])
im=Image.new('RGB',(1500,1190),'white');draw=ImageDraw.Draw(im)
font=lambda s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s)
draw.text((65,20),'接近阶段的跳动来源：S1 交点与 F3 计量尺度均有变化',font=font(32),fill='#222222')
draw.text((65,72),'固定 d=2238.7；缺测不补值；本图无人工真值、未做平滑拟合',font=font(23),fill='#555555')
blue='#1967a5';orange='#d15a26';green='#218354'
def panel(top,title,limits,ticks,series):
    left,right,bottom=120,1440,top+240
    draw.text((left,top-42),title,font=font(24),fill='#222222')
    def xy(xx,yy):return (left+(float(xx)-t[0])/(t[-1]-t[0])*(right-left),bottom-(float(yy)-limits[0])/(limits[1]-limits[0])*(bottom-top))
    for val in ticks:
        y=xy(t[0],val)[1];draw.line((left,y,right,y),fill='#e5e5e5',width=1)
        draw.text((12,y-13),f'{val:g}',font=font(20),fill='#555555')
    for val in (12.1,14,16.2,18,20,22,24):
        x=xy(val,limits[0])[0];draw.line((x,top,x,bottom),fill='#eeeeee',width=1)
        draw.text((x-20,bottom+8),f'{val:g}',font=font(19),fill='#555555')
    draw.line((left,top,left,bottom,right,bottom),fill='#555555',width=2)
    for values,color,dots,label in series:
        last=None
        for xx,yy in zip(t,values):
            if not np.isfinite(yy):last=None;continue
            p=xy(xx,yy)
            if last is not None:draw.line((last,p),fill=color,width=2)
            if dots:draw.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=color)
            last=p
    lx=800
    for values,color,dots,label in series:
        draw.line((lx,top-25,lx+30,top-25),fill=color,width=3)
        draw.text((lx+38,top-40),label,font=font(20),fill=color);lx+=300
panel(170,'剩余绳距 N（通过有效性检查的当前读数）',(0,6),range(7),[(n,blue,True,'F3 观测')])
lo=np.floor(min(active.min(),edge.min())/20)*20-20;hi=np.ceil(max(active.max(),edge.max())/20)*20+20
panel(520,'原图纵坐标 y / 像素',(lo,hi),np.arange(lo,hi+1,20),[(active,blue,False,'S1 交点'),(edge,orange,False,'边界位置')])
panel(870,'F3 局部绳距 / 像素',(14,38),range(14,39,4),[(pitch,'#bbbbbb',False,'所有帧'),(np.where(valid,pitch,np.nan),green,True,'计数通过帧')])
draw.text((660,1150),'视频回放时间 / 秒',font=font(23),fill='#444444')
path=ROOT/'count_jitter_diagnostic.png';im.save(path)
print(str(path))
