import cv2,numpy as np,subprocess
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from common21 import *

FONTS={}
def font(n):
    if n not in FONTS:FONTS[n]=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    return FONTS[n]

def line(im,xy,color,width=3,dashed=False):
    p=np.rint(xy).astype(np.int32)
    if len(p)<2:return
    if not dashed:cv2.polylines(im,[p],False,color,width,cv2.LINE_AA);return
    for k in range(0,len(p)-1,8):cv2.polylines(im,[p[k:min(k+5,len(p))]],False,color,width,cv2.LINE_AA)

CONTACT={'unavailable':'贴边证据不足','separated':'与边缘分离','near_edge_approaching':'接近边缘',
    'near_edge_uncertain':'近边，遮挡或误差范围内无法确认','projection_conflict':'投影与边界模型不一致',
    'visual_contact_candidate':'视觉贴边候选（实体接触未确认）'}

def annotate(im,r,row,groups,data):
    ann=im.copy();m=row['measurement'];central=m.get('central') or {};d=m.get('d');axis=row.get('axis');b=r['boundary']
    curves=central.get('curves') or (groups or {}).get('curves',[])
    for c in curves:
        xy=np.asarray(c['xy']);states=np.asarray(c.get('point_states',[]))
        if len(states)==len(xy):
            for observed in (True,False):
                ix=np.flatnonzero((states=='observed')==observed)
                for indices in np.split(ix,np.flatnonzero(np.diff(ix)>1)+1):
                    if len(indices)>1:line(ann,xy[indices],(50,155,50) if observed else (75,130,75),2,not observed)
        else:line(ann,xy,(50,155,50),2)
    if b.get('fit_valid'):
        model_top=data.get('analysis_top_curve_xy') if data is not None else None
        if model_top is None:model_top=np.asarray(b.get('top_curve_xy',[]))
        if len(model_top)>1:
            xx=np.arange(model_top[0,0],model_top[-1,0]+1,5.);n=b['normalized_x'];yy=np.polynomial.polynomial.polyval((xx-n['center'])/n['scale'],b['fit_coefficients_ascending'])
            line(ann,np.c_[xx,yy],(255,160,50),3,True)
        for segment in b.get('observed_segments',[]):line(ann,segment,(255,240,20),4)
    for segment in r['track'].get('supported_segments_xy',[]):line(ann,segment,(0,220,255),3)
    if axis is not None:
        pts=np.asarray(axis['points_xy'],float)
        if len(pts):
            for xy in pts[::3]:cv2.circle(ann,tuple(np.rint(xy).astype(int)),2,(0,220,255),-1)
            abc=axis['line_abc'];xx=np.arange(pts[:,0].min(),max(pts[:,0].max(),d or 0)+1,6.);yy=-(abc[0]*xx+abc[2])/abc[1]
            line(ann,np.c_[xx,yy],(0,190,255),2,True)
    if d is not None:
        interval=m.get('d_interval') or [d,d];lo,hi=[int(np.clip(round(x),0,ann.shape[1]-1)) for x in interval]
        tint=ann.copy();cv2.rectangle(tint,(lo,1650),(hi,2999),(120,35,100),-1);ann=cv2.addWeighted(ann,.88,tint,.12,0)
        dynamic=m.get('measurement_mode')=='current_dynamic_column'
        if dynamic:
            for yy in range(1650,2999,28):cv2.line(ann,(round(d),yy),(round(d),yy+12),(240,240,240),3)
        else:cv2.line(ann,(round(d),1650),(round(d),2999),(230,60,220),3)
        if data is not None:
            points=data['points'][:,1:3]*2
            for xy in points[abs(points[:,0]-d)<=120][::2]:cv2.circle(ann,tuple(np.rint(xy).astype(int)),2,(0,220,255),-1)
        for c in central.get('crossings',[]):cv2.circle(ann,(round(d),round(c['y'])),6,(0,255,255),-1)
        if central.get('active_y') is not None:cv2.circle(ann,(round(d),round(central['active_y'])),8,(0,150,255),3)
    local_d=row.get('contact',{}).get('diagnostic_column')
    if d is None and local_d is not None:
        for yy in range(1650,2999,28):cv2.line(ann,(round(local_d),yy),(round(local_d),yy+12),(240,240,240),2)
    return ann

def pic(im,size):return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize(size,Image.Resampling.LANCZOS)

def panel(raw,r,row,item,cal,groups,data):
    mirror=item['camera']=='right';im=cv2.flip(raw,1) if mirror else raw;ann=annotate(im,r,row,groups,data);ann=cv2.flip(ann,1) if mirror else ann
    m=row['measurement'];central=m.get('central') or {};d=m.get('d');f=row['filtered'];axis=row.get('axis');phase=next(x for x in cal['epochs'] if x['start']<=row['frame']<x['end'])
    center=float(d if d is not None else 2100);xa=int(np.clip(center-1200,0,4096-2400));xb=xa+2400
    if mirror:xa,xb=4096-xb,4096-xa
    yy=central.get('active_y')
    by=(central.get('boundary') or {}).get('y')
    cy=(yy+by)/2 if yy is not None and by is not None and 0<=yy<3000 else 2360
    ya=int(np.clip(cy-390,1650,2220));yb=ya+780
    canvas=Image.new('RGB',(1920,1080),'#111b25');dr=ImageDraw.Draw(canvas)
    dr.text((24,16),'RopeEdgeSuite R21 · F3 / 固定与动态 d / 连续近边事件',font=font(30),fill='white')
    dr.text((1450,22),f"{row['time_s']:05.1f}s  {row['frame']+1}/{item['planned_frames']}",font=font(27),fill='white')
    label='右相机 · 已还原原图朝向' if mirror else '左相机 · 原图朝向'
    dr.text((24,66),item['session']+' / '+label,font=font(22),fill='#b8cad7')
    if item.get('partial_source'):dr.text((1140,68),f"源视频截断，仅 {item['planned_frames']} 帧",font=font(21),fill='#edbd6c')
    canvas.paste(pic(ann,(800,586)),(24,120))
    dr.text((852,110),'原图局部',font=font(22),fill='white');canvas.paste(pic(raw[ya:yb,xa:xb],(1040,338)),(852,144))
    dr.text((852,495),'同帧叠加：黄点 / F3 曲线 / 活动绳 / d',font=font(22),fill='white');canvas.paste(pic(ann[ya:yb,xa:xb],(1040,338)),(852,529))
    if not f['available']:main='剩余根数：当前无法可靠估计'
    elif f['interval'][0]>5:main='距离边缘：大于 5 圈'
    elif f['N']<0:main='近边：活动绳投影与边界估计重叠'
    else:main=f"剩余距离估计：约 {f['N']:.1f} 圈"
    if f['available'] and f.get('source')=='short_prediction_max_0_5s':main=f"短时推算（{f['age_s']:.1f}s）：约 {f['N']:.1f} 圈"
    dr.text((24,728),main,font=font(28),fill='#edbd6c' if not f['available'] or f['N']<0 else '#7edeb6')
    if f['available']:
        dr.text((24,770),f"模型范围 [{f['interval'][0]:.1f}, {f['interval'][1]:.1f}] 圈；非统计置信区间",font=font(20),fill='#bacbd9')
    else:
        reason='标定参考不足或入卷点不可见' if d is None else '当前图像证据不足'
        dr.text((24,770),'原因：'+reason,font=font(21),fill='#edbd6c')
    if d is not None:
        dd=4095-d if mirror else d
        dynamic=m.get('measurement_mode')=='current_dynamic_column'
        lo,hi=sorted([4095-x for x in m['d_interval']]) if mirror else m['d_interval']
        dr.text((24,808),f'd（原图）={dd:.1f}px  范围 [{lo:.0f}, {hi:.0f}]',font=font(21),fill='#d997d2')
        dlabel='当前动态列；固定 d 未锁定，数值为条件几何估计' if dynamic else ('完整弯折标定' if phase['status']=='observed_bend_calibrated' else '固定列条件估计；参考帧满足 ≥5 圈')
        dr.text((24,839),'d：'+dlabel,font=font(18),fill='#bacbd9')
    else:dr.text((24,812),'d：该运动阶段未能独立标定',font=font(22),fill='#edbd6c')
    abc=axis['line_abc'] if axis else r['track'].get('line_abc')
    if abc is not None:
        a,b,c=abc
        if mirror:a,c=-a,c+a*4095
        dr.text((24,879),f'{a:+.5f} x {b:+.5f} y {c:+.2f} = 0',font=font(20),fill='#f1cd72')
    event=row.get('near_event') or {};names={'unknown':'近边事件：当前证据不足','near':'近边停留（视觉）','departing':'正在离开边缘（持续回升）','separated':'远离边缘','approaching':'接近边缘'}
    elabel=names.get(event.get('state'),'近边事件待判定')
    if event.get('active_event_start_s') is not None:elabel+=f"，起点 {event['active_event_start_s']:.1f}s"
    dr.text((852,882),elabel,font=font(24),fill='#edbd6c')
    obs=central.get('count_observed_grooves');complete=m.get('complete_F3_phase')
    detail=f"计数区间内观测绳沟 {obs} 条；"+('F3 相位链完整' if complete else 'F3 相位链有缺口，数值依赖几何估计') if obs is not None else '未输出计数值；仍显示当前 F3 检测与分组'
    if row['contact'].get('source')=='current_local_tangency_without_locked_d':detail='当前帧局部近边观测；白虚线为诊断列，固定 d 尚未标定'
    dr.text((852,918),CONTACT[row['contact']['state']]+'；'+detail,font=font(16),fill='#bacbd9')
    dr.text((24,952),'青：可见边界　蓝虚线：边界模型　绿：F3 绳沟　黄：活动绳　紫：固定 d　白虚线：当前动态列',font=font(21),fill='#bacbd9')
    dr.rectangle((24,995,1892,1008),fill='#243947');dr.rectangle((24,995,24+int(1868*(row['frame']+1)/item['planned_frames']),1008),fill='#427f94')
    dr.text((24,1030),'距离单位：活动绳中心线到可见卷面边界的当前绳节距。视觉贴边候选不等同于实体接触确认。',font=font(20),fill='#bacbd9')
    return np.asarray(canvas)

def writer(path,fps):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],
        stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
