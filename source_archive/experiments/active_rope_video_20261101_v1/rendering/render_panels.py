"""Pure four-panel rendering, without detection, decisions or media-file I/O.

render_frame(bgr, heat, engine_result, temporal_result, meta) -> uint8 BGR.
Paths and scene tracks use original image coordinates. The renderer does not
accept a path, bridge gaps, fill an absent response, or stretch response values.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

PANEL_WIDTH=800
BODY_HEIGHT=586
HEADER_HEIGHT=54
PANEL_HEIGHT=BODY_HEIGHT+HEADER_HEIGHT
FONT_PATH='C:/Windows/Fonts/msyh.ttc'
GRAY=(150,150,150)
GREEN=(55,230,65)
ORANGE=(30,160,255)
YELLOW=(15,225,255)
CYAN=(245,205,40)
CANDIDATE_COLORS=[(15,225,255),(220,80,240),(245,210,30)]

@lru_cache(maxsize=8)
def _font(size):
    if Path(FONT_PATH).is_file():return ImageFont.truetype(FONT_PATH,size)
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)

def _text(canvas,text,xy,size=14,color=(245,245,245),stroke=0):
    # color is RGB at the Pillow boundary only.
    im=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im)
    draw.text(xy,str(text),font=_font(size),fill=color,stroke_width=stroke,stroke_fill=(0,0,0))
    return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)

def _header(body,lines):
    top=Image.new('RGB',(PANEL_WIDTH,HEADER_HEIGHT),(245,247,250));draw=ImageDraw.Draw(top)
    for line,y,size in zip(lines,[1,22,38],[17,13,12]):
        draw.text((7,y),str(line),font=_font(size),fill=(25,30,38))
    return np.concatenate([cv2.cvtColor(np.asarray(top),cv2.COLOR_RGB2BGR),body],axis=0)

def _geometry(bgr):
    h,w=bgr.shape[:2];scale=min(PANEL_WIDTH/w,BODY_HEIGHT/h)
    rw,rh=round(w*scale),round(h*scale)
    return dict(scale=scale,offset=np.asarray([(PANEL_WIDTH-rw)//2,(BODY_HEIGHT-rh)//2],float),size=(rw,rh),original_shape=(h,w))

def _body(bgr,geo,dim=1.):
    canvas=np.zeros((BODY_HEIGHT,PANEL_WIDTH,3),np.uint8)
    small=cv2.resize(bgr,geo['size'],interpolation=cv2.INTER_AREA)
    if dim!=1.:small=np.rint(small.astype(np.float32)*dim).astype(np.uint8)
    x,y=geo['offset'].astype(int);w,h=geo['size'];canvas[y:y+h,x:x+w]=small
    return canvas

def _points(value,geo):
    p=np.asarray(value if value is not None else [],dtype=np.float64)
    if p.size==0:return np.empty((0,2),np.float64)
    if p.ndim!=2 or p.shape[1]!=2:raise ValueError('Path must be N x2 original-image coordinates')
    return p*geo['scale']+geo['offset']

def _segments(canvas,p,color,width=2,mask=None,dashed=False):
    if len(p)<2:return
    good=np.isfinite(p[:-1]).all(axis=1)&np.isfinite(p[1:]).all(axis=1)
    if mask is not None:good&=np.asarray(mask,dtype=bool)
    if not np.any(good):return
    if dashed:
        # Fixed screen-space5px on/off even for sparse2-point paths. Never
        # bridge an invalid or unsupported source segment during resampling.
        delta=np.linalg.norm(np.diff(np.nan_to_num(p),axis=0),axis=1)
        distance=np.r_[0.,np.cumsum(delta)];pieces=[]
        for i in np.flatnonzero(good):
            start,end=distance[i],distance[i+1];length=end-start
            if length<1e-8:continue
            cuts=np.r_[start,np.arange((np.floor(start/5)+1)*5,end,5),end]
            for left,right in zip(cuts[:-1],cuts[1:]):
                if int(np.floor((left+right)*.5/5))%2:continue
                a=p[i]+(p[i+1]-p[i])*((left-start)/length)
                b=p[i]+(p[i+1]-p[i])*((right-start)/length)
                pieces.append([a,b])
        if not pieces:return
        paired=np.asarray(pieces)
    else:paired=np.stack([p[:-1][good],p[1:][good]],axis=1)
    polylines=np.rint(np.clip(paired,-1e6,1e6)).astype(np.int32)
    cv2.polylines(canvas,list(polylines),False,color,width,cv2.LINE_AA)

def _surface_points(track,engine,geo):
    p=np.asarray(track.get('points_xy',[]),dtype=float)
    coordinate=track.get('coordinate_system',engine.get('surface_coordinate_system','original_xy'))
    if p.size and coordinate=='crop_half_xy':
        bbox=engine.get('crop_bbox',[0,1050,4096,3000]);scale=float(engine.get('crop_scale',.5))
        p=p/scale+np.asarray(bbox[:2])
    elif coordinate not in ['original_xy','original_image_xy','original_pixels','original_image_pixels','crop_half_xy']:
        raise ValueError('Unrecognized surface coordinate system: '+str(coordinate))
    return _points(p,geo)

def _heat_body(heat,geo,meta):
    canvas=np.full((BODY_HEIGHT,PANEL_WIDTH,3),40,np.uint8)
    if heat is None:
        label=meta.get('heat_missing_reason','无入口，热图未计算')
        return _text(canvas,label,(120,BODY_HEIGHT//2-22),28)
    a=np.asarray(heat)
    if a.ndim!=2:raise ValueError('Heatmap must be single-channel2D')
    finite=np.isfinite(a)
    if not finite.any():return _text(canvas,'响应缺失，未显示伪零热图',(95,BODY_HEIGHT//2-22),26)
    h,w=geo['original_shape'];bbox=meta.get('heat_bbox_xyxy',[0,1050,w,h])
    x0,y0,x1,y1=map(float,bbox)
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Heat bbox outside original image')
    edges=np.rint(np.asarray([[x0,y0],[x1,y1]])*geo['scale']+geo['offset']).astype(int)
    (lx,ty),(rx,by)=edges;rw,rh=rx-lx,by-ty
    valid=cv2.resize(finite.astype(np.uint8),(rw,rh),interpolation=cv2.INTER_NEAREST).astype(bool)
    # Clip only for fixed-scale display. Input float array is never modified.
    small=cv2.resize(np.nan_to_num(a,nan=0.,posinf=1.,neginf=0.).astype(np.float32),(rw,rh),interpolation=cv2.INTER_LINEAR)
    pixels=np.rint(np.clip(small,0,1)*255).astype(np.uint8)
    colored=cv2.applyColorMap(pixels,cv2.COLORMAP_TURBO);colored[~valid]=40
    canvas[ty:by,lx:rx]=colored
    ramp=np.linspace(255,0,151).astype(np.uint8)[:,None].repeat(13,axis=1)
    canvas[12:163,PANEL_WIDTH-49:PANEL_WIDTH-36]=cv2.applyColorMap(ramp,cv2.COLORMAP_TURBO)
    for label,y in [('1.0',7),('0.5',80),('0.0',151)]:canvas=_text(canvas,label,(PANEL_WIDTH-33,y),12,stroke=1)
    return canvas

def _age(meta,key,default):
    value=meta.get(key,default)
    return None if value is None else max(0.,float(value))

def _number(value):return '—' if value is None else f'{float(value):.2f}s'

def _accepted_path(canvas,candidate,geo):
    p=_points(candidate.get('points_xy'),geo)
    if len(p)<2:return
    status=candidate.get('point_status')
    if status is None or len(status)!=len(p):
        _segments(canvas,p,GREEN,3);return
    s=np.asarray(status,dtype=str);left,right=s[:-1],s[1:]
    entry=(left=='entry_condition')&(right=='entry_condition')
    image_states=['image_supported','joint_heat_image_supported']
    img=np.isin(left,image_states)&np.isin(right,image_states)
    heat=np.isin(left,image_states+['heatmap_supported'])&np.isin(right,image_states+['heatmap_supported'])&~img
    a7=np.isin(left,image_states+['heatmap_supported','raw_a7_supported'])&np.isin(right,image_states+['heatmap_supported','raw_a7_supported'])&~img&~heat
    weak=~(entry|img|heat|a7)
    _segments(canvas,p,(245,245,245),2,entry)
    _segments(canvas,p,GREEN,3,img);_segments(canvas,p,YELLOW,3,heat);_segments(canvas,p,CYAN,3,a7)
    _segments(canvas,p,ORANGE,2,weak,True)

def render_frame(bgr,heat,engine_result,temporal_result,meta):
    """Return1600x1280 BGR, preserving raw paths and showing observation age.

    engine_result: raw_candidates<=3 with original points_xy; surface_tracks
      with original points_xy and optional inferred/current_image_supported.
    temporal_result: gate.update output. accepted_candidate_index selects the
      supplied raw_candidates exactly; reference is rendered only for STALE.
    meta: video_id,media_time_s,detection_time_s,source_frame_index,
      heat_age_s,detection_age_s,analysis_hz(2),playback_fps(5),heat_bbox_xyxy,
      optional display_status and display_reference_age_s. Caller must set
      display_status='UNKNOWN' if a cached reference expires between updates.
    """
    if not isinstance(bgr,np.ndarray) or bgr.dtype!=np.uint8 or bgr.ndim!=3 or bgr.shape[2]!=3:
        raise ValueError('bgr must be HxWx3 uint8')
    engine=engine_result or {};temporal=temporal_result or {};meta=meta or {}
    candidates=list(engine.get('raw_candidates',[]))
    if len(candidates)>3:raise ValueError('At most3 current candidates may be displayed')
    geo=_geometry(bgr);media=meta.get('media_time_s');det=meta.get('detection_time_s')
    elapsed=max(0.,float(media)-float(det)) if media is not None and det is not None else None
    heat_age=_age(meta,'heat_age_s',elapsed);detection_age=_age(meta,'detection_age_s',elapsed)
    video_id=meta.get('video_id','未提供视频ID');frame=meta.get('source_frame_index','—')
    hz=float(meta.get('analysis_hz',2));play=float(meta.get('playback_fps',5))
    original=_header(_body(bgr,geo),['1 原视频',video_id,
        f'媒体 t={_number(media)}  原帧={frame}  |  检测 {hz:g} Hz / 播放 {play:g} Hz'])
    hbody=_heat_body(heat,geo,meta)
    entry=meta.get('heatmap_entry_index');entry_label='—' if entry is None else str(int(entry)+1)
    hp=_header(hbody,[f'2 方案 B 原始响应（固定 0～1）| E{entry_label}',
        f'检测媒体 t={_number(det)}  热图观察年龄={_number(heat_age) if heat is not None else "未计算"}',
        '响应强度不是身份概率；灰色区域未计算；不逐帧拉伸'])
    candidate_body=_body(bgr,geo,.65)
    for track in engine.get('surface_tracks',[]):
        p=_surface_points(track,engine,geo)
        if len(p)<2:continue
        inferred=np.asarray(track.get('inferred',[False]*len(p)),bool)
        if inferred.shape!=(len(p),):raise ValueError('Surface inferred flags must match path points')
        weak=inferred[:-1]|inferred[1:]
        _segments(candidate_body,p,GRAY,1,~weak);_segments(candidate_body,p,GRAY,1,weak,True)
    for i in reversed(range(len(candidates))):
        p=_points(candidates[i].get('points_xy'),geo);_segments(candidate_body,p,CANDIDATE_COLORS[i],2)
        if len(p) and np.isfinite(p[0]).all():
            x,y=np.rint(p[0]).astype(int);cv2.putText(candidate_body,f'C{i+1}/E{int(candidates[i].get("entry_index",0))+1}',(x+4,y-4),cv2.FONT_HERSHEY_SIMPLEX,.43,CANDIDATE_COLORS[i],1,cv2.LINE_AA)
    cp=_header(candidate_body,['3 候选区域 / 路径',
        '灰：历史 F3 卷面参考（虚线含推断）；彩：当前候选中心',
        f'当次候选 {len(candidates)} 条；场景参考不等于当帧新识别或同根真值'])
    final_body=_body(bgr,geo,.75)
    state=str(meta.get('display_status',temporal.get('status','UNKNOWN')))
    title='未知';detail='本次无合格位置；不显示为已识别';idx=temporal.get('accepted_candidate_index')
    if state in ['ACCEPTED','REACQUIRED']:
        if not temporal.get('accepted') or not isinstance(idx,(int,np.integer)) or not 0<=idx<len(candidates):
            raise ValueError('Accepted state must reference an actual current raw candidate index')
        supplied=temporal.get('points_xy')
        if supplied is not None:
            q=np.asarray(supplied);r=np.asarray(candidates[idx].get('points_xy'))
            if q.shape!=r.shape or not np.allclose(q,r,rtol=0,atol=1e-5):raise ValueError('Gate path differs from the indexed raw candidate')
        _accepted_path(final_body,candidates[idx],geo)
        title='接受' if state=='ACCEPTED' else '重新捕获（新状态段）'
        if state=='ACCEPTED' and not temporal.get('temporal_confirmed',False):title+=' / 历史待积累'
        detail='绿图像支持 黄热图 蓝A7 橙虚弱证据；均不保证同根身份'
    elif state=='STALE':
        p=_points(temporal.get('reference_points_xy'),geo)
        if len(p):_segments(final_body,p,ORANGE,2,dashed=True)
        reference_age=_age(meta,'display_reference_age_s',
            (float(temporal.get('reference_age_s') or 0.)+float(detection_age or 0.)))
        title='STALE / 当帧未通过'
        detail=f'橙虚线仅旧参考，观察年龄 {_number(reference_age)}；不是当前测量'
    elif state!='UNKNOWN':raise ValueError('Unknown display state: '+state)
    if state=='UNKNOWN':final_body=_text(final_body,'UNKNOWN：本次无合格位置',(14,BODY_HEIGHT-35),20,(255,190,80),1)
    elif state=='STALE':final_body=_text(final_body,'旧参考 / 非当前测量',(14,BODY_HEIGHT-35),20,(255,190,80),1)
    fp=_header(final_body,['4 活动绳位置 | '+title,detail,
        f'检测媒体 t={_number(det)}  观察年龄={_number(detection_age)}  历史 N={temporal.get("temporal_history_count",0)}  epoch={temporal.get("epoch",0)}'])
    return np.concatenate([np.concatenate([original,hp],axis=1),np.concatenate([cp,fp],axis=1)],axis=0)
