import cv2,numpy as np,subprocess
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from common import *
FONTS={}
def font(n):
    if n not in FONTS:FONTS[n]=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    return FONTS[n]
def polyline(im,pts,color,width=4):
    if len(pts)>1:cv2.polylines(im,[np.rint(pts).astype(np.int32)],False,color,width,cv2.LINE_AA)
def annotate(analysis_image,r):
    a=analysis_image.copy();b=r['boundary'];s=r['track']
    if b['fit_valid']:
        polyline(a,b['top_curve_xy'],(0,180,255),5)
        for p in b['observed_segments']:polyline(a,p,(255,215,25),5)
        layer=a.copy()
        for p in b['foreground_polygons']:cv2.fillPoly(layer,[np.rint(p).astype(np.int32)],(20,20,245))
        a=cv2.addWeighted(a,.82,layer,.18,0)
        if not b.get('d_locally_observed') and b.get('edge_y') is not None:
            x=int(round(D));y=int(round(b['edge_y']));u=int(round(b.get('edge_uncertainty_px') or 0))
            cv2.circle(a,(x,y),9,(0,180,255),3,cv2.LINE_AA)
            cv2.line(a,(x,y-u),(x,y+u),(0,180,255),3,cv2.LINE_AA)
            for yy in [y-u,y+u]:cv2.line(a,(x-8,yy),(x+8,yy),(0,180,255),3,cv2.LINE_AA)
    if s['valid']:
        for p in s['supported_segments_xy']:polyline(a,p,(70,255,70),4)
    cv2.line(a,(round(D),1850),(round(D),2750),(220,60,210),3)
    return a
def pic(im,size):return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize(size,Image.Resampling.LANCZOS)
def panel(original,r,item):
    mirror=item['camera']=='right';analysis=cv2.flip(original,1) if mirror else original
    ann=annotate(analysis,r);ann=cv2.flip(ann,1) if mirror else ann
    w=original.shape[1];d0=w-1-D if mirror else D
    xa,xb=(w-1-2800,w-1-800) if mirror else (800,2800)
    p=Image.new('RGB',(1920,1080),'#111b25');draw=ImageDraw.Draw(p);b=r['boundary'];s=r['track']
    draw.text((24,18),'RES-1.5 · SG3 跟踪 / BG9 边缘分组拟合',font=font(34),fill='white')
    draw.text((1420,22),f"{r['time_s']:05.1f}s  |  {r['frame_index']}/{max(0,item['planned_frames']-1)}",font=font(28),fill='white')
    if item.get('partial_source'):
        draw.text((870,27),f"截断源文件：仅恢复 {item['planned_frames']} 帧",font=font(23),fill='#edbd6c')
    label='右相机 · 镜像分析后还原显示' if mirror else '左相机 · 原始朝向'
    draw.text((24,72),item['session']+'  /  '+label,font=font(23),fill='#bacbd9')
    p.paste(pic(ann,(800,586)),(24,132))
    draw.text((850,114),'当前原图局部',font=font(22),fill='white')
    p.paste(pic(original[1850:2500,xa:xb],(1040,338)),(852,148))
    draw.text((852,499),'同帧活动绳 / 曲线拟合 / 遮挡剔除',font=font(22),fill='white')
    p.paste(pic(ann[1850:2500,xa:xb],(1040,338)),(852,534))
    draw.text((24,748),'活动绳：'+('当前支撑检查通过' if s['valid'] else '当前跟踪证据不足'),font=font(25),fill='#7edeb6' if s['valid'] else '#edbd6c')
    if s['valid']:
        A,B,C=r['original_line_abc'];draw.text((24,786),f'{A:+.5f}x {B:+.5f}y {C:+.2f}=0',font=font(21),fill='#bacbd9')
    draw.text((24,826),f'参考 d（原图）={d0:.1f}px；本片未重新标定',font=font(22),fill='#edbd6c')
    if b['fit_valid']:
        mode='当前可见锚点支持' if b.get('d_locally_observed') else 'd 处为模型推断'
        text=f"边界：{mode}；y(d)={b['edge_y']:.1f}px"
    else:
        labels={'initial_shape_evidence_accumulating':'可靠形状初始化中','current_boundary_candidates_insufficient':'候选边缘不足',
            'unoccluded_current_anchor_support_insufficient':'未遮挡锚点不足','outline_at_d_model_sensitive':'模型推断不稳定',
            'current_anchor_fit_residual_excessive':'锚点与形状不一致','camera_motion_invalid':'相机背景连续性不足',
            'shape_prior_too_old':'形状参考过期','robust_anchor_support_insufficient':'鲁棒拟合支持不足',
            'current_boundary_group_ambiguous':'边缘分组仍有歧义'}
        if 'camera_motion_invalid' in b['reasons']:
            text='边界：相机背景稳定性检查未通过'
        else:text='边界：'+labels.get(b['reasons'][0],b['reasons'][0])
    draw.text((24,866),text,font=font(21),fill='#bacbd9')
    state='计数列受遮挡；实体贴边待核验' if b.get('d_foreground_occluded') else '实体贴边与剩余根数：本轮不作数值确认'
    draw.text((852,892),state,font=font(23),fill='#edbd6c')
    draw.text((24,940),'青：可见边缘支持　橙：同一曲线的推断段　绿：活动绳实测段　红：宽亮带/反光剔除　紫：参考 d',font=font(22),fill='#bacbd9')
    draw.rectangle((24,983,1892,1000),fill='#243947')
    draw.rectangle((24,983,24+int(1868*(r['frame_index']+1)/max(1,item['planned_frames'])),1000),fill='#427f94')
    draw.text((24,1023),'SG3 当前实测分段 + BG9 边缘分组共识；未运行全片 F3/根数。通过内部检查不等于检测正确。',font=font(22),fill='#edbd6c')
    return np.asarray(p)
def writer(path,fps):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
