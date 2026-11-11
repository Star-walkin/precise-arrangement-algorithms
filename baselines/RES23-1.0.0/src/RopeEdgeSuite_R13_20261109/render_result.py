"""Display current evidence, rejected proposals, and conditional count separately."""
import sys,time,subprocess,hashlib,argparse
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from common import *
FONT={s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s) for s in [18,20,22,25,28,32,36]}

def text(dr,xy,value,size=22,color='#e3ebf2'):dr.text(xy,value,font=FONT[size],fill=color)
def dashed(a,xy,color,thick=2,step=3):
    p=np.rint(xy).astype(np.int32)
    for i in range(1,len(p),2*step):cv2.polylines(a,[p[i-1:min(i+step,len(p))]],False,color,thick,cv2.LINE_AA)

REASONS={'unsupported_extrapolation_too_long':'实测支撑距离 d 太远',
    'projected_position_sensitive_to_support_subset':'交点对拟合区段敏感',
    'near_edge_position_unresolved':'误差范围覆盖边界，位置未确定',
    'projection_boundary_incompatible':'直线投影与边界不相容',
    'current_target_boundary_unavailable':'计数位置的目标边界不可用',
    'current_active_rope_support_missing':'本帧活动绳证据不足',
    'independent_line_support_insufficient':'独立空间支撑不足'}

def annotate(im,r):
    a=im.copy();reg=r['region'];g=r['geometry_GV1'];s=r['S1_tracker_full'];guard=reg['identity_guard']
    overlay=a.copy()
    for poly in guard.get('foreground_polygons_xy',[]):cv2.fillPoly(overlay,[np.rint(poly).astype(np.int32)],(20,30,220))
    a=cv2.addWeighted(a,.72,overlay,.28,0)
    if reg['valid']:
        for xy in reg['top_curve_supported_segments']:cv2.polylines(a,[np.rint(xy).astype(np.int32)],False,(255,210,20),4,cv2.LINE_AA)
    elif len(reg.get('candidate_top_curve_xy',[])):
        dashed(a,reg['candidate_top_curve_xy'],(70,70,220),2,2)
    bg=r.get('BG4')
    if reg['valid'] and bg:
        greens=[];ambers=[]
        for c in bg['curves']:
            p=np.rint(c['xy']).astype(np.int32);flags=np.asarray(c['point_states'])=='observed'
            yes=flags[:-1]&flags[1:]
            for flags,target in [(yes,greens),(~yes&(np.arange(len(yes))%8<3),ambers)]:
                e=np.diff(np.r_[False,flags,False].astype(np.int8))
                for lo,hi in zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)):target.append(p[lo:hi+1])
        if greens:cv2.polylines(a,greens,False,(50,170,70),1,cv2.LINE_AA)
        if ambers:cv2.polylines(a,ambers,False,(0,130,200),1,cv2.LINE_AA)
    if s['valid']:
        for xy in s['supported_segments_xy']:cv2.polylines(a,[np.rint(xy).astype(np.int32)],False,(80,255,80),4,cv2.LINE_AA)
        abc=np.asarray(s['current_line_abc']);p=np.asarray(s['line_points_xy']);start=float(p[:,0].max())
        if start<D:
            x=np.arange(start,D+1,12);xy=np.c_[x,-(abc[0]*x+abc[2])/abc[1]]
            dashed(a,xy,(0,160,255) if g['conditional_projection_valid'] else (80,80,230),3,1)
        y=g['y_at_d']
        if y is not None:
            if g['conditional_projection_valid']:cv2.circle(a,(round(D),round(y)),9,(0,180,255),2)
            else:cv2.drawMarker(a,(round(D),round(y)),(80,80,230),cv2.MARKER_TILTED_CROSS,18,3)
    cv2.line(a,(round(D),1800),(round(D),2999),(220,90,240),3)
    return a

def panel(im,r,n=600):
    a=annotate(im,r);reg=r['region'];g=r['geometry_GV1'];s=r['s1'];t=r['temporal'];i=r['frame_index']
    p=Image.new('RGB',(1920,1080),'#111b25');dr=ImageDraw.Draw(p)
    text(dr,(24,16),'RES-1.3 · BG6 边界遮挡 / GV1 交点验证',36)
    text(dr,(1320,25),f'{i/10:05.1f}s  |  {i}/{n-1}',28)
    text(dr,(24,69),'113745 / left · 完整视频逐帧验证',22)
    blocked='broad_foreground_occludes_count_interface' in reg['reasons']
    label='导杆/宽亮带遮挡：拒绝目标边界' if blocked else ('参考 d 的交点检查通过' if g['conditional_projection_valid'] else '跟踪与计数资格分别检查')
    text(dr,(1015,70),label,25,'#ec9a90' if blocked else '#e6bb6b')
    p.paste(Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)).resize((800,586),Image.Resampling.LANCZOS),(24,131))
    edge=reg.get('candidate_edge_y') or 2200;yy=int(np.clip(edge-180,1700,2450))
    for j,img in enumerate([im,a]):
        top=147+j*324;text(dr,(853,top-29),'原图局部' if j==0 else '实测支撑 / 接受边界 / 拒绝候选 / F3',20)
        q=Image.fromarray(cv2.cvtColor(img[yy:yy+550,850:2850],cv2.COLOR_BGR2RGB)).resize((1040,286),Image.Resampling.LANCZOS);p.paste(q,(853,top))
    text(dr,(24,741),'参考 d = 2238.7 px；本视频重新标定未通过',22,'#e4a3ec')
    text(dr,(24,778),'活动绳跟踪：'+('当前支撑检查通过' if s['valid'] else '当前支撑不足'),22)
    if s['valid']:
        abc=s['current_line_abc'];text(dr,(24,811),f'{abc[0]:+.5f}x {abc[1]:+.5f}y {abc[2]:+.2f}=0',20)
    text(dr,(24,846),'计数边界：'+('当前检查通过' if reg['valid'] else '不可用，暂停 F3 与计数'),22,'#66d4b2' if reg['valid'] else '#ec9a90')
    if g.get('extrapolation_px') is not None:
        text(dr,(24,881),f"至 d 外推 {g['extrapolation_px']:.0f}px；交点检查：{'通过（有条件）' if g['conditional_projection_valid'] else '未通过'}",20,'#b8c7d3')
    if t['display_N'] is not None:value=f"参考 d 试验估计：{t['display_N']:.2f} 根"
    elif t['value'] is not None:value='参考 d 试验估计：超过五根'
    else:value='剩余根数：暂停输出'
    text(dr,(853,786),value,28)
    if not reg['valid']:why='宽亮带/导杆遮挡目标交界' if blocked else '当前卷面边界证据不足'
    elif not g['conditional_projection_valid']:why=REASONS.get(g['reasons'][0],g['reasons'][0])
    elif t['value'] is None:why='等待连续的当前 F3 相位证据'
    else:why='仅在固定参考 d 成立的条件下解释该数值'
    text(dr,(853,828),why,22,'#e6bb6b')
    line=g.get('line_sensitivity_bound_px');gap=g.get('raw_gap_px')
    if line is not None:text(dr,(853,867),f"直线交点敏感度 ±{line:.1f}px（开发指标）",20,'#b8c7d3')
    old=r['baseline_RES12']['count']['display_N'];oldtxt='未知/超出五根' if old is None else f'{old:.2f} 根'
    text(dr,(853,903),f"上一版显示：{oldtxt}；本帧 F3 点 {r['fresh_F3']['point_count']}",20,'#b8c7d3')
    text(dr,(24,935),'青：通过检查的边界  绿：同帧实测支撑  红：遮挡/拒绝的外推  紫：参考 d',20,'#b8c7d3')
    dr.rectangle((25,979,1892,997),fill='#2c3e4c');dr.rectangle((25,979,25+1867*i/max(1,n-1),997),fill='#477b91')
    text(dr,(24,1024),'d 未完成本片标定；试验数值不代表已验证根数精度。遮挡及远距离外推不转为有效读数。',22,'#e6bb6b')
    return np.asarray(p)

def writer(path):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
        '-pix_fmt','rgb24','-s','1920x1080','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','22',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--live',action='store_true');a=ap.parse_args()
    cv2.setNumThreads(1);run=ROOT/'runs/dual_20260923_113745';out=run/'delivery';out.mkdir(exist_ok=True)
    bind=read(run/'binding.json');cap=cv2.VideoCapture(bind['source']);n=bind['processed_frames_planned']
    path=out/'RES13_BG6_GV1_113745_full.mp4';enc=writer(path);start=time.perf_counter()
    previews={20,75,100,175,225,250,275,300,316,325,332,336,355,369,375,395,396,404,415,450,475,526,575,599}
    for i in range(n):
        deadline=time.monotonic()+600
        while not (run/'frames'/f'{i:06d}.json.gz').exists():
            assert a.live and time.monotonic()<deadline;time.sleep(.2)
        ok,im=cap.read();assert ok;r=record(i,run);assert pixelsha(im)==r['source_pixel_sha256']
        rgb=panel(im,r,n);enc.stdin.write(rgb.tobytes())
        if i in previews:Image.fromarray(rgb).save(out/f'preview_{i:03d}.jpg',quality=94)
        if i%100==0:print('rendered',i,round(time.perf_counter()-start,1),flush=True)
    cap.release();enc.stdin.close();err=enc.stderr.read().decode(errors='replace');assert enc.wait()==0,err[-2000:]
    check=cv2.VideoCapture(str(path));count=0
    while True:
        ok,im=check.read()
        if not ok:break
        assert im.shape==(1080,1920,3);count+=1
    fps=check.get(5);check.release();assert count==n and fps==10
    dump(out/'video_receipt.json',dict(decoded_frames=count,fps=fps,sha256=sha(path),path=str(path),elapsed_s=time.perf_counter()-start))
if __name__=='__main__':main()
