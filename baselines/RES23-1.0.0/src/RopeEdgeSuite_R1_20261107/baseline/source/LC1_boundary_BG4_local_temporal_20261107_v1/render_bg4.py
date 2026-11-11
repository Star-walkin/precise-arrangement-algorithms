"""Render the exact current-frame curves; predictive values never fabricate lines."""
import sys,time,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from bg4_common import *
FONT='C:/Windows/Fonts/msyh.ttc';FONTS={s:ImageFont.truetype(FONT,s) for s in (18,20,22,24,28,32,36)}
LABELS=dict(measured='实测校正',estimated='估计校正',predicted='短时预测',unknown='未知')
COLORS=dict(measured=(115,245,145),estimated=(255,205,95),predicted=(90,200,255),unknown=(255,135,130))

def overlay(image,r):
    a=image.copy();b=r['BG4'];good=r['region_gate_passed'];d=round(D)
    if good:
        for name,col in [('yellow',(0,165,255)),('surface',(245,130,20))]:
            m=cv2.imdecode(np.fromfile(BASE/r['masks'][name]['path'],np.uint8),0)>0
            a[m]=(a[m]*.94+np.array(col)*.06).astype(np.uint8)
    for x,y in r['region'].get('top_observed_xy') or []:cv2.circle(a,(round(x),round(y)),3,(230,60,255) if good else (100,100,100),-1)
    for c in (b['curves'] if b else []):
        xy=np.rint(c['xy']).astype(np.int32);ss=np.asarray(c['point_states'])=='observed';edge=ss[:-1]&ss[1:]
        starts=np.r_[0,np.flatnonzero(edge[1:]!=edge[:-1])+1];ends=np.r_[starts[1:],len(edge)]
        for lo,hi in zip(starts,ends):cv2.polylines(a,[xy[lo:hi+1]],False,(45,245,45) if edge[lo] else (0,150,255),2)
    cv2.line(a,(d,1750),(d,2999),(0,255,255),2)
    abc=r['s1'].get('current_line_abc')
    if r['s1']['valid'] and abc and abs(abc[1])>.001:
        for x in range(0,4096,40):
            xx=min(4095,x+20);y=-(abc[0]*x+abc[2])/abc[1];yy=-(abc[0]*xx+abc[2])/abc[1]
            cv2.line(a,(x,round(y)),(xx,round(yy)),(255,255,255),2)
        xy=np.rint(r['s1'].get('line_points_xy') or []).astype(np.int32)
        if len(xy)>1:cv2.polylines(a,[xy],False,(255,255,0),2)
    edge=r['region'].get('edge_y');active=r['intersection_y_px']
    if good and edge is not None:cv2.circle(a,(d,round(edge)),7,(230,60,255),2)
    if active is not None:cv2.circle(a,(d,round(active)),7,(255,255,255),2)
    if b and edge is not None:
        for q in b['crossings']:
            if edge-25<=q['y']<=edge+180:
                cv2.circle(a,(d,round(q['y'])),4,(255,255,255) if not q['lower_confidence'] else (0,170,255),-1)
                cv2.putText(a,q['boundary_id'].split('_')[-1],(d+10,round(q['y'])+3),cv2.FONT_HERSHEY_SIMPLEX,.4,(245,245,245),1,cv2.LINE_AA)
    return a

def render(image,r):
    src=overlay(image,r);a=np.full((1080,1920,3),(22,18,15),np.uint8)
    a[:700,:1280]=cv2.resize(src[1570:3000,0:3000],(1280,700))
    a[755:1070,:430]=cv2.resize(src,(430,315))
    edge=r['region'].get('edge_y');y0=max(1850,min(2650,round(edge or 2335)-60))
    a[755:1070,445:850]=cv2.resize(image[y0:y0+340,2020:2450],(405,315))
    a[755:1070,865:1270]=cv2.resize(src[y0:y0+340,2020:2450],(405,315))
    img=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));dr=ImageDraw.Draw(img)
    def txt(x,y,s,n=22,col=(225,230,238)):dr.text((x,y),s,font=FONTS[n],fill=col)
    txt(18,12,'BG4：d 附近复核 + 分级观测 + 有限时序预测',28)
    txt(10,718,'完整原图叠加',20);txt(450,718,'计数区原图',20);txt(870,718,'相同位置的曲线与交点',20)
    x=1310;y=18;b=r['BG4'];tr=r['temporal'];st=tr['state'];obs=r['observation']
    txt(x,y,f"帧 {r['frame_index']:03d}/600     视频 {r['nominal_time_s']:.1f}s",28);y+=48
    txt(x,y,'d = 2238.7 px（原图固定列）',24);y+=42
    txt(x,y,'卷面门控：'+('通过' if r['region_gate_passed'] else '未通过，当前帧无测量'),22);y+=36
    abc=r['s1'].get('current_line_abc')
    if abc:
        aa,bb,c=abc;txt(x,y,'S1 当前直线（原图坐标）');y+=30
        txt(x,y,f'{aa:.6f} x + {bb:.6f} y');y+=30;txt(x,y,f'+ ({c:.3f}) = 0');y+=39
    active=r['intersection_y_px']
    txt(x,y,'活动绳交点 y：'+(f'{active:.2f} px' if active is not None else '未知'));y+=32
    txt(x,y,'卷面上界 y：'+(f'{edge:.2f} px' if edge is not None and r['region_gate_passed'] else '未确认'));y+=45
    if tr['display_N'] is not None:txt(x,y,f"剩余约 {tr['display_N']:.2f} 个绳距",32,COLORS[st])
    elif tr['value'] is not None:txt(x,y,'五圈范围以外 · 不显示距离',28,COLORS[st])
    else:txt(x,y,'剩余根数：未知',32,COLORS[st])
    y+=47;txt(x,y,'状态：'+LABELS[st],28,COLORS[st]);y+=40
    direction=dict(approaching='接近边界',departing='远离边界',unconfirmed='尚未确认')[tr['direction']]
    txt(x,y,'运动方向：'+direction);y+=32
    if tr['value'] is not None:
        txt(x,y,f"距上次观测 {tr['last_observation_age_s']:.1f}s；距相位锚点 {tr['last_phase_age_s']:.1f}s",20);y+=31
        txt(x,y,f"模型内部 σ = {tr['nominal_filter_sigma']:.2f} 个绳距",20);y+=31
    if obs:
        txt(x,y,f"当前图像候选：{obs['value']:.2f}（滤波前）",20);y+=31
        txt(x,y,'证据：'+('边界相位' if obs['phase_anchor'] else '间距换算，需近期相位锚点'),20);y+=31
    if b:
        nr=sum(q['recovered'] for q in b['local_recovery'])
        txt(x,y,f"连续曲线 {len(b['curves'])} 条；局部找回 {nr} 条",20);y+=31
        txt(x,y,f"d 处有效交点 {len(b['crossings'])}；绳距 {b['local_pitch_px']:.2f}px",20);y+=31
    base=r['baseline_display_N']
    txt(x,max(750,y+8),'旧版 BG3：'+(f'{base:.2f} 个绳距' if base is not None else '未知' if not r['baseline_count_valid'] else '五圈外'),22,(190,190,195))
    for yy,label,col in [(810,'绿：图像支持曲线；橙：模型补全',(120,240,130)),(843,'黄竖线：x=d；紫色：卷面上界',(240,210,120)),
        (876,'短时预测最长 0.3 秒；不补画缺失绳边',(125,215,250)),(909,'仅用历史与当前帧；换向后重新确认',(225,230,238)),
        (942,'σ 为滤波内部量，不等于真实测量误差',(255,205,140)),(987,'离线结果；真实根数与触边零点未标定',(255,205,140))]:txt(x,yy,label,18,col)
    return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

def main():
    cv2.setNumThreads(1);run=ROOT/'full_001';out=ROOT/'delivery';out.mkdir(exist_ok=True)
    dst=out/'BG4_local_temporal_full.mp4';assert not dst.exists()
    frozen=sha(__file__);cap=cv2.VideoCapture(str(SOURCE));log=(out/'ffmpeg.log').open('wb')
    ff=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24',
        '-s','1920x1080','-r','10','-i','-','-an','-c:v','libx264','-threads','2','-crf','20','-preset','fast','-pix_fmt','yuv420p',str(dst)],
        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
    for i in range(601):
        path=run/'frames'/f'{i:06d}.json';start=time.monotonic()
        while not path.exists():
            if time.monotonic()-start>1800:raise TimeoutError(path)
            time.sleep(.2)
        r=read(path);ok,im=cap.read();assert ok and hashlib.sha256(im.tobytes()).hexdigest()==r['source_pixel_sha256']
        result=render(im,r);ff.stdin.write(result.tobytes())
        if i in (0,60,126,127,163,200,240,270,300,450,600):cv2.imencode('.png',result)[1].tofile(str(out/f'preview_{i:03d}.png'))
        if i%50==0:print('Rendered',i,flush=True);dump(out/'render_progress.json',dict(rendered=i+1,planned=601))
    cap.release();ff.stdin.close();assert ff.wait()==0;log.close();assert frozen==sha(__file__)
    cap=cv2.VideoCapture(str(dst));fps=cap.get(cv2.CAP_PROP_FPS);n=0
    while True:
        ok,a=cap.read()
        if not ok:break
        assert a.shape==(1080,1920,3);n+=1
    cap.release();assert n==601 and abs(fps-10)<1e-6
    dump(out/'video_receipt.json',dict(status='completed',decoded_frames=n,fps=fps,duration_s=n/fps,
        source_pixels_bound=True,sha256=sha(dst),renderer_sha256=frozen,curves_and_count_share_same_models=True))
    print('VIDEO COMPLETE',flush=True)
if __name__=='__main__':main()
