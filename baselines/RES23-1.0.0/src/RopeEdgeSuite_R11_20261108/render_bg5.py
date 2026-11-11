"""Render new-video output and raw evidence side by side; no recomputation."""
import argparse,gzip,json,subprocess,time,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parent
FONTS={s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s) for s in [18,20,22,26,30,36]}
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def record(run,i):
    with gzip.open(run/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:return json.load(f)
def text(d,xy,s,size=22,color='#e3ebf2'):d.text(xy,s,font=FONTS[size],fill=color)
def annotate(im,r):
    a=im.copy();reg=r['region'];b=r.get('BG4');s=r['s1'];ec=r['ECW'];d=2238.7
    if reg['valid']:
        for xy in reg.get('top_curve_supported_segments',[]):
            cv2.polylines(a,[np.rint(xy).astype(np.int32)],False,(255,200,40),4,cv2.LINE_AA)
    if b:
        green=[];amber=[]
        for c in b['curves']:
            pts=np.rint(c['xy']).astype(np.int32);observed=np.asarray(c['point_states'])=='observed'
            # Fitted models are continuous. Unsupported parts remain visually distinct.
            supported=observed[1:]&observed[:-1]
            inferred=(~supported)&(np.arange(1,len(pts))%9<4)
            # Batch identical segments to reduce rendering overhead; geometry,
            # state classification and the original dash pattern are unchanged.
            for mask,dest in [(supported,green),(inferred,amber)]:
                changes=np.diff(np.r_[False,mask,False].astype(np.int8))
                for lo,hi in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):dest.append(pts[lo:hi+1])
        if green:cv2.polylines(a,green,False,(50,190,70),1,cv2.LINE_AA)
        if amber:cv2.polylines(a,amber,False,(0,150,235),1,cv2.LINE_AA)
    if s['valid']:
        p=np.rint(s['line_points_xy']).astype(np.int32);cv2.polylines(a,[p],False,(80,255,80),4,cv2.LINE_AA)
        abc=s['current_line_abc'];x0=float(np.max(p[:,0]));xs=np.arange(x0,d+1,16)
        for k in range(1,len(xs),2):
            q=np.c_[xs[k-1:k+1],-(abc[0]*xs[k-1:k+1]+abc[2])/abc[1]]
            cv2.polylines(a,[np.rint(q).astype(np.int32)],False,(0,160,255),3,cv2.LINE_AA)
    if ec.get('boundary',{}).get('valid'):
        for p in ec['boundary']['points_xy']:cv2.circle(a,tuple(np.rint(p).astype(int)),4,(0,240,255),-1)
    cv2.line(a,(2239,1800),(2239,2999),(220,90,240),3)
    if ec.get('rope',{}).get('valid'):
        for p in ec['rope']['points_xy'][::3]:cv2.circle(a,tuple(np.rint(p).astype(int)),3,(245,245,245),-1)
    if ec.get('valid'):
        yr=int(round(ec['rope']['y_at_d']));yb=int(round(ec['boundary']['y_at_d']))
        cv2.line(a,(2256,yr),(2256,yb),(245,245,245),3,cv2.LINE_AA)
        cv2.circle(a,(2239,yr),9,(245,245,245),2,cv2.LINE_AA)
        cv2.circle(a,(2239,yb),7,(0,240,255),2,cv2.LINE_AA)
    return a
def panel(im,r,session,n):
    frame=r['frame_index'];a=annotate(im,r);ec=r['ECW'];decision=r['ECW_decision'];reg=r['region'];s=r['s1'];count=r['temporal']
    p=Image.new('RGB',(1920,1080),'#111b25');d=ImageDraw.Draw(p)
    text(d,(24,16),'RES-1.1 · BG5.1 卷面验证优化' if r.get('BG5_first_pass_record_sha256') else 'RES-1.1 · BG5 卷面验证优化',36)
    text(d,(1260,25),f'{frame/10:05.1f}s  |  {frame}/{n-1}',30)
    text(d,(24,67),session+' / left',22)
    status=decision['temporal']['state'];labels={'near_candidate':'视觉贴边候选','near_pending':'近边，等待持续证据',
        'candidate_pending_evidence':'候选窗内，本帧未确认','separated':'当前分离证据','unknown':'当前贴边状态未定'}
    text(d,(1040,70),labels[status],26,'#66d4b2' if status=='near_candidate' else '#e6bb6b')
    pic=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)).resize((800,586),Image.Resampling.LANCZOS);p.paste(pic,(24,131))
    edge=reg.get('edge_y')
    if edge is None:edge=ec.get('boundary',{}).get('y_at_d')
    if edge is None:edge=r.get('intersection_y_px')
    yy=int(np.clip((edge if edge is not None else 2350)-180,1700,2450))
    for j,img in enumerate([im,a]):
        top=147+j*324;text(d,(853,top-29),'原图局部' if j==0 else '同帧叠加：卷面边界 / F3 分组曲线 / 活动绳',20)
        crop=img[yy:yy+550,850:2850];pic=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((1040,286),Image.Resampling.LANCZOS);p.paste(pic,(853,top))
    text(d,(24,743),'固定 d = 2238.7 px；本片未重新标定 d',22,'#e4a3ec')
    support=np.asarray(s.get('line_points_xy') or [],float)
    extrap=max(0.,2238.7-float(support[:,0].max()),float(support[:,0].min())-2238.7) if s['valid'] else None
    text(d,(24,779),'活动绳 S1：'+(f'光学候选，至 d 外推 {extrap:.0f} px' if s['valid'] else '未通过当前支撑门槛'),22)
    if s['valid']:
        abc=s['current_line_abc'];text(d,(24,812),f'{abc[0]:+.5f}x {abc[1]:+.5f}y {abc[2]:+.2f}=0',20)
    else:
        why=r['S1_tracker_full']['diagnostics'].get('accept_reason','')
        meaning={'no_current_verified_E1':'当前入绳入口未通过验证','current_S1_support_points_insufficient':'当前活动绳支撑点不足',
                 'current_S1_span_insufficient':'当前活动绳有效直线段过短'}.get(why,why)
        text(d,(24,812),meaning[:64],18,'#b8c7d3')
    old=r.get('comparison_baseline',{}).get('region_valid')
    text(d,(24,846),f"卷面区域：{'可用' if reg['valid'] else '证据不足'}  |  旧版：{'通过' if old else '未通过'}",22,'#66d4b2' if reg['valid'] else '#e6bb6b')
    reasons=reg.get('reasons',[])
    translations={'global_flange_structure_insufficient':'黄色区域与卷面交界的邻接证据不足','outer_flange_not_confirmed':'黄色外轮缘局部证据不足','insufficient_yellow_flange':'黄色区域不足',
        'd_not_bracketed_by_current_interface_evidence':'d 两侧缺少当前边界点','local_edge_residual_too_large':'d 附近边界拟合残差过大',
        'incomplete_top_interface_support':'卷面顶部边界支撑不足'}
    if reasons:text(d,(24,880),translations.get(reasons[0],reasons[0])[:50],20,'#e6bb6b')
    else:text(d,(24,880),'依据：当前黄灰交界、d 两侧支撑及卷面纹理',20,'#b8c7d3')
    val=count['display_N'];label={'measured':'图像校正','estimated':'估计','predicted':'短时预测','unknown':'未知'}[count['state']]
    txt=f'{val:.2f} 根（{label}）' if val is not None else ('超过五根显示范围' if count['value'] is not None else '未知')
    text(d,(853,785),'剩余根数：'+txt,30)
    if val is None and count['value'] is None:
        if not s['valid']:why='当前活动绳跟踪不足'
        elif not reg['valid']:why='当前卷面区域证据不足'
        elif r.get('intersection_y_px') is not None and r['intersection_y_px']<reg['edge_y']:why='S1 投影越过参考边界，禁止计数'
        elif 'prediction_or_uncertainty_limit' in count['events']:why='观测中断超过允许时限'
        else:why='F3 局部计数或连续相位证据不足'
        text(d,(853,826),'原因：'+why,20,'#e6bb6b')
    if decision['evidence']['local_valid']:
        text(d,(853,861),f"ECW 投影差：{ec['center_gap_proxy_px']:+.1f} px（非实体间隙）",20)
    else:text(d,(853,861),'ECW 当前贴边证据不足',20,'#b8c7d3')
    text(d,(853,895),f"F3 原始点 {r['fresh_F3']['point_count']}  /  连续边界 {len(r['BG4']['curves']) if r['BG4'] else 0}",20)
    text(d,(24,930),'青色：卷面边界   绿色：当前支撑   橙色虚线：插值/外推   黄色点：ECW 边界   白点/白圈：ECW 绳支撑/投影交点',20,'#b8c7d3')
    d.rectangle((25,978,1892,998),fill='#2c3e4c');xx=25+1867*frame/max(1,n-1);d.rectangle((25,978,xx,998),fill='#477b91')
    text(d,(24,1025),'卷面对照实验：沿用旧版同帧 S1；新算卷面、F3、计数与贴边。d 未重新标定；无实体接触及根数真值。',20,'#e6bb6b')
    return np.asarray(p)
def writer(path):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-nostats','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','10','-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--session',required=True);ap.add_argument('--preview',type=str);ap.add_argument('--live',action='store_true');a=ap.parse_args()
    run=ROOT/'runs'/a.session;bind=read(run/'binding.json');out=run/'delivery';out.mkdir(exist_ok=True)
    source_session=a.session.split('_pilot')[0].removesuffix('_refined')
    oldrows=read(ROOT.parent/'RopeEdgeSuite_R1_20261107'/'transfer_runs'/source_session/'compact.json')
    n=bind['processed_frames_planned'];cap=cv2.VideoCapture(bind['source']);preview=[0,50,100,150,200,250,300,350,400,450,500,550,n-1]
    if a.preview:
        for i in map(int,a.preview.split(',')):
            cap.set(1,i);ok,im=cap.read();assert ok;r=record(run,i);r['comparison_baseline']=oldrows[i];rgb=panel(im,r,a.session,n);Image.fromarray(rgb).save(out/f'preview_{i:03d}.jpg',quality=95)
        cap.release();return
    if not a.live:assert read(run/'receipt.json')['status']=='complete'
    dest=out/f'RES11_BG5_{a.session}_full.mp4';enc=writer(dest);started=time.perf_counter()
    for i in range(n):
        deadline=time.monotonic()+600
        while not (run/'frames'/f'{i:06d}.json.gz').exists():
            assert a.live and time.monotonic()<deadline,'Timed out waiting for a committed frame'
            time.sleep(.25)
        ok,im=cap.read();assert ok;r=record(run,i);assert hashlib.sha256(im.tobytes()).hexdigest()==r['source_pixel_sha256']
        r['comparison_baseline']=oldrows[i];rgb=panel(im,r,a.session,n);enc.stdin.write(rgb.tobytes())
        if i in preview:Image.fromarray(rgb).save(out/f'preview_{i:03d}.jpg',quality=95)
        if i%100==0:print('render',a.session,i,round(time.perf_counter()-started,1),flush=True)
    cap.release();enc.stdin.close();err=enc.stderr.read().decode('utf8',errors='replace');assert enc.wait()==0,err[-2000:]
    deadline=time.monotonic()+60
    while not (run/'receipt.json').exists():
        assert time.monotonic()<deadline;time.sleep(.25)
    assert read(run/'receipt.json')['status']=='complete'
    c=cv2.VideoCapture(str(dest));count=0
    while True:
        ok,img=c.read()
        if not ok:break
        count+=1;assert img.shape==(1080,1920,3)
    fps=c.get(5);c.release();assert count==n and fps==10
    (out/'video_receipt.json').write_text(json.dumps(dict(path=str(dest),decoded_frames=count,fps=fps,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),elapsed_s=time.perf_counter()-started),indent=2),encoding='utf8')
if __name__=='__main__':main()
