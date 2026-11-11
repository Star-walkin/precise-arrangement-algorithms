import hashlib,subprocess,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from local_measurement import ROOT,BG4,SOURCE,D,read,dump,line_y
OUT=ROOT/'delivery'
FONT='C:/Windows/Fonts/msyh.ttc'
FONTS={s:ImageFont.truetype(FONT,s) for s in [19,21,23,26,30,36]}
LABEL={'separated':'尚未贴边 / 已离开','unknown':'当前证据不足','near_pending':'接近边缘，等待持续证据',
       'near_candidate':'视觉贴边候选','candidate_pending_evidence':'候选窗口内，本帧未确认'}
COLOR={'separated':'#a6bbc9','unknown':'#a6bbc9','near_pending':'#e7b653','near_candidate':'#67d5b4','candidate_pending_evidence':'#e7b653'}
def line(draw,xy,text,size=23,fill='#e6edf3'):draw.text(xy,text,font=FONTS[size],fill=fill)
def annotate(im,r):
    a=im.copy();rope=r.get('rope',{});b=r.get('boundary',{})
    if rope.get('valid'):
        p=np.rint(rope['points_xy']).astype(np.int32)
        cv2.polylines(a,[p],False,(70,230,60),4,cv2.LINE_AA)
        xs=np.arange(int(rope['xmax']),int(D)+70,14)
        for k in range(1,len(xs),2):
            q=np.c_[xs[k-1:k+1],line_y(rope['abc'],xs[k-1:k+1])]
            cv2.polylines(a,[np.rint(q).astype(np.int32)],False,(30,160,255),3,cv2.LINE_AA)
        cv2.circle(a,(round(D),round(rope['y_at_d'])),8,(30,160,255),2)
    if b.get('valid'):
        p=np.asarray(b['points_xy']);xs=np.linspace(p[:,0].min(),p[:,0].max(),120)
        yy=np.polyval(b['polynomial'],(xs-D)/160)
        cv2.polylines(a,[np.rint(np.c_[xs,yy]).astype(np.int32)],False,(230,200,40),3,cv2.LINE_AA)
        for p in b['points_xy']:cv2.circle(a,tuple(np.rint(p).astype(int)),3,(0,240,255),-1)
    cv2.line(a,(round(D),2020),(round(D),2700),(220,90,240),3)
    return a
def panel(im,r,s,b,all_states):
    canvas=Image.new('RGB',(1920,1080),'#111b25');d=ImageDraw.Draw(canvas)
    state=s['temporal']['state'];f=r['frame'];a=annotate(im,r)
    line(d,(24,15),'ECW-3  活动绳视觉贴边窗口实验',36)
    line(d,(1130,24),f"{f/10:05.1f} s  |  第 {f:03d} / 600 帧",30)
    line(d,(24,66),LABEL[state],30,COLOR[state])
    line(d,(840,73),'候选表示图像接近；未标定实体接触零点',23,'#e7b653')
    full=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)).resize((800,586),Image.Resampling.LANCZOS)
    canvas.paste(full,(24,131));d.rectangle((24,131,824,717),outline='#304455',width=1)
    for j,img in enumerate([im,a]):
        yy=147+j*324
        line(d,(853,yy-29),'入卷区域原图' if j==0 else '同一帧叠加：绿色实测自由段 / 橙色虚线为外推',21)
        crop=img[2030:2580,850:2850]
        pic=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((1040,286),Image.Resampling.LANCZOS)
        canvas.paste(pic,(853,yy))
    rope=r.get('rope',{});e=s['evidence'];abc=r.get('s1_abc')
    line(d,(24,743),'原图坐标：x = d = 2238.7 像素',23,'#e4a3ec')
    if abc:line(d,(24,778),f'S1: {abc[0]:+.5f}x {abc[1]:+.5f}y {abc[2]:+.2f} = 0',21)
    if rope.get('valid'):
        line(d,(24,810),f"局部自由段：y = {rope['slope']:.5f}x + {rope['abc'][0]*D+rope['y_at_d']:.2f}",21)
        line(d,(24,841),f"至 d 外推 {rope['extrapolation_to_d_px']:.0f} px；局部线拟合并非完整弯绳",21,'#bbc6cf')
    else:line(d,(24,810),'局部自由段门槛未通过，不延续伪实测',21,'#bbc6cf')
    n=b['temporal'].get('value')
    line(d,(24,875),'BG4 剩余根数辅助估计：'+(f'{n:.2f} 根' if n is not None else '证据不足'),21,'#bbc6cf')
    if e['local_valid']:
        line(d,(853,792),f"中心线—可见轮廓投影差：{r['center_gap_proxy_px']:+.1f} px",26)
        line(d,(853,830),f"拟合敏感度界：±{r['sensitivity_bound_px']:.1f} px（不是统计置信区间）",21,'#bbc6cf')
        line(d,(853,861),f"归一化投影差：{r['center_gap_width']:+.2f} 个光学绳宽",23)
    else:
        line(d,(853,792),'本帧局部几何不可靠，不输出有效间隙',26,'#e7b653')
    line(d,(853,900),'青色：可见黄/灰界面；黄色点：当前边界证据',21,'#61d3df')
    # Only already-observed decisions are shown. Future frames stay dark.
    x0,x1,y0=26,1890,972;d.rectangle((x0,y0,x1,y0+23),fill='#2a3742')
    for i,t in enumerate(all_states[:f+1]):
        st=t['temporal']['state'];col=COLOR[st] if st in ['near_candidate','near_pending','candidate_pending_evidence'] else '#526579'
        xx=x0+(x1-x0)*i/601;xe=x0+(x1-x0)*(i+1)/601;d.rectangle((xx,y0,xe,y0+23),fill=col)
    xx=x0+(x1-x0)*f/601;d.line((xx,y0-8,xx,y0+29),fill='white',width=2)
    line(d,(24,935),'因果时序：仅绘制当前时刻之前的状态',19,'#bbc6cf')
    for t in [0,10,20,30,40,50,60]:line(d,(x0+(x1-x0)*t/60.1-5,1001),f'{t}s',19,'#bbc6cf')
    line(d,(24,1040),'原始左相机视频 10 fps；601 帧完整回放。局部合并、遮挡或证据过期均保留不确定性。',19,'#bbc6cf')
    return np.asarray(canvas)
def writer(path):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','10','-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
def main():
    rows=read(ROOT/'v3_full/records.json');states=read(OUT/'frame_decisions.json');base=read(BG4/'full_003/compact_validation.json')
    cap=cv2.VideoCapture(str(SOURCE));dest=OUT/'ECW3_full_601frames.mp4';clip=OUT/'ECW3_near_edge_20-35s.mp4'
    enc=writer(dest);short=writer(clip);ids=[195,239,244,245,249,252,275,298,300,302,303,305,308,350,500]
    checks=[];start=time.perf_counter()
    for f,r in enumerate(rows):
        ok,im=cap.read();assert ok and hashlib.sha256(im.tobytes()).hexdigest()==r['source_pixel_sha256']
        rgb=panel(im,r,states[f],base[f],states);enc.stdin.write(rgb.tobytes())
        if 200<=f<350:short.stdin.write(rgb.tobytes())
        if f in ids:
            Image.fromarray(rgb).save(OUT/f'preview_{f:03d}.jpg',quality=95);checks.append(f)
        if f%100==0:print('render',f,round(time.perf_counter()-start,1),flush=True)
    cap.release()
    for e in [enc,short]:
        e.stdin.close();err=e.stderr.read().decode('utf8',errors='replace');code=e.wait();assert code==0,err[-2500:]
    verify=[]
    for path,count in [(dest,601),(clip,150)]:
        c=cv2.VideoCapture(str(path));n=0;shape=None
        while True:
            ok,a=c.read()
            if not ok:break
            n+=1;shape=list(a.shape)
        fps=c.get(cv2.CAP_PROP_FPS);c.release();assert n==count and shape==[1080,1920,3] and fps==10
        verify.append(dict(path=str(path),decoded_frames=n,shape=shape,fps=fps,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    dump(OUT/'video_receipt.json',dict(videos=verify,preview_frames=checks,elapsed_s=time.perf_counter()-start))
if __name__=='__main__':main()
