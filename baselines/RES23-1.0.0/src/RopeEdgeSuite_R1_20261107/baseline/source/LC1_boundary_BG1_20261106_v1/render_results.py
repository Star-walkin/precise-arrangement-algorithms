"""Render source-bound current observations, including every rejected frame."""
from pathlib import Path
import argparse,cv2,numpy as np,json,hashlib,subprocess,time
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from inspect_regions import SOURCE
ROOT=Path(__file__).resolve().parent
FONT='C:/Windows/Fonts/msyh.ttc'
fonts={n:ImageFont.truetype(FONT,n) for n in [18,21,24,27,32]}
COLORS={'yellow':(0,165,255),'surface':(245,130,20),'top':(230,60,255),'F3':(50,245,50)}
REASONS={'outer_flange_not_confirmed':'黄色外轮缘证据不足',
 'd_not_bracketed_by_current_interface_evidence':'计数列附近缺少双侧边界支持',
 'count_neighborhood_surface_incomplete':'计数列附近卷面提取不完整',
 'local_edge_residual_too_large':'接触边界局部残差过大',
 'incomplete_top_interface_support':'卷面顶部边界支持不足',
 'unknown_or_excluded_region_on_count_segment':'计数段有遮挡或未知区域',
 'active_y_above_edge':'活动绳交点越过当前卷面边界',
 'edge_endpoint_far_from_observed_grooves':'边界端缺少邻近沟槽',
 'active_endpoint_far_from_observed_grooves':'活动绳端缺少邻近沟槽',
 'observed_gap_too_large':'沟槽观测间隙过大',
 'unsupported_count_gap_gt_1_6_pitch':'沟槽漏检，计数段间隙过大',
 'active_phase_bracket_has_unobserved_large_gap':'活动绳交点附近沟槽支持不足',
 'edge_phase_bracket_has_unobserved_large_gap':'卷面边界附近沟槽支持不足',
 'local_pitch_not_confirmed_by_two_observed_intervals':'可用间隔不足，无法确认绳距',
 'sub_pitch_competing_grooves_after_deduplication':'沟槽过密，存在竞争检测结果',
 'pitch_support_insufficient':'绳距支持不足'}

def read_mask(path):return cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
def draw_dashed(a,p0,p1,col,thick=2,step=22):
    p0=np.asarray(p0,float);p1=np.asarray(p1,float);length=np.linalg.norm(p1-p0)
    if length<1:return
    for z in np.arange(0,length,step):
        q=p0+(p1-p0)*z/length;v=p0+(p1-p0)*min(z+step*.55,length)/length
        cv2.line(a,tuple(np.rint(q).astype(int)),tuple(np.rint(v).astype(int)),col,thick)

def source_overlay(image,r,run):
    a=image.copy();region=r['region'];good=region['valid']
    # Filled semantic regions shown only when accepted; failed proposals remain
    # sparse outlines in the inset for diagnosis, not a valid extraction.
    for name in ['yellow','surface']:
        mask=read_mask(run/r['masks'][name]['path'])>0
        if good:a[mask]=(a[mask]*.90+np.array(COLORS[name])*.10).astype(np.uint8)
        cc,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(a,cc,-1,COLORS[name] if good else (100,100,100),3)
    for x,y in region.get('top_observed_xy',[]):
        cv2.circle(a,(round(x),round(y)),3,COLORS['top'] if good else (100,100,100),-1)
    for x,y in region.get('outer_rim',{}).get('points_xy',[]):cv2.circle(a,(round(x),round(y)),2,COLORS['yellow'],-1)
    if r['F3_called']:
        green=np.zeros(image.shape[:2],np.uint8)
        for line in r['F3']['raw_observed_polylines_xy']:
            if len(line)>1:cv2.polylines(green,[np.rint(line).astype(np.int32)],False,255,2)
        a[(green>0)&mask]=COLORS['F3']
    d=2238.7;cv2.line(a,(round(d),1750),(round(d),2999),(0,255,255),3)
    tr=r['s1'];abc=tr.get('current_line_abc')
    if tr.get('valid') and abc:
        aa,b,c=abc
        if abs(b)>.01:draw_dashed(a,[0,-c/b],[4095,-(aa*4095+c)/b],(255,255,255),3)
        pts=np.asarray(tr.get('line_points_xy',[]))
        for p in pts[::3]:cv2.circle(a,tuple(np.rint(p).astype(int)),3,(255,255,0),-1)
        y=r['intersection_y_px']
        if y is not None and 0<=y<3000:cv2.circle(a,(round(d),round(y)),12,(255,255,255),3)
    edge=region.get('edge_y')
    if edge is not None:
        cv2.circle(a,(round(d),round(edge)),10,COLORS['top'] if good else (100,100,100),3)
    return a

def render(image,r,run):
    src=source_overlay(image,r,run);a=np.full((1080,1920,3),(22,18,15),np.uint8)
    # Larger process-area view plus original full-frame inset.
    a[:670,:1280]=cv2.resize(src[1570:3000,0:3000],(1280,670))
    a[700:1080,:520]=cv2.resize(src,(520,380))
    # x=d detail: original 320 x 350 crop, with unaltered coordinate labels.
    edge=r['region'].get('edge_y');y0=max(1850,min(2650,round(edge or 2335)-70));x0=2079
    a[700:1080,540:1200]=cv2.resize(src[y0:y0+350,x0:x0+320],(660,380))
    img=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(img)
    def text(x,y,s,n=24,fill=(225,230,238)):d.text((x,y),s,font=fonts[n],fill=fill)
    text(18,12,'BG1 边界校验 → 可用卷面 → 01/F3 绳边观测',27)
    text(18,674,'完整原图',18);text(545,674,'x=d 计数列放大（本帧）',18)
    x=1310;y=20
    text(x,y,f"帧 {r['frame_index']:03d}/600   视频 {r['nominal_time_s']:.1f}s",27);y+=45
    good=r['region']['valid'];text(x,y,'边界门控：'+('通过' if good else '未通过 · 暂停后续检测'),24,(100,235,150) if good else (255,165,80));y+=42
    text(x,y,'d = 2238.7 px（用户临时固定值）',21);y+=40
    abc=r['s1'].get('current_line_abc')
    if abc:
        aa,b,c=abc;text(x,y,'当前 S1 直线（原图坐标）',21);y+=30
        text(x,y,f'{aa:.6f} x + {b:.6f} y',21);y+=29
        text(x,y,f'+ ({c:.3f}) = 0',21);y+=34
    active=r.get('intersection_y_px');edge=r['region'].get('edge_y')
    text(x,y,'S1交点 y = '+(f'{active:.2f} px' if active is not None else '未知'),21);y+=31
    text(x,y,'卷面接触边界 y = '+(f'{edge:.2f} px' if good and edge is not None else '未确认'),21);y+=40
    n=r['display_N']
    if n is not None:
        text(x,y,f'距上侧边界约 {n:.2f} 个绳距',27,(120,245,155))
    elif r['count_valid'] and r['F3'] and r['F3']['continuous_pitch_distance']>5:
        text(x,y,'五圈以外 · 不输出距离',27,(255,213,110))
    else:text(x,y,'剩余根数：未知',32,(255,180,100))
    y+=48;text(x,y,'单帧视觉估计；未用单调拟合掩盖误检',18);y+=40
    text(x,y,'01/F3：'+('本帧已运行' if r['F3_called'] else '已拦截，未调用'),24);y+=33
    if r['F3']:
        f=r['F3'];text(x,y,f"实际观测点 {len(f['raw_observed_points_xy'])} 个",21);y+=30
        text(x,y,f"x=d 沟槽观测 {len(f['crossings'])} 条",21);y+=30
        p=f.get('local_pitch_px');text(x,y,'本帧局部绳距：'+(f'{p:.2f} px' if p is not None else '未知'),21);y+=32
    else:y+=20
    for reason in r['reasons'][:3]:
        label=REASONS.get(reason,reason)
        for offset in range(0,len(label),27):text(x,y,label[offset:offset+27],18,(255,190,140));y+=26
    y=max(y+18,770)
    for label,col in [('橙：黄色轮缘及当前外缘证据',(255,175,40)),('紫：卷面顶部接触边界观测',(245,90,255)),
                      ('蓝：保守可用卷面 ROI',(60,180,255)),('绿：当前图像实际沟槽观测',(90,245,90)),
                      ('白虚线：S1 拟合/外推线',(240,240,240))]:
        text(x,y,label,18,col);y+=28
    text(x,966,'边界原图精修；绳边处理尺寸 2048×1500',18)
    text(x,994,'暗侧、遮挡和画外部分不宣称完整提取',18)
    text(x,1022,'离线验证版 · 未标定真实圈数或触边零点',18,(255,210,120))
    return cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='full_002');p.add_argument('--preview',type=int)
    p.add_argument('--stream',action='store_true');args=p.parse_args()
    run=ROOT/args.run;out=ROOT/'delivery';out.mkdir(exist_ok=True)
    c=cv2.VideoCapture(str(SOURCE))
    if args.preview is not None:
        i=args.preview;c.set(1,i);ok,a=c.read();assert ok
        r=json.loads((run/'frames'/f'{i:06d}.json').read_text(encoding='utf8'))
        cv2.imencode('.png',render(a,r,run))[1].tofile(str(out/f'preview_{i:03d}.png'));return
    if not args.stream:assert (run/'receipt.json').exists()
    dst=out/'BG1_boundary_gated_full.mp4'
    ff=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24',
        '-s','1920x1080','-r','10','-i','-','-an','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',str(dst)],
        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=(out/'ffmpeg.log').open('wb'))
    for i in range(601):
        ok,a=c.read();assert ok
        rp=run/'frames'/f'{i:06d}.json';started=time.monotonic()
        while True:
            try:
                r=json.loads(rp.read_text(encoding='utf8'));break
            except (FileNotFoundError,json.JSONDecodeError):
                if not args.stream or time.monotonic()-started>300:raise
                time.sleep(.25)
        assert hashlib.sha256(a.tobytes()).hexdigest()==r['source_pixel_sha256']
        frame=render(a,r,run);ff.stdin.write(frame.tobytes())
        if i in [0,60,121,163,200,240,270,300,450,600]:cv2.imencode('.png',frame)[1].tofile(str(out/f'preview_{i:03d}.png'))
        if i%100==0:print('Rendered',i,flush=True)
    c.release();ff.stdin.close();assert ff.wait()==0
    if args.stream:
        for _ in range(120):
            if (run/'receipt.json').exists():break
            time.sleep(.25)
        assert (run/'receipt.json').exists(),'Analysis did not finish successfully'
    print(dst,flush=True)
if __name__=='__main__':main()
