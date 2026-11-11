"""Render exactly the BG3 class models used for counting; orange is inference."""
import argparse,json,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from inputs import ROOT,BASE,SOURCE
from render_results import REASONS,draw_dashed
FONT='C:/Windows/Fonts/msyh.ttc'
FONTS={n:ImageFont.truetype(FONT,n) for n in (18,20,22,24,27,32)}
PALETTE=[(255,100,80),(30,220,255),(255,80,255),(80,255,110),(255,220,50),(80,120,255)]
REASONS.update(uncertain_class_near_count_interval='计数区有未解决的边界类别或交点',
              S1_not_valid='活动绳直线未通过校验')

def draw_curves(a,curves):
    for c in curves:
        xy=np.rint(c['xy']).astype(np.int32)
        ss=np.asarray(c['point_states'])=='observed';edge=ss[:-1]&ss[1:]
        starts=np.r_[0,np.flatnonzero(edge[1:]!=edge[:-1])+1]
        ends=np.r_[starts[1:],len(edge)]
        for lo,hi in zip(starts,ends):
            cv2.polylines(a,[xy[lo:hi+1]],False,(45,245,45) if edge[lo] else (0,150,255),2)

def classified(image,r):
    a=image.copy();b=r['BG3']
    if b:
        for p,k in zip(b['raw_points'],b['diagnostics']['point_membership']):
            if k>=0:cv2.circle(a,tuple(np.rint(np.asarray(p[1:3])*2).astype(int)),2,PALETTE[k%len(PALETTE)],-1)
    return a

def overlay(image,r):
    a=image.copy();good=r['region_gate_passed'];b=r['BG3']
    if good:
        for name,col in [('yellow',(0,165,255)),('surface',(245,130,20))]:
            m=cv2.imdecode(np.fromfile(BASE/r['masks'][name]['path'],np.uint8),0)>0
            a[m]=(a[m]*.94+np.array(col)*.06).astype(np.uint8)
    for x,y in r['region']['top_observed_xy']:
        cv2.circle(a,(round(x),round(y)),3,(230,60,255) if good else (100,100,100),-1)
    # Orange model intervals may bridge holes in the evidence mask. They are
    # within the separately validated geometric envelope, and never relabelled green.
    if b:draw_curves(a,b['curves'])
    d=r['fixed_d_original_px'];cv2.line(a,(round(d),1750),(round(d),2999),(0,255,255),2)
    abc=r['s1'].get('current_line_abc')
    if r['s1']['valid'] and abc:
        aa,bb,c=abc
        if abs(bb)>.01:draw_dashed(a,[0,-c/bb],[4095,-(aa*4095+c)/bb],(255,255,255),2)
        xy=np.rint(r['s1'].get('line_points_xy',[])).astype(np.int32)
        if len(xy)>1:cv2.polylines(a,[xy],False,(255,255,0),2)
    edge=r['region'].get('edge_y');active=r['intersection_y_px']
    if good and edge is not None:cv2.circle(a,(round(d),round(edge)),7,(230,60,255),2)
    if active is not None:cv2.circle(a,(round(d),round(active)),7,(255,255,255),2)
    if b and edge is not None:
        for c in b['crossings']:
            if edge-25<=c['y']<=edge+150:
                col=(255,255,255) if c['status']=='image_supported_fitted_curve' else (15,145,255)
                cv2.circle(a,(round(d),round(c['y'])),3,col,-1)
                cv2.putText(a,c['boundary_id'].split('_')[-1],(round(d)+10,round(c['y'])+4),cv2.FONT_HERSHEY_SIMPLEX,.42,col,1,cv2.LINE_AA)
    return a

def render(image,r):
    src=overlay(image,r);points=classified(image,r)
    a=np.full((1080,1920,3),(22,18,15),np.uint8)
    a[:700,:1280]=cv2.resize(src[1570:3000,0:3000],(1280,700))
    a[755:1070,:430]=cv2.resize(src,(430,315))
    edge=r['region'].get('edge_y');y0=max(1850,min(2650,round(edge or 2335)-60))
    for left,im in [(445,points),(865,src)]:
        a[755:1070,left:left+405]=cv2.resize(im[y0:y0+340,2020:2450],(405,315))
    img=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));dr=ImageDraw.Draw(img)
    def txt(x,y,s,n=22,col=(225,230,238)):dr.text((x,y),s,font=FONTS[n],fill=col)
    txt(18,12,'BG3 v2：原始黄点归类 → 每类一条连续曲线',27)
    txt(10,718,'完整原图叠加',20);txt(450,718,'黄点按类别着色',20);txt(870,718,'相同位置：曲线及计数列',20)
    x=1310;y=20;b=r['BG3']
    txt(x,y,f"帧 {r['frame_index']:03d}/600    视频 {r['nominal_time_s']:.1f}s",27);y+=47
    txt(x,y,'卷面门控：'+('通过' if r['region_gate_passed'] else '未通过，暂停计数'),24);y+=42
    txt(x,y,'d = 2238.7 px（原图固定列）');y+=39
    abc=r['s1'].get('current_line_abc')
    if abc:
        aa,bb,c=abc;txt(x,y,'S1 当前直线（原图坐标）');y+=31
        txt(x,y,f'{aa:.6f} x + {bb:.6f} y');y+=31
        txt(x,y,f'+ ({c:.3f}) = 0');y+=39
    active=r['intersection_y_px']
    txt(x,y,'活动绳交点 y = '+(f'{active:.2f} px' if active is not None else '未知'));y+=32
    txt(x,y,'卷面上界 y = '+(f'{edge:.2f} px' if edge is not None and r['region_gate_passed'] else '未确认'));y+=44
    n=r['display_N']
    if n is not None:txt(x,y,f'距上侧边界约 {n:.2f} 个绳距',27,(120,245,155))
    elif r['count_valid']:txt(x,y,'五圈以外 · 不显示距离',27,(255,210,100))
    else:txt(x,y,'剩余根数：未知',32,(255,180,100))
    y+=48
    if b:
        dg=b['diagnostics']
        txt(x,y,f"黄点：{dg['raw_point_count']} 个 → {len(b['curves'])} 类");y+=32
        txt(x,y,f"已归类 {dg['classified_point_count']} / 未归类 {dg['unclassified_point_count']}");y+=32
        txt(x,y,f"连续曲线 {len(b['curves'])} 条；类内拆段 0");y+=32
        txt(x,y,f"d 交点：{b['supported_crossings']} 实据 / {b['inferred_crossings']} 插值");y+=32
        txt(x,y,f"当前局部绳距：{b['local_pitch_px']:.2f} px");y+=34
    for reason in r['reasons'][:2]:
        label=REASONS.get(reason,reason)
        for off in range(0,len(label),27):txt(x,y,label[off:off+27],18,(255,190,140));y+=25
    y=max(y+12,780)
    for label,col in [('绿：当前图像支持的曲线',(90,245,90)),('橙：该类别的模型补全部分',(255,175,40)),
        ('紫：卷面上界；黄竖线：x=d',(210,180,255)),('白虚线：S1 解析式',(235,235,210))]:
        txt(x,y,label,18,col);y+=27
    txt(x,930,'拟合范围：本类最左点至最右点',18)
    txt(x,960,'连续补线 ≠ 整段都有可靠观测',18)
    txt(x,990,'长缺口、卷面不确定处暂停计数',18)
    txt(x,1020,'离线结果；未标定真实根数与触边零点',18,(255,210,120))
    return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

def comparison(image,r,out):
    ims=(image,classified(image,r),overlay(image,r))
    x0,y0,x1,y1=1750,2260,2470,2780;w=x1-x0;h=y1-y0
    canvas=Image.new('RGB',(w*3+40,h+160),'#f1f4f8');dr=ImageDraw.Draw(canvas)
    for k,(a,title) in enumerate(zip(ims,('原图','原始黄点按类别着色','每类一条连续模型：绿=支持 / 橙=补全'))):
        xx=10+k*(w+10);dr.text((xx,16),title,font=FONTS[20],fill='#172435')
        canvas.paste(Image.fromarray(cv2.cvtColor(a[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)),(xx,60))
    dg=r['BG3']['diagnostics']
    dr.text((20,h+77),f"第 {r['frame_index']} 帧：{dg['raw_point_count']} 个原始点，{dg['classified_point_count']} 点归入 {dg['class_count']} 类；每类一条连续曲线。",font=FONTS[24],fill='#172435')
    dr.text((20,h+116),'颜色分类编号仅用于当前帧；橙线为模型插值，不能自动作为可靠计数证据。',font=FONTS[20],fill='#172435')
    canvas.save(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='pilot_001');ap.add_argument('--frames',default='127,163,200,240,270')
    args=ap.parse_args();cap=cv2.VideoCapture(str(SOURCE));out=ROOT/'delivery';out.mkdir(exist_ok=True)
    for i in map(int,args.frames.split(',')):
        cap.set(1,i);ok,a=cap.read();assert ok
        r=json.loads((ROOT/args.run/'frames'/f'{i:06d}.json').read_text(encoding='utf8'))
        assert hashlib.sha256(a.tobytes()).hexdigest()==r['source_pixel_sha256']
        cv2.imencode('.png',render(a,r))[1].tofile(str(out/f'preview_{i:03d}.png'))
        if r['BG3']:comparison(a,r,out/f'comparison_{i:03d}.png')
    cap.release()
