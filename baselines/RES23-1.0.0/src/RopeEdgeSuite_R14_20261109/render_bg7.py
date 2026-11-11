import time,subprocess,argparse
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from common import *
RUN=ROOT/'boundary_final';OUT=ROOT/'delivery'
def font(n):return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
def line(im,pts,color,width=4):
    if len(pts)>1:cv2.polylines(im,[np.rint(pts).astype(np.int32)],False,color,width,cv2.LINE_AA)
def annotate(im,n,t):
    im=im.copy()
    if n['fit_valid']:
        # Entire curve is global; orange portions lack direct current support.
        line(im,n['top_curve_xy'],(0,180,255),5)
        for p in n['observed_segments']:line(im,p,(255,215,25),5)
        layer=im.copy()
        for p in n['foreground_polygons']:cv2.fillPoly(layer,[np.rint(p).astype(np.int32)],(20,20,245))
        im=cv2.addWeighted(im,.82,layer,.18,0)
        for p in n['top_observed_xy']:cv2.circle(im,tuple(np.rint(p).astype(int)),3,(255,240,100),-1)
    if t['valid']:
        for p in t['supported_segments_xy']:line(im,p,(70,255,70),4)
    cv2.line(im,(round(D),1900),(round(D),2700),(220,60,210),3)
    return im
def picture(bgr,size):return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize(size,Image.Resampling.LANCZOS)
def panel(im,n,r,old):
    p=Image.new('RGB',(1920,1080),'#111b25');d=ImageDraw.Draw(p);t=r['S1_tracker_full'];i=n['frame']
    d.text((24,17),'RES-1.4 · SG2 + BG7 遮挡边界拟合',font=font(36),fill='white')
    d.text((1510,22),f'{i/10:05.1f}s  |  {i}/599',font=font(30),fill='white')
    d.text((24,73),'113745 / 左相机；同帧跟踪 + 当前可见锚点 + 因果形状约束',font=font(23),fill='#bacbd9')
    ann=annotate(im,n,t);p.paste(picture(ann,(800,586)),(24,132))
    before=im.copy();line(before,old['region'].get('top_curve_xy',[]),(255,215,25),5)
    d.text((852,112),'R11：原蓝色边界（含误收的局部边缘）',font=font(22),fill='#bacbd9')
    p.paste(picture(before[1950:2450,900:2700],(1040,289)),(852,148))
    d.text((852,452),'BG7：先排除干扰，再拟合整体曲线',font=font(22),fill='white')
    p.paste(picture(ann[1950:2450,900:2700],(1040,289)),(852,489))
    status='当前活动绳支撑检查通过' if t['valid'] else '当前活动绳测量不足'
    d.text((24,748),status,font=font(25),fill='#7edeb6' if t['valid'] else '#edbd6c')
    if t['valid']:
        abc=t['current_line_abc'];d.text((24,785),f'{abc[0]:+.5f}x {abc[1]:+.5f}y {abc[2]:+.2f}=0',font=font(21),fill='#bacbd9')
    d.text((24,829),f"R11 同帧跟踪：{'通过' if old['s1']['valid'] else '不足'}",font=font(22),fill='#bacbd9')
    if n['fit_valid']:
        mode='当前锚点约束的整体边界' if n.get('d_locally_observed') else 'd 处为模型推断，非直接观测'
        d.text((852,801),mode,font=font(25),fill='#edbd6c')
        d.text((852,839),f"y(d)={n['edge_y']:.1f}px；敏感度 ±{n['edge_uncertainty_px']:.1f}px（开发指标）",font=font(22),fill='#bacbd9')
        c=n['fit_coefficients_ascending'];formula=f'y={c[0]:.2f}{c[1]:+.2f}u{c[2]:+.2f}u²{c[3]:+.2f}u³{c[4]:+.2f}u⁴'
        d.text((852,877),formula,font=font(20),fill='#bacbd9')
        d.text((852,907),'u=(x−1720.32)/1228.80；参考 d=2238.7px',font=font(20),fill='#bacbd9')
    else:
        labels={'initial_shape_evidence_accumulating':'初始化：累计可靠形状',
            'outline_at_d_model_sensitive':'其他可见点不足以稳定推断 d 处边界',
            'current_anchor_fit_residual_excessive':'可见锚点与整体形状不一致',
            'unoccluded_current_anchor_support_insufficient':'未遮挡的当前锚点不足'}
        d.text((852,805),'边界拟合暂未通过，不沿用错误蓝线',font=font(24),fill='#ee9c95')
        d.text((852,850),labels.get(n['reasons'][0],'边界证据不足'),font=font(22),fill='#bacbd9')
    if not t['valid']:ecw='ECW：当前绳轴缺失，不能判定贴边'
    elif n.get('d_foreground_occluded'):ecw='ECW：计数线受遮挡；贴边待核验'
    elif not n['fit_valid']:ecw='ECW：边界模型不稳定；贴边待核验'
    elif not n.get('d_locally_observed'):ecw='ECW：边界为拟合估计；贴边待核验'
    elif not r['ECW'].get('rope',{}).get('valid'):ecw='ECW：绳轴已恢复；局部贴边证据不足'
    else:ecw='ECW：自由绳段有观测，仍需核验入卷接触点'
    d.text((24,874),ecw,font=font(21),fill='#edbd6c')
    d.text((24,940),'青：可见锚点支持的边界　橙：同一曲线的推断段　绿：活动绳实测段　红：剔除的宽亮带　紫：参考 d',font=font(22),fill='#bacbd9')
    d.rectangle((24,983,1892,1000),fill='#243947');d.rectangle((24,983,24+int(1868*(i+1)/600),1000),fill='#427f94')
    d.text((24,1023),'本视频验证边界与跟踪；未重算全片 F3/根数。拟合轮廓不代表已看见遮挡后的实体接触。',font=font(22),fill='#edbd6c')
    return np.asarray(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');a=ap.parse_args()
    OUT.mkdir(exist_ok=True);cap=cv2.VideoCapture(read(RUN/'binding.json')['source']);ids=[100,336,370,388,400,450,599]
    if a.preview:
        for i in ids:
            cap.set(1,i);ok,im=cap.read();assert ok
            Image.fromarray(panel(im,record(i,RUN),record(i),record(i,RUN11))).save(OUT/f'preview_{i:03d}.jpg',quality=95)
        cap.release();return
    path=OUT/'RES14_SG2_BG7_113745_boundary_full.mp4';start=time.perf_counter()
    writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','10','-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for i in range(600):
        ok,im=cap.read();assert ok;n=record(i,RUN);assert pixelsha(im)==n['source_pixel_sha256']
        rgb=panel(im,n,record(i),record(i,RUN11));writer.stdin.write(rgb.tobytes())
        if i in ids:Image.fromarray(rgb).save(OUT/f'preview_{i:03d}.jpg',quality=95)
        if i%100==0:print('render',i,round(time.perf_counter()-start,1),flush=True)
    cap.release();writer.stdin.close();err=writer.stderr.read();assert writer.wait()==0,err
    cap=cv2.VideoCapture(str(path));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1080,1920,3);n+=1
    fps=cap.get(5);cap.release();assert n==600 and fps==10
    dump(OUT/'video_receipt.json',dict(path=str(path),decoded_frames=n,fps=fps,sha256=sha(path),elapsed_s=time.perf_counter()-start))
if __name__=='__main__':main()
