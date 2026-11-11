"""Source-bound full-rate review of the fixed-d current-distance trial."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def render(analysis='analysis'):
    data=ROOT/analysis;receipt=read(data/'receipt.json')
    if receipt['status']!='completed':raise RuntimeError('Analysis is incomplete')
    out=ROOT/('video' if analysis=='analysis' else 'video_'+analysis);out.mkdir(exist_ok=True)
    source=Path(receipt['source_video']);cap=cv2.VideoCapture(str(source))
    count=receipt['processed_frames'];fps=receipt['source_fps'];target=out/'fixed_d2238p7_full.mp4'
    if analysis=='analysis' and count!=receipt['source_frames']:
        raise RuntimeError('A full-video render requires every source frame')
    ff=imageio_ffmpeg.get_ffmpeg_exe();log=(out/'encoder.log').open('wb')
    enc=subprocess.Popen([ff,'-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1760x938',
        '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','21',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=log,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    fonts={s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s) for s in (17,20,25,38)}
    first_inside=receipt.get('first_inside_confirmation_frame');saved=[]
    requested={0,100,150,200,300,400,500,count-1}
    if first_inside is not None:requested.add(first_inside)
    unknown_reason_names={
        'current_S1_not_valid':'当前 S1 无有效直线',
        'current_F3_distance_not_valid_candidate_ignored':'当前绳边/端点支持不充分',
        'current_S1_intersection_unavailable_or_nonfinite':'当前活动绳交点未知',
    }
    for i in range(count):
        ok,bgr=cap.read()
        if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(i+1))>.1:raise RuntimeError('Source decode mismatch')
        r=read(data/'frames'/f'frame_{i:06d}.json')
        if hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()!=r['source_pixel_sha256']:
            raise RuntimeError('Displayed source pixels differ from analyzed pixels')
        track=r['S1_tracker'];surf=r['surface'];pred=r['prediction'];y=r['intersection_y_px'];d=2238.7
        full=bgr.copy();height,width=bgr.shape[:2]
        abc=track.get('current_line_abc') if track.get('valid') else None
        if abc is not None:
            A,B,C=abc
            for x in np.linspace(0,min(width-1,d+80),210):
                yy=-(A*x+C)/B
                if 0<=yy<height:cv2.circle(full,(round(x),round(yy)),3,(255,225,40),-1)
            pts=np.asarray(track.get('line_points_xy',[]),float).reshape(-1,2)
            for p in pts[::3]:cv2.circle(full,tuple(np.rint(p).astype(int)),3,(255,220,40),-1)
        for segment in surf.get('boundary_polylines_xy',[]):
            pts=np.rint(np.asarray(segment,float)).astype(np.int32)
            if len(pts)>1:cv2.polylines(full,[pts],False,(70,235,65),3)
        cv2.line(full,(round(d),1800),(round(d),height-1),(20,230,255),4)
        edge=surf.get('edge_y');confirmed=surf.get('edge_current_frame_confirmed') is True
        if edge is not None:
            cv2.line(full,(round(d-90),round(edge)),(round(d+90),round(edge)),(20,160,255) if confirmed else (130,130,130),6)
        if y is not None and 0<=y<height:
            cv2.circle(full,(round(d),round(y)),14,(255,255,255),4)
            if pred['emit_distance_estimate'] and edge is not None:
                cv2.arrowedLine(full,(round(d-25),round(edge)),(round(d-25),round(y)),(255,255,255),4,tipLength=.08)
        for crossing in surf.get('crossing_records',[]):
            cy=crossing['y']
            if edge is not None and y is not None and min(edge,y)<=cy<=max(edge,y):
                cv2.circle(full,(round(d),round(cy)),6,(245,80,230),-1)
        canvas=np.full((938,1760,3),20,np.uint8);canvas[:,:1280]=cv2.resize(full,(1280,938))
        # Zoom original-resolution annotated pixels, preserving observed gaps.
        top=max(0,int((edge if edge is not None else 2150)-65))
        bottom=min(height,max(top+310,int((y if y is not None else top+300)+90)))
        left=max(0,round(d)-225);right=min(width,round(d)+225)
        crop=full[top:bottom,left:right]
        if crop.size:canvas[555:855,1295:1745]=cv2.resize(crop,(450,300))
        im=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im)
        draw.rectangle((8,8,1200,65),fill=(20,20,20))
        draw.text((20,17),'固定 d=2238.7px 试运行｜进入≤5圈时显示距离｜离线回放',font=fonts[25],fill=(255,220,170))
        x=1295;yy=18
        def text(t,color=(235,235,235),size=20,step=31):
            nonlocal yy
            draw.text((x,yy),t,font=fonts[size],fill=color);yy+=step
        text(f'左相机'+('完整视频' if count==receipt['source_frames'] else '片段试跑')+f'  {i:03d}/{count-1}',size=25,step=42)
        text(f'视频时间 {r["source_nominal_time_s"]:.1f}s  |  {fps:g} FPS')
        text('固定列：原图 x=2238.7 像素')
        text('d 为用户暂定值，尚非标定结果',color=(255,205,100),size=17)
        region=pred['current_region']
        if region=='inside':
            text('已进入 5 圈以内',color=(120,245,120),size=25,step=40)
            text(f'剩余约 {pred["current_N"]:.2f} 圈',color=(120,245,120),size=38,step=55)
            text('连续绳距估计；不取整',size=17)
        elif region=='outside':
            text('距离大于 5 圈',color=(255,215,100),size=25,step=40)
            text('距离预测未启用',color=(255,215,100),size=25,step=45)
            text('当前有效测量仅用于区域触发',size=17)
        else:
            text('当前距离未知',color=(255,150,120),size=25,step=40)
            text('本帧不输出预测数值',color=(255,150,120),size=25,step=45)
            text(unknown_reason_names.get(pred['reason'],'当前输入未通过一致性检查'),size=17)
        text('当前 S1：'+('有效' if track.get('valid') else '未知'),size=17)
        text('当前 F3：'+('计数有效' if surf.get('count_valid') else '支持不足'),size=17)
        text('S1 解析式（原图归一化）：',size=17)
        if abc is not None:
            A,B,C=abc;text(f'{A:+.5f}x {B:+.5f}y {C:+.2f}=0',size=17)
        else:text('ABC = 未知；不沿用旧直线',color=(255,170,140),size=17)
        text('绿：F3实测绳边；青：当前S1',size=17)
        text('黄：固定d；橙：当前边界',size=17)
        text('白圈：S1直线与固定列的交点',size=17)
        text('未知帧不显示上一次的距离',color=(255,205,100),size=17)
        draw.text((1295,868),'原图 ROI 放大；真实缺口保留',font=fonts[17],fill=(235,235,235))
        draw.text((1295,899),'按原帧率回放；算法非实时速度',font=fonts[17],fill=(255,205,100))
        rendered=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR);enc.stdin.write(rendered.tobytes())
        if i in requested:
            path=out/f'frame_{i:06d}.jpg';cv2.imencode('.jpg',rendered,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(path);saved.append(str(path))
    cap.release();enc.stdin.close();rc=enc.wait();log.close()
    if rc:raise RuntimeError('Encoder failed; inspect encoder.log')
    verify=cv2.VideoCapture(str(target));decoded=0
    while True:
        ok,image=verify.read()
        if not ok:break
        if image.shape!=(938,1760,3):raise RuntimeError('Output shape mismatch')
        decoded+=1
    verify.release()
    if decoded!=count:raise RuntimeError(f'Decoded {decoded} frames, expected {count}')
    result=dict(status='completed',video=str(target),video_sha256=sha(target),decoded_frames=decoded,
        source_frames=count,source_pixel_hashes_matched_for_every_frame=True,frame_stride=1,
        output_fps=fps,output_shape=[938,1760],first_inside_confirmation_frame=first_inside,
        fixed_d_original_px=2238.7,first_inside_is_confirmation_not_true_crossing=True,
        outside_and_unknown_display_distance=False,renderer_uses_no_future_smoothing=True,
        analysis_receipt_sha256=sha(data/'receipt.json'),renderer_sha256=sha(Path(__file__)),previews=saved)
    (out/'receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis-dir',default='analysis')
    render(p.parse_args().analysis_dir)
