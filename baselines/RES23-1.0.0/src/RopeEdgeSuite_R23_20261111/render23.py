import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import concurrent.futures as cf,subprocess,cv2,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from common23 import *
FONTS={}
def font(n):
    if n not in FONTS:FONTS[n]=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    return FONTS[n]
BG='#111b25'

def panel(old,row,item):
    im=Image.fromarray(cv2.cvtColor(old,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
    d.rectangle((0,0,1919,64),fill=BG)
    d.text((24,16),'RopeEdgeSuite R23 · 根数周期 / 两周期施力参考 / 图像核验',font=font(28),fill='white')
    d.text((1450,22),f"{row['time_s']:05.1f}s  {row['frame']+1}/{item['planned_frames']}",font=font(27),fill='white')
    d.rectangle((0,910,839,1079),fill=BG);d.rectangle((840,944,1919,1079),fill=BG)
    f=row['fusion23'];clock=f.get('timeline');state=f['state']
    label='周期参考：等待近边锚点';detail='CPL1：从可信排绳进度估计周期；与轮缘纹理比较'
    phase='尚未进入周期参考阶段'
    if clock:
        label=f"周期参考施力 t2 ≈ {clock['t2']:.1f}s"
        detail=f"t0={clock['t0']:.1f}s   T={clock['T']:.2f}s   t1≈{clock['t1']:.1f}s   t2≈{clock['t2']:.1f}s"
        phase={'near_transition_reference':'近边过渡（周期推算）','new_layer_first_turn_reference':'新层首圈阶段（周期推算）',
            'motor_time_window_reference':'已到施力参考窗口，等待图像核验',
            'model_window_passed_without_image_confirmation':'参考窗口已过，当前图像尚未支持',
            'reference_expired':'周期参考已过期，等待新的近边事件',
            'image_supported_motor_candidate':'当前图像支持施力候选；物理事件仍需复核'}[state]
        if state=='image_supported_motor_candidate':
            label=f"图像支持的施力候选：{f['candidate_s']:.1f}s"
            detail=f"预测 {clock['t2']:.1f}s；{f['confirmation_s']:.1f}s 完成图像核验；T={clock['T']:.2f}s"
    elif state=='period_unavailable':
        label='周期参考：证据不足或两种周期冲突';detail='保留当前图像判断，不用时钟强行确认';phase='周期参考不可用'
    d.text((24,914),label,font=font(24),fill='#f0ca7f')
    d.text((24,950),detail,font=font(17),fill='#bed1dc')
    d.text((852,949),phase,font=font(18),fill='#8ed3e0')
    d.rectangle((24,986,1892,999),fill='#243947')
    d.rectangle((24,986,24+int(1868*(row['frame']+1)/item['planned_frames']),999),fill='#427f94')
    if clock:
        for time,color in [(clock['t0'],'#82d4c0'),(clock['t1'],'#85b6ed'),(clock['t2'],'#f2c76b')]:
            xx=24+int(1868*(time/(item['planned_frames']/item['fps'])))
            if 24<=xx<=1892:d.line((xx,981,xx,1004),fill=color,width=3)
    d.text((24,1012),'青：可见边界　蓝虚线：边界模型　绿：F3 绳沟　黄：活动绳　紫：固定 d　白虚线：动态列',font=font(18),fill='#bacbd9')
    d.text((24,1046),'t1 / t2 为两周期模型参考；图像核验另列。首次爬升、稳定入新层、首圈完成是不同事件，不能仅凭计时确认。',font=font(18),fill='#bacbd9')
    return np.asarray(im)

def process(item):
    cv2.setNumThreads(1);id=item['id'];out=ROOT/'delivery'/id
    rows=read(out/'frames.json');summary=read(out/'summary.json');src=summary['baseline_video']
    cap=cv2.VideoCapture(src);assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows)
    target=Path(summary['video']);temp=target.with_suffix('.part.mp4')
    writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080',
        '-r',str(item['fps']),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],
        stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    select={0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1}
    for e in summary['events']:
        if e['timeline']:
            for t in [e['timeline']['t0'],e['timeline']['t1'],e['timeline']['t2'],e.get('candidate_s'),e.get('confirmation_s')]:
                if t is not None:
                    for shift in [-3,0,3]:
                        i=round(t*item['fps'])+shift
                        if 0<=i<len(rows):select.add(i)
    if '111500' in id:select.update([100,200,247,249,273,276,290,300,304,311,320])
    for i,r in enumerate(rows):
        ok,old=cap.read();assert ok
        canvas=panel(old,r,item);writer.stdin.write(canvas.tobytes())
        if i in select:
            (out/'review').mkdir(exist_ok=True);cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(out/'review'/f'{i:06d}.jpg')
    assert not cap.read()[0];cap.release();writer.stdin.close();err=writer.stderr.read().decode('utf8','replace');assert writer.wait()==0,err
    temp.replace(target);summary.update(video_sha256=sha(target),baseline_video_sha256=sha(src),review_frames=sorted(select),
        render_code_sha256=sha(Path(__file__)),geometry_reused_from_R22=True)
    dump(out/'summary.json',summary);print('rendered',id,len(rows),flush=True);return summary

if __name__=='__main__':
    jobs=list(plan());jobs.sort(key=lambda x:('111500' not in x['id'],x['id']))
    with cf.ProcessPoolExecutor(3) as pool:receipt=list(pool.map(process,jobs))
    dump(ROOT/'delivery_receipt.json',receipt)
