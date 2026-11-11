"""Refresh state/motor overlays on the verified R21 frame-aligned geometry render."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,collections,concurrent.futures as cf,subprocess,cv2,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from common22 import *
FONTS={}
def font(n):
    if n not in FONTS:FONTS[n]=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    return FONTS[n]
BG='#111b25'
NAMES={'unknown':'当前证据不足','steady':'方向判稳中','approaching':'正在靠近边缘','departing':'正在离开边缘','near':'近边停留（视觉）'}

def panel(old,row,item):
    canvas=Image.fromarray(cv2.cvtColor(old,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(canvas)
    d.rectangle((0,0,1919,64),fill=BG)
    d.text((24,16),'RopeEdgeSuite R22 · 稳定运动趋势 / 新层首圈施力时点',font=font(29),fill='white')
    d.text((1450,22),f"{row['time_s']:05.1f}s  {row['frame']+1}/{item['planned_frames']}",font=font(27),fill='white')
    d.rectangle((840,874,1919,949),fill=BG)
    d.rectangle((0,910,839,1079),fill=BG);d.rectangle((840,950,1919,1079),fill=BG)
    s=row['state22'];m=row['motor22'];name=NAMES[s['state']]
    d.text((852,881),'状态：'+name,font=font(25),fill='#7edeb6' if s['state']=='approaching' else '#edbd6c')
    evidence={'current':'当前测量支持','history_hold':f"历史保持 {s.get('age_s') or 0:.1f}s（本帧未确认）",'unavailable':'证据中断，等待重捕获'}[s['evidence_mode']]
    direction={'approaching':'靠近','departing':'离开','unknown':'待判'}[s['motion']]
    d.text((852,918),f'运动趋势：{direction}  |  {evidence}',font=font(19),fill='#bacbd9')
    state=m['state'];target=m.get('target')
    if state=='waiting_layer_start':label='施力时点：等待新层起点证据';detail='MFT1：识别首圈完成 / 第二圈即将开始的时刻'
    elif state=='layer_start_unverified':label='施力时点：新层起点尚未核实';detail='已见首行重叠；等待活动绳姿态变化支持'
    elif state=='period_unresolved':label='施力时点：转动周期证据不足';detail='不输出仅凭时间外推的施力标记'
    elif state=='cycle_return_candidate':
        t=target['refined_time_s'];lo=min(target['interval_s'][0],target['refined_interval_s'][0]);hi=max(target['interval_s'][1],target['refined_interval_s'][1])
        label=f'首圈完成候选 / 施力候选：{t:.1f}s'
        detail=f"范围 {lo:.1f}–{hi:.1f}s；{m['confirmation_s']:.1f}s 完成纹理回归复核"
    elif target:
        label=f"预计首圈完成 / 施力候选：{target['time_s']:.1f}s"
        detail=f"范围 {target['interval_s'][0]:.1f}–{target['interval_s'][1]:.1f}s；轮缘周期 {m['period_s']:.1f}s"
        if state=='cycle_evidence_pending':detail='预测窗口已过，尚无足够当前证据完成复核'
    else:label='施力时点：证据不足';detail='等待更多当前视频证据'
    d.text((24,914),label,font=font(23),fill='#eec778')
    d.text((24,949),detail,font=font(17),fill='#bacbd9')
    c=row['contact']['state']
    clabel={'unavailable':'本帧贴边证据不足','separated':'本帧投影分离','near_edge_approaching':'本帧近边',
        'near_edge_uncertain':'本帧近边，实体接触未确认','projection_conflict':'本帧投影存在模型冲突',
        'visual_contact_candidate':'本帧视觉贴边候选'}[c]
    d.text((852,949),clabel+'；F3 / d / 距离沿用已核验测量',font=font(17),fill='#bacbd9')
    d.rectangle((24,986,1892,999),fill='#243947')
    d.rectangle((24,986,24+int(1868*(row['frame']+1)/item['planned_frames']),999),fill='#427f94')
    if state=='cycle_return_candidate':
        x=24+int(1868*target['refined_time_s']/(item['planned_frames']/item['fps']))
        d.line((x,980,x,1006),fill='#ffd369',width=3)
    d.text((24,1012),'青：可见边界　蓝虚线：边界模型　绿：F3 绳沟　黄：活动绳　紫：固定 d　白虚线：动态列',font=font(18),fill='#bacbd9')
    d.text((24,1046),'首圈完成标记是视频复核候选；新层起点与转满一圈仍需现场确认。历史保持不产生新的贴边或施力证据。',font=font(18),fill='#bacbd9')
    return np.asarray(canvas)

def process(item):
    cv2.setNumThreads(1);id=item['id'];out=ROOT/'delivery'/id
    rows=read(out/'frames.json');summary=read(out/'summary.json');src=summary['baseline_video']
    target=Path(summary['video']);tmp=target.with_suffix('.part.mp4');cap=cv2.VideoCapture(src)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows)
    writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080',
        '-r',str(item['fps']),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],
        stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    select={0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1}
    if '111500' in id:select.update([10,35,50,100,190,200,246,249,250,259,270,273,278,280,290,300,489,507])
    if '111143' in id:select.update([120,140,160,180,200])
    last=None
    for i,r in enumerate(rows):
        ok,old=cap.read();assert ok
        canvas=panel(old,r,item);writer.stdin.write(canvas.tobytes())
        if i in select:
            (out/'review').mkdir(exist_ok=True);cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(out/'review'/f'{i:06d}.jpg')
        last=r
    assert not cap.read()[0];cap.release();writer.stdin.close();err=writer.stderr.read().decode('utf8','replace');assert writer.wait()==0,err
    tmp.replace(target);summary.update(video_sha256=sha(target),baseline_video_sha256=sha(src),review_frames=sorted(select),
        render_method='frame_aligned_R21_geometry_reuse_with_R22_temporal_overlays',render_code_sha256=sha(Path(__file__)))
    dump(out/'summary.json',summary);print('rendered',id,len(rows),flush=True);return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--id');a=ap.parse_args()
    jobs=[i for i in plan() if not a.id or a.id in i['id']]
    jobs.sort(key=lambda x:('111500' not in x['id'],x['id']))
    with cf.ProcessPoolExecutor(a.workers) as pool:receipt=list(pool.map(process,jobs))
    dump(ROOT/'delivery_receipt.json',receipt)
