"""Render all current outputs; observed segments and inference stay distinct."""
from pathlib import Path
import sys,time,json,hashlib,argparse
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'RopeEdgeSuite_R11_20261108'))
import render_bg5 as render

original_annotate=render.annotate
def annotate(im,r):
    rr=dict(r);rr['s1']=dict(r['s1'],valid=False)
    a=original_annotate(im,rr);s=r['s1']
    if not s['valid']:return a
    segments=r['S1_tracker_full']['supported_segments_xy']
    for p in segments:
        if len(p)>1:cv2.polylines(a,[np.rint(p).astype(np.int32)],False,(80,255,80),4,cv2.LINE_AA)
    abc=s['current_line_abc'];pts=np.asarray(s['line_points_xy']);d=2238.7
    gaps=[(float(np.asarray(p)[-1,0]),float(np.asarray(q)[0,0])) for p,q in zip(segments[:-1],segments[1:])]
    if pts[:,0].max()<d:gaps.append((float(pts[:,0].max()),d))
    for lo,hi in gaps:
        xs=np.arange(lo,hi+1,12.)
        for j in range(1,len(xs),2):
            xx=xs[j-1:j+1];q=np.c_[xx,-(abc[0]*xx+abc[2])/abc[1]]
            cv2.polylines(a,[np.rint(q).astype(np.int32)],False,(0,160,255),3,cv2.LINE_AA)
    return a
render.annotate=annotate

def panel(im,r,baseline,n):
    rr=dict(r);rr['comparison_baseline']=baseline
    rgb=render.panel(im,rr,'dual_20260923_113745',n)
    p=Image.fromarray(rgb);d=ImageDraw.Draw(p)
    d.rectangle((0,0,1210,65),fill='#111b25')
    render.text(d,(24,16),'RES-1.2 · SG2 活动绳连续跟踪优化',36)
    d.rectangle((14,774,827,841),fill='#111b25')
    s=r['s1'];track=r['S1_tracker_full'];diag=track['diagnostics']
    if s['valid']:
        render.text(d,(24,779),'活动绳：当前图像支撑及连续性检查通过',22)
        a=s['current_line_abc'];render.text(d,(24,812),f'{a[0]:+.5f}x {a[1]:+.5f}y {a[2]:+.2f}=0',20)
    else:
        labels={'SG2_identity_confirmation_pending':'重捕获候选：等待连续帧确认',
            'SG2_short_loss_search_memory_retained':'短时中断：保留搜索记忆，当前无测量',
            'SG2_reacquisition_required':'重捕获中：当前证据不足'}
        render.text(d,(24,779),labels.get(diag['accept_reason'],diag['accept_reason']),22,'#e6bb6b')
        render.text(d,(24,812),'本帧不沿用历史解析式作为当前测量',20,'#b8c7d3')
    d.rectangle((14,922,1900,969),fill='#111b25')
    render.text(d,(24,930),f"青色：卷面边界　绿色：实测支撑　橙色虚线：插值/外推　旧版活动绳：{'通过' if baseline['s1_valid'] else '不足'}",20,'#b8c7d3')
    d.rectangle((14,1010,1900,1079),fill='#111b25')
    render.text(d,(24,1025),'当前帧重算活动绳、F3及计数；短时记忆仅用于搜索。固定 d 沿用2238.7 px；根数与实体接触尚无人工真值。',20,'#e6bb6b')
    return np.asarray(p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview');ap.add_argument('--live',action='store_true');a=ap.parse_args()
    run=ROOT/'runs/dual_20260923_113745';meta=render.read(run/'binding.json');n=meta['processed_frames_planned']
    old=render.read(ROOT.parent/'RopeEdgeSuite_R11_20261108/runs/dual_20260923_113745_refined/compact.json')
    out=run/'delivery';out.mkdir(exist_ok=True);cap=cv2.VideoCapture(meta['source'])
    if a.preview:
        for i in map(int,a.preview.split(',')):
            cap.set(1,i);ok,im=cap.read();assert ok;r=render.record(run,i)
            Image.fromarray(panel(im,r,old[i],n)).save(out/f'preview_{i:03d}.jpg',quality=95)
        cap.release();return
    dest=out/'RES12_SG2_113745_full.mp4';enc=render.writer(dest);start=time.perf_counter()
    snapshots={0,10,20,50,100,150,200,250,300,310,315,316,336,345,369,370,400,404,450,500,526,550,599}
    for i in range(n):
        deadline=time.monotonic()+600
        while not (run/'frames'/f'{i:06d}.json.gz').exists():
            assert a.live and time.monotonic()<deadline,'Waiting for analysis timed out';time.sleep(.25)
        ok,im=cap.read();assert ok;r=render.record(run,i)
        assert hashlib.sha256(im.tobytes()).hexdigest()==r['source_pixel_sha256']
        rgb=panel(im,r,old[i],n);enc.stdin.write(rgb.tobytes())
        if i in snapshots:Image.fromarray(rgb).save(out/f'preview_{i:03d}.jpg',quality=95)
        if i%50==0:print('rendered',i,round(time.perf_counter()-start,1),flush=True)
    cap.release();enc.stdin.close();err=enc.stderr.read().decode(errors='replace');assert enc.wait()==0,err[-2000:]
    check=cv2.VideoCapture(str(dest));count=0
    while True:
        ok,im=check.read()
        if not ok:break
        assert im.shape==(1080,1920,3);count+=1
    fps=check.get(5);check.release();assert count==n and fps==10
    (out/'video_receipt.json').write_text(json.dumps(dict(path=str(dest),decoded_frames=count,fps=fps,
        sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),elapsed_s=time.perf_counter()-start),indent=2))

if __name__=='__main__':main()
