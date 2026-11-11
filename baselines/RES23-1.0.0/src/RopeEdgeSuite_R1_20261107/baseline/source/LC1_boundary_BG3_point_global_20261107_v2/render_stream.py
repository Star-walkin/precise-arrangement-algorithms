"""Streaming offline renderer, waits for the matching frame and final receipt."""
import time,json,hashlib,subprocess
import cv2,imageio_ffmpeg
from inputs import ROOT,SOURCE,sha,dump
from render_bg3 import render,comparison

def wait_json(path):
    start=time.monotonic()
    while True:
        try:return json.loads(path.read_text(encoding='utf8'))
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()-start>1800:raise
            time.sleep(.25)

def main():
    cv2.setNumThreads(1)
    run=ROOT/'full_001';out=ROOT/'delivery';out.mkdir(exist_ok=True)
    dst=out/'BG3_point_classes_continuous_full.mp4'
    assert not dst.exists(),'Preserve existing video'
    frozen=sha(ROOT/'render_bg3.py');cap=cv2.VideoCapture(str(SOURCE));log=(out/'ffmpeg.log').open('wb')
    ff=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24',
        '-s','1920x1080','-r','10','-i','-','-an','-c:v','libx264','-threads','2','-crf','20','-preset','fast','-pix_fmt','yuv420p',str(dst)],
        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
    for i in range(601):
        r=wait_json(run/'frames'/f'{i:06d}.json');ok,a=cap.read();assert ok
        assert hashlib.sha256(a.tobytes()).hexdigest()==r['source_pixel_sha256']
        frame=render(a,r);ff.stdin.write(frame.tobytes())
        if i in (0,60,127,163,200,240,270,300,450,600):
            cv2.imencode('.png',frame)[1].tofile(str(out/f'preview_{i:03d}.png'))
            if r['BG3'] and i in (127,163,200,240,270):comparison(a,r,out/f'comparison_{i:03d}.png')
        if i%50==0:
            dump(out/'render_progress.json',dict(rendered=i+1,planned=601))
            print('Rendered',i,flush=True)
    cap.release();ff.stdin.close();assert ff.wait()==0;log.close()
    receipt=wait_json(run/'receipt.json');assert receipt['status']=='completed'
    v=cv2.VideoCapture(str(dst));fps=v.get(cv2.CAP_PROP_FPS);n=0
    while True:
        ok,a=v.read()
        if not ok:break
        assert a.shape==(1080,1920,3);n+=1
    v.release();assert n==601 and abs(fps-10)<1e-6
    assert frozen==sha(ROOT/'render_bg3.py')
    dump(out/'video_receipt.json',dict(status='completed',frames=n,fps=fps,duration_s=n/fps,
        sha256=sha(dst),source_pixels_bound=True,rendered_curve_source='BG3.curves',
        count_crossing_source='the same BG3.curves',renderer_sha256=frozen,stream_runner_sha256=sha(__file__)))
    print('VIDEO COMPLETED',dst,flush=True)
if __name__=='__main__':main()
