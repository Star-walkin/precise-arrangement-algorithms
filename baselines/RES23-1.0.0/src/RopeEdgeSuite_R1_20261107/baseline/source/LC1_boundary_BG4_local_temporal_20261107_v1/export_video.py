"""Final full video export from fixed-reference causal replay."""
import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import time,hashlib,subprocess,cv2,imageio_ffmpeg
from bg4_common import *
from render_bg4 import render

def main():
    cv2.setNumThreads(1);run=ROOT/'full_002';out=ROOT/'delivery_final';out.mkdir(exist_ok=True)
    dst=out/'BG4_local_temporal_full.mp4';assert not dst.exists()
    frozen={str(ROOT/n):sha(ROOT/n) for n in ('export_video.py','render_bg4.py')}
    cap=cv2.VideoCapture(str(SOURCE));log=(out/'ffmpeg.log').open('wb');start=time.perf_counter()
    ff=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24',
        '-s','1920x1080','-r','10','-i','-','-an','-c:v','libx264','-threads','2','-crf','20','-preset','fast','-pix_fmt','yuv420p',str(dst)],
        stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
    for i in range(601):
        path=run/'frames'/f'{i:06d}.json';waiting=time.monotonic()
        while not path.exists():
            if time.monotonic()-waiting>1800:raise TimeoutError(path)
            time.sleep(.2)
        r=read(path);ok,im=cap.read();assert ok and hashlib.sha256(im.tobytes()).hexdigest()==r['source_pixel_sha256']
        result=render(im,r);ff.stdin.write(result.tobytes())
        if i in (0,60,126,127,163,200,240,270,300,450,600):cv2.imencode('.png',result)[1].tofile(str(out/f'preview_{i:03d}.png'))
        if i%50==0:print('Rendered',i,flush=True);dump(out/'render_progress.json',dict(rendered=i+1,planned=601))
    cap.release();ff.stdin.close();assert ff.wait()==0;log.close();assert all(sha(p)==h for p,h in frozen.items())
    cap=cv2.VideoCapture(str(dst));fps=cap.get(cv2.CAP_PROP_FPS);n=0
    while True:
        ok,a=cap.read()
        if not ok:break
        assert a.shape==(1080,1920,3);n+=1
    cap.release();assert n==601 and abs(fps-10)<1e-6
    assert read(run/'receipt.json')['status']=='completed'
    dump(out/'video_receipt.json',dict(status='completed',decoded_frames=n,fps=fps,duration_s=n/fps,elapsed_s=time.perf_counter()-start,
        source_pixels_bound=True,sha256=sha(dst),renderer_code_sha256=frozen,curves_and_count_share_same_models=True))
    print('VIDEO COMPLETE',flush=True)
if __name__=='__main__':main()
