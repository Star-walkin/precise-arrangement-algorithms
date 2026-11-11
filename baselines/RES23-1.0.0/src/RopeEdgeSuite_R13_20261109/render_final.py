import time
import cv2
from PIL import Image,ImageDraw
from common import *
from render_result import panel,writer,text

def main():
    cv2.setNumThreads(1);run=ROOT/'runs/dual_20260923_113745_final';out=run/'delivery';out.mkdir(exist_ok=True)
    binding=read(run/'binding.json');cap=cv2.VideoCapture(binding['source']);path=out/'RES13_BG63_GV1_113745_full.mp4'
    enc=writer(path);tick=time.perf_counter()
    previews={20,75,100,175,225,244,250,275,302,304,305,308,313,314,316,321,325,332,336,355,368,369,375,388,392,395,396,400,404,410,415,450,475,526,575,599}
    for i in range(600):
        deadline=time.monotonic()+600
        while not (run/'frames'/f'{i:06d}.json.gz').exists():
            assert time.monotonic()<deadline;time.sleep(.2)
        ok,im=cap.read();assert ok;r=record(i,run);assert pixelsha(im)==r['source_pixel_sha256']
        rgb=panel(im,r);p=Image.fromarray(rgb);dr=ImageDraw.Draw(p);dr.rectangle((0,0,1290,65),fill='#111b25')
        text(dr,(24,16),'RES-1.3 · BG6.3 边界遮挡 / GV1 交点验证',36)
        enc.stdin.write(p.tobytes())
        if i in previews:p.save(out/f'preview_{i:03d}.jpg',quality=94)
        if i%100==0:print('refined render',i,round(time.perf_counter()-tick,1),flush=True)
    cap.release();enc.stdin.close();err=enc.stderr.read().decode(errors='replace');assert enc.wait()==0,err[-2000:]
    check=cv2.VideoCapture(str(path));n=0
    while True:
        ok,im=check.read()
        if not ok:break
        n+=1;assert im.shape==(1080,1920,3)
    fps=check.get(5);check.release();assert n==600 and fps==10
    dump(out/'video_receipt.json',dict(path=str(path),decoded_frames=n,fps=fps,sha256=sha(path),elapsed_s=time.perf_counter()-tick))
if __name__=='__main__':main()
