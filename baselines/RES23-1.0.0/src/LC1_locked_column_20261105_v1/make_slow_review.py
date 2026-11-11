"""Explicitly labelled 5x slow replay of the 16 real per-frame results."""
from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from lc1_video import dump,sha

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'runs'/'live_observed_turn_left_10_26'


def main():
    source=FOLDER/'overlay.mp4';output=FOLDER/'review_slow5x.mp4'
    if output.exists():raise FileExistsError(output)
    source_sha=sha(source)
    cap=cv2.VideoCapture(str(source))
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert n==16 and abs(fps-10)<1e-6
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-hide_banner','-loglevel','error',
         '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),
         '-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20',
         '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    with (FOLDER/'review_encoder.log').open('wb') as log:
        encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            for i in range(n):
                ok,frame=cap.read();assert ok
                pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pil)
                draw.rectangle((12,305,min(width-12,1450),355),fill=(17,23,31))
                draw.text((24,315),'5 倍慢放｜源帧 10–25 均独立实际推理；重复画面只用于查看，非实时处理',font=font,fill='#ffe39b')
                rendered=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
                for _ in range(5):encoder.stdin.write(rendered.tobytes())
        finally:
            cap.release();encoder.stdin.close();code=encoder.wait()
    assert code==0 and sha(source)==source_sha
    verify=cv2.VideoCapture(str(output));actual=int(verify.get(cv2.CAP_PROP_FRAME_COUNT))
    first,_=verify.read();verify.set(cv2.CAP_PROP_POS_FRAMES,actual-1);last,_=verify.read();eof,_=verify.read();verify.release()
    assert actual==80 and first and last and not eof
    dump(FOLDER/'review_slow5x_receipt.json',dict(source_video=str(source),source_video_sha256=source_sha,
        source_frames=n,source_fps=fps,output_frames=actual,output_fps=fps,output_seconds=actual/fps,
        repeat_factor=5,burned_in_slow_review_label=True,
        additional_algorithm_inference=False,continuous_source_video_inference_result=True,
        source_video_unchanged=True,output_sha256=sha(output),renderer_sha256=sha(Path(__file__))))
    print(str(output))


if __name__=='__main__':main()
