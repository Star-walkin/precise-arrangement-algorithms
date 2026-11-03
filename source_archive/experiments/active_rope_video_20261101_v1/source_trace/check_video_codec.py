"""Existing OpenCV-only MP4 encode/decode check using synthetic frames."""
from pathlib import Path
import hashlib,json,sys
import cv2
import numpy as np

def main():
    here=Path(__file__).resolve().parent;path=here/'synthetic_mp4v_smoke.mp4'
    assert not path.exists(),'Do not overwrite the first execution receipt'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10.,(128,96))
    assert writer.isOpened()
    for i in range(10):
        image=np.zeros((96,128,3),np.uint8);image[:,:,0]=np.arange(128,dtype=np.uint8)[None,:]
        image[:,:,1]=i*20;image[20:70,10+i*4:40+i*4,2]=200;writer.write(image)
    writer.release()
    cap=cv2.VideoCapture(str(path));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT);shape=(int(cap.get(3)),int(cap.get(4)))
    decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(96,128,3);decoded+=1
    cap.release();assert decoded==10 and count==10 and fps==10 and shape==(128,96)
    result=dict(command=sys.argv,exit_code=0,opencv=cv2.__version__,codec='mp4v',requested_frames=10,decoded_frames=decoded,
        fps=fps,dimensions=list(shape),path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        independent_cli_ffmpeg_found=False,installation_performed=False,source_video_decoded=False,
        interpretation='Existing OpenCV VideoWriter MP4 MPEG4 works. This is codec functionality, not processing-speed or rope-recognition validation.')
    (here/'video_codec_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
