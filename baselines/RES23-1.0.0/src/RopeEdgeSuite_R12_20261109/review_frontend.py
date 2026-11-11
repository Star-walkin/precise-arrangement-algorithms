"""Diagnostic contact sheets: current actual support only, no hidden fill."""
from pathlib import Path
import argparse,gzip,json,hashlib,math
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
def load(path):return json.loads(gzip.decompress(path.read_bytes()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='dual_20260923_113745_v2');ap.add_argument('--frames',required=True);ap.add_argument('--output',default='frontend_review.png');a=ap.parse_args()
    folder=ROOT/'frontend'/a.name;bind=json.loads((folder/'binding.json').read_text(encoding='utf-8'))
    selected=[int(i) for i in a.frames.split(',')];cap=cv2.VideoCapture(bind['source'])
    canvas=Image.new('RGB',(1800,380*math.ceil(len(selected)/2)),'#14202b');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    for j,i in enumerate(selected):
        r=load(folder/'frames'/f'{i:06d}.json.gz');s=r['track'];cap.set(1,i);ok,im=cap.read();assert ok
        assert hashlib.sha256(im.tobytes()).hexdigest()==r['source_pixel_sha256']
        for seg in s['supported_segments_xy']:
            if len(seg)>1:cv2.polylines(im,[np.rint(seg).astype(np.int32)],False,(30,255,30),4)
        crop=Image.fromarray(cv2.cvtColor(im[1800:2400,550:2100],cv2.COLOR_BGR2RGB)).resize((880,340))
        x=10+900*(j%2);y=380*(j//2)
        draw.text((x,y+4),f'{i/10:.1f}s  '+('当前实测支撑' if s['valid'] else '当前无测量'),font=font,fill='white')
        canvas.paste(crop,(x,y+36))
    cap.release();canvas.save(ROOT/a.output)
    print(str(ROOT/a.output))
if __name__=='__main__':main()
