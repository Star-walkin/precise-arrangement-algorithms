import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,cv2,numpy as np
from PIL import Image,ImageDraw
from common import *
from render import annotate,pic,font,panel
def main():
    a=argparse.ArgumentParser();a.add_argument('--id',required=True);a.add_argument('--frames',required=True);opt=a.parse_args()
    item=next(x for x in read(ROOT/'plan.json')['videos'] if x['id']==opt.id)
    ids=[int(x) for x in opt.frames.split(',')];images=[];checks=[]
    cap=cv2.VideoCapture(item['source'])
    for i in ids:
        r=record(ROOT/'runs'/opt.id,i);cap.set(1,i);ok,raw=cap.read();assert ok
        assert pixelsha(raw)==r['source_pixel_sha256']
        im=cv2.flip(raw,1) if item['camera']=='right' else raw
        ann=annotate(im,r);tile=Image.new('RGB',(1000,810),'#111b25');d=ImageDraw.Draw(tile)
        label=f"{opt.id} {i/item['fps']:.1f}s | track {r['track']['valid']} / edge {r['boundary']['fit_valid']}"
        d.text((10,8),label,font=font(20),fill='white')
        tile.paste(pic(im[1750:3000,:3400],(1000,368)),(0,45))
        tile.paste(pic(ann[1750:3000,:3400],(1000,368)),(0,430))
        images.append(tile);checks.append(dict(frame=i,source_pixel_verified=True,tracking=r['track']['valid'],boundary=r['boundary']['fit_valid']))
        if len(ids)<=3:
            path=ROOT/'review'/f'{opt.id}__final_{i:06d}.jpg';path.parent.mkdir(exist_ok=True)
            Image.fromarray(panel(raw,r,item)).save(path,quality=95)
    cap.release();sheet=Image.new('RGB',(2000,810*((len(images)+1)//2)),'#111b25')
    for j,tile in enumerate(images):sheet.paste(tile,((j%2)*1000,(j//2)*810))
    out=ROOT/'review'/f'{opt.id}__sheet_{ids[0]:06d}_{ids[-1]:06d}.jpg';out.parent.mkdir(exist_ok=True);sheet.save(out,quality=94)
    dump(out.with_suffix('.json'),dict(id=opt.id,frames=checks,image=str(out),visual_judgment_pending=True));print(str(out),flush=True)
if __name__=='__main__':main()
