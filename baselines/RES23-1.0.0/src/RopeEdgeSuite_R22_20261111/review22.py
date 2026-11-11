import cv2
from PIL import Image,ImageDraw,ImageFont
from common22 import *
F=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
def sheet(tiles,path,cols=2):
    w,h=tiles[0].size;out=Image.new('RGB',(w*cols,h*((len(tiles)+cols-1)//cols)),'#111b25')
    for j,t in enumerate(tiles):out.paste(t,((j%cols)*w,(j//cols)*h))
    out.save(path,quality=95)
def tile(im,label):
    # Actual raw and overlaid closeups, with main state and motor status separately.
    canvas=Image.new('RGB',(700,560),'#111b25');d=ImageDraw.Draw(canvas)
    pic=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    canvas.paste(pic.crop((852,144,1892,482)).resize((700,227)),(0,30))
    canvas.paste(pic.crop((852,529,1892,867)).resize((700,227)),(0,260))
    canvas.paste(pic.crop((840,874,1892,945)).resize((700,47)),(0,490))
    d.text((5,3),label,font=F,fill='white')
    return canvas
def main():
    out=ROOT/'review';out.mkdir(exist_ok=True);alltiles=[];record=[]
    for item in plan():
        s=read(ROOT/'delivery'/item['id']/'summary.json');rows=read(ROOT/'delivery'/item['id']/'frames.json')
        cap=cv2.VideoCapture(s['video']);indices=[len(rows)//4,3*len(rows)//4]
        if len(rows)<10:indices=[len(rows)-1]
        if '101517' in item['id']:indices=[460,490]
        for i in indices:
            cap.set(1,i);ok,im=cap.read();assert ok
            label=f"{item['id'].replace('dual_','')}  {i/10:.1f}s"
            alltiles.append(tile(im,label));record.append(dict(id=item['id'],frame=i,time_s=i/10,state=rows[i]['state22']))
        cap.release()
    for k in range(0,len(alltiles),4):sheet(alltiles[k:k+4],out/f'all_videos_{k//4}.jpg')
    id='dual_20260924_111500__left';cap=cv2.VideoCapture(str(ROOT/'delivery'/id/f'RES22_{id}_full.mp4'));tiles=[]
    for i in [100,200,246,259,273,280,290,300]:
        cap.set(1,i);ok,im=cap.read();assert ok;tiles.append(tile(im,f'111500 left  {i/10:.1f}s'))
        record.append(dict(id=id,frame=i,time_s=i/10))
    cap.release()
    for k in range(0,len(tiles),4):sheet(tiles[k:k+4],out/f'target_{k//4}.jpg')
    dump(ROOT/'visual_review_frames.json',record)
    print('review sheets',len(alltiles),len(tiles))
if __name__=='__main__':main()
