import cv2
from common23 import *
from review22 import tile as base_tile,sheet
from PIL import Image
def tile(im,label):
    base=base_tile(im,label);canvas=Image.new('RGB',(700,646),'#111b25');canvas.paste(base,(0,0))
    p=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    canvas.paste(p.crop((24,910,830,977)).resize((700,58)),(0,561))
    canvas.paste(p.crop((852,944,1892,978)).resize((700,23)),(0,620))
    return canvas
def main():
    out=ROOT/'review';out.mkdir(exist_ok=True);tiles=[];frames=[]
    for item in plan():
        id=item['id'];s=read(ROOT/'delivery'/id/'summary.json');rr=read(ROOT/'delivery'/id/'frames.json')
        indices={len(rr)//2}
        for e in s['events']:
            if e['timeline']:
                indices.add(min(len(rr)-1,round(e['timeline']['t2']*item['fps'])))
        cap=cv2.VideoCapture(s['video'])
        for i in sorted(indices):
            cap.set(1,i);ok,im=cap.read();assert ok
            tiles.append(tile(im,id.replace('dual_','')+f' {i/10:.1f}s'))
            frames.append(dict(id=id,frame=i,phase=rr[i]['fusion23']['state']))
        cap.release()
    for k in range(0,len(tiles),4):sheet(tiles[k:k+4],out/f'all_{k//4}.jpg')
    id='dual_20260924_111500__left';cap=cv2.VideoCapture(str(ROOT/'delivery'/id/f'RES23_{id}_full.mp4'));tiles=[]
    for i in [249,273,276,290,300,304,311,320]:
        cap.set(1,i);ok,im=cap.read();assert ok;tiles.append(tile(im,f'111500 left {i/10:.1f}s'));frames.append(dict(id=id,frame=i))
    cap.release()
    for k in range(0,len(tiles),4):sheet(tiles[k:k+4],out/f'target_{k//4}.jpg')
    dump(ROOT/'review_selection.json',frames);print('selected',len(frames))
if __name__=='__main__':main()
