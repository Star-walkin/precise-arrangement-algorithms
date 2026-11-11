"""Exploratory read-only baseline and video diagnostics, not a detector."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parent
R21=ROOT.parent/'RopeEdgeSuite_R21_20261110'
sys.path.insert(0,str(R21))
from common21 import plan,read,dump,pixelsha
from PIL import Image,ImageDraw

def main():
    cv2.setNumThreads(1);out=ROOT/'inspection';out.mkdir(exist_ok=True)
    summaries=[]
    for item in plan():
        rows=read(R21/'delivery'/item['id']/'frames.json')
        states=[r['near_event']['state'] for r in rows]
        from collections import Counter
        summaries.append(dict(id=item['id'],states=dict(Counter(states)),transitions=sum(a!=b for a,b in zip(states,states[1:])),
            first20=dict(Counter(states[:200])),events=read(R21/'events'/f"{item['id']}.json")['events']))
    dump(out/'baseline_states.json',summaries)
    item=next(x for x in plan() if x['id'].endswith('111500__left'))
    cap=cv2.VideoCapture(item['source']);frames=[];tiles=[]
    for i in range(item['planned_frames']):
        ok,im=cap.read();assert ok
        small=cv2.resize(im,(1024,750),interpolation=cv2.INTER_AREA)
        frames.append(small[380:575,150:875])
        if i in [0,100,200,246,260,270,280,290,300]:
            cv2.imencode('.jpg',small)[1].tofile(out/f'full_{i:06d}.jpg')
            tile=Image.fromarray(cv2.cvtColor(small[380:650,150:875],cv2.COLOR_BGR2RGB));ImageDraw.Draw(tile).text((5,5),f'{i/10:.1f}s',fill='red');tiles.append(tile)
    cap.release();np.savez_compressed(out/'flange_probe.npz',frames=np.array(frames))
    sheet=Image.new('RGB',(1450,270*((len(tiles)+1)//2)))
    for j,tile in enumerate(tiles):sheet.paste(tile,((j%2)*725,(j//2)*270))
    sheet.save(out/'flange_sheet.jpg',quality=94)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
