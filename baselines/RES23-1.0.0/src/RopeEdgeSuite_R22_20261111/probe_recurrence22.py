import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import cv2,numpy as np,json
from scipy.signal import find_peaks
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent
frames=np.load(R/'inspection/flange_probe.npz')['frames'];cv2.setNumThreads(1)
rois=[(100,80,220,115),(220,80,350,115),(250,120,410,165),(400,155,530,190)]
result=[];ref=246
for roi in rois:
    x0,y0,x1,y1=roi
    features=[]
    for f in frames:
        g=cv2.cvtColor(f[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
        features.append(cv2.resize(g-cv2.GaussianBlur(g,(0,0),3),(60,20)).ravel())
    feat=np.array(features);mean=np.mean(feat[100:ref+1],axis=0)
    ff=feat-mean;ff/=np.maximum(np.linalg.norm(ff,axis=1,keepdims=True),1e-6)
    scores=ff@ff[ref];peaks=find_peaks(scores[ref+8:ref+81],prominence=.03)[0]+ref+8
    result.append(dict(roi=roi,peaks=[dict(t=float(k/10),score=float(scores[k])) for k in sorted(peaks,key=lambda k:scores[k],reverse=True)[:8]],scores=scores[240:321].tolist()))
    print(result[-1]['roi'],result[-1]['peaks'])
tiles=[]
for i in [246,259,273,274,275,276,300,301,302]:
    tile=Image.fromarray(cv2.cvtColor(frames[i][65:195,0:600],cv2.COLOR_BGR2RGB)).resize((900,195));ImageDraw.Draw(tile).text((5,5),f'{i/10:.1f}s',fill='red');tiles.append(tile)
sheet=Image.new('RGB',(1800,195*5))
for j,tile in enumerate(tiles):sheet.paste(tile,((j%2)*900,(j//2)*195))
sheet.save(R/'inspection/recurrence_sheet.jpg',quality=96)
(R/'inspection/recurrence_probe.json').write_text(json.dumps(result))
