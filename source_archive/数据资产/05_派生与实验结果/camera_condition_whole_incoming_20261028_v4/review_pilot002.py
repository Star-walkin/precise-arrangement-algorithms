"""Independent comparison of saved phase-1/phase-2 predictions only."""
from pathlib import Path
import hashlib,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT/'pilot_001'
NEW=ROOT/'pilot_002'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def im(p):return cv2.imdecode(np.fromfile(str(p),np.uint8),cv2.IMREAD_UNCHANGED)
items=[]
for rel in read(ROOT/'protocol.json')['pilot_inputs']:
 q=Path(rel);role=q.parent.name
 o=read(OLD/'记录'/role/(q.stem+'.json'));n=read(NEW/'记录'/role/(q.stem+'.json'))
 oc=o['whole'].get('candidates',[]);nc=n['whole'].get('candidates',[])
 om=im(OLD/'01_整根候选掩膜'/role/q.name);nm=im(NEW/'01_整根候选掩膜'/role/q.name)
 items.append({'input':rel,'mask_sha_equal':sha(OLD/'01_整根候选掩膜'/role/q.name)==sha(NEW/'01_整根候选掩膜'/role/q.name),
  'selected_xy_equal':(oc[0]['xy'] if oc else None)==(nc[0]['xy'] if nc else None),
  'old_length_px':oc[0]['length_px'] if oc else 0.,'new_length_px':nc[0]['length_px'] if nc else 0.,
  'added_mask_pixels':int(np.count_nonzero((nm>0)&(om==0))),
  'removed_mask_pixels':int(np.count_nonzero((om>0)&(nm==0))),
  'old_status':o['record']['status'],'new_status':n['record']['status'],
  'new_case_sha256':sha(NEW/'记录'/role/(q.stem+'.json'))})
out=NEW/'independent_visual_comparison';out.mkdir(exist_ok=False)
(out/'geometry_comparison.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
role='right_camera';name='dual_20260923_113305__pair0015__right.png';x0,y0,x1,y1=900,2250,1700,2650
panels=[]
for title,path in [('Original',ROOT.parent/'camera_condition_images_20261009_v2/images'/role/name),
                   ('A4 phase1',OLD/'02_原图叠加'/role/name),
                   ('A4 phase2 - unknown rows give no bonus',NEW/'02_原图叠加'/role/name)]:
 a=im(path)[y0:y1,x0:x1].copy();cv2.rectangle(a,(0,0),(a.shape[1],32),(15,15,15),cv2.FILLED)
 cv2.putText(a,title,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA);panels.append(a)
cv2.imencode('.jpg',np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,96])[1].tofile(str(out/'dual113305_right_tail_comparison.jpg'))
print(json.dumps({'cases':len(items),'changed':[v['input'] for v in items if not v['mask_sha_equal']],'output':str(out)},ensure_ascii=False))
