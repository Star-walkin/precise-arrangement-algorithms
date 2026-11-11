import cv2
from common20 import *
from render20 import panel
from temporal_contact20 import DistanceFilter,contact_evidence

id='dual_20260923_113745__left';i=370;item=next(x for x in plan() if x['id']==id);r=record(id,i);probe=read(ROOT/'roi21_probes'/id/f'{i:06d}.json')
cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,raw=cap.read();cap.release();assert ok
row=dict(frame=i,time_s=i/item['fps'],measurement=probe['result'],axis=probe['geometry']['axis'])
row['filtered']=DistanceFilter().step(row['time_s'],0,row['measurement']);row['contact']=contact_evidence(row['measurement'],row['axis'])
cal=dict(epochs=[dict(start=0,end=600,status='conditional_geometry_estimate',d_interval_original_image=row['measurement']['d_interval'])])
with np.load(ROOT/'roi21_probes'/id/f'{i:06d}.npz') as z:data={k:z[k] for k in z.files}
canvas=panel(raw,r,row,item,cal,probe['groups'],data);cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))[1].tofile(ROOT/'layout_preview_370.jpg')
print(ROOT/'layout_preview_370.jpg')
