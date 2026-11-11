import cv2
from common20 import *
from render22 import panel
from temporal_contact20 import contact_evidence,DistanceFilter
for id,i in [('dual_20260923_113029__left',569),('dual_20260923_113745__left',248),('dual_20260923_113745__left',405)]:
    x=read(ROOT/'fragments24_trials'/id/f'{i:06d}.json');a=x['geometry']['axis'];m=x['measurement']
    print(id,i,{k:v for k,v in a.items() if k!='points_xy'},{k:m.get(k) for k in ['available','N','N_interval','complete_F3_phase']},'proposal',x['frontend']['proposal']['valid'],flush=True)
    item=next(x for x in plan() if x['id']==id);cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,raw=cap.read();cap.release()
    with gzip.open(ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:row=json.load(f)
    row.update(axis=a,measurement=m);row['filtered']=DistanceFilter().step(row['time_s'],row['epoch'],m);row['contact']=contact_evidence(m,a)
    with np.load(x['field']) as z:data={k:z[k] for k in z.files}
    im=panel(raw,x['frontend'],row,item,read(ROOT/'calibration'/f'{id}.json'),x['groups'],data)
    cv2.imencode('.jpg',cv2.cvtColor(im,cv2.COLOR_RGB2BGR))[1].tofile(ROOT/f'candidate_{id}_{i:06d}.jpg')
