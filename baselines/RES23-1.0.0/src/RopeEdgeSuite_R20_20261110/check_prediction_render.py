import cv2
from common20 import *
from render22 import panel
id='dual_20260924_111500__left';i=310;item=next(x for x in plan() if x['id']==id);r=record(id,i)
old=read(ROOT/'delivery'/id/'frames.json')[i];assert old['filtered']['source']=='short_prediction_max_0_5s'
with gzip.open(ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:row=json.load(f)
row.update(filtered=old['filtered'],contact=old['contact'])
with gzip.open(ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:g=json.load(f)
fr=read(ROOT/'f3_v2'/id/'records'/f'{i:06d}.json');data=None
if fr['called']:
    with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,im=cap.read();cap.release();assert ok and pixelsha(im)==r['source_pixel_sha256']
canvas=panel(im,r,row,item,read(ROOT/'calibration'/f'{id}.json'),g['groups'],data)
cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(ROOT/'render_qa_prediction.jpg')
print('Prediction label rendered',flush=True)
