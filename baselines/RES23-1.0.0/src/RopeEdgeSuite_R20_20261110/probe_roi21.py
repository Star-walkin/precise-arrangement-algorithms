import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,time,cv2
from common20 import *
from roi21 import masks
from groups20 import group
from geometry22 import measure as geometry
from count20 import measure as count

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--frame',type=int,required=True);ap.add_argument('--d',type=float,required=True);a=ap.parse_args()
    item=next(x for x in plan() if x['id']==a.id);i=a.frame;r=record(a.id,i)
    cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,raw=cap.read();cap.release();assert ok and pixelsha(raw)==r['source_pixel_sha256']
    im=cv2.flip(raw,1) if item['camera']=='right' else raw
    surface,mask,meta,top=masks(im,r);print('roi',meta,flush=True);b=dict(r['boundary'],top_curve_xy=top)
    out=ROOT/'roi21_probes'/a.id;out.mkdir(parents=True,exist_ok=True);field=out/f'{i:06d}.npz'
    if field.exists():
        with np.load(field) as z:data={k:z[k] for k in z.files}
        assert str(data['source_pixel_sha256'])==pixelsha(im)
    else:
        ref=load('_roi21_f3_probe',PARENT/'RopeEdgeSuite_R12_20261109/run_full.py');tick=time.time();data=ref.extract_current(im,mask,r['track'])
        data['analysis_top_curve_xy']=top;np.savez_compressed(field,**data);print('extract_seconds',time.time()-tick,flush=True)
    tick=time.time();g=group(data,b,a.d);geom=geometry(im,r,g)
    result=count(g,data,b,geom['axis'],dict(d=a.d,interval=[a.d-100,a.d+100],status='conditional_geometry_estimate'))
    dump(out/f'{i:06d}.json',dict(roi=meta,groups=g,geometry=geom,result=result,source_pixel_sha256=r['source_pixel_sha256'],fields_sha256=sha(field),
        producer_sha256=sha(__file__),roi_code_sha256=sha(ROOT/'roi21.py')))
    print({k:v for k,v in result.items() if k not in ('central','column_sensitivity_samples')},flush=True)
    ann=im.copy()
    for c in result.get('central',{}).get('curves',[]):cv2.polylines(ann,[np.rint(c['xy']).astype(np.int32)],False,(30,180,30),2)
    for q in result.get('central',{}).get('crossings',[]):cv2.circle(ann,(round(a.d),round(q['y'])),4,(0,255,255),-1)
    cv2.polylines(ann,[np.rint(top).astype(np.int32)],False,(255,180,60),3)
    cv2.line(ann,(round(a.d),1600),(round(a.d),2999),(255,0,255),2)
    crop=lambda x:cv2.resize(x[1700:3000,700:3300],(1300,650))
    cv2.imencode('.jpg',np.vstack([crop(im),crop(ann)]))[1].tofile(out/f'{i:06d}.jpg')
if __name__=='__main__':main()
