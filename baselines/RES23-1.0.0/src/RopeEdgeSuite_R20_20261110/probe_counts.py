import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,time,cv2
from common20 import *
from count20 import measure
from geometry21 import measure as geometry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--frame',type=int,required=True);ap.add_argument('--d',type=float,required=True);ap.add_argument('--half',type=float,default=60);a=ap.parse_args()
    id=a.id;i=a.frame;r=record(id,i);item=next(x for x in plan() if x['id']==id)
    cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,raw=cap.read();cap.release();assert ok and pixelsha(raw)==r['source_pixel_sha256']
    im=cv2.flip(raw,1) if item['camera']=='right' else raw
    field=ROOT/'f3_v1'/id/'fields'/f'{i:06d}.npz'
    if not field.exists():field=ROOT/'probes'/id/f'f3_{i:06d}.npz'
    if field.exists():
        with np.load(field) as z:data={k:z[k] for k in z.files}
        assert str(data['source_pixel_sha256'])==pixelsha(im)
    else:
        from roi20 import masks
        ref=load('_probe_count_fresh',PARENT/'RopeEdgeSuite_R12_20261109/run_full.py')
        _,mask,meta=masks(im,r);assert meta['valid'];data=ref.extract_current(im,mask,r['track'])
        folder=ROOT/'probes'/id;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/f'f3_{i:06d}.npz',**data)
        dump(folder/f'f3_{i:06d}.json',dict(id=id,frame=i,analysis_pixel_sha256=pixelsha(im),roi=meta,
            source='fresh_current_01_probe',producer_sha256=sha(__file__),input_record_sha256=sha(selected_run(id)/'frames'/f'{i:06d}.json.gz')))
    groupfile=ROOT/'groups_v1'/id/'frames'/f'{i:06d}.json.gz'
    if groupfile.exists():
        with gzip.open(groupfile,'rt',encoding='utf8') as f:g=json.load(f)['groups']
    else:
        from groups20 import group
        g=group(data,r['boundary'],a.d)
    gp=ROOT/'geometry_v4'/id/'frames'/f'{i:06d}.json.gz'
    if gp.exists():
        with gzip.open(gp,'rt',encoding='utf8') as f:geom=json.load(f)
    else:geom=geometry(im,r,g)
    tick=time.time();result=measure(g,data,r['boundary'],geom['axis'],dict(d=a.d,interval=[a.d-a.half,a.d+a.half],status='conditional_geometry_estimate'))
    out=ROOT/'count_probes_ce4'/id;out.mkdir(parents=True,exist_ok=True);dump(out/f'{i:06d}.json',dict(result=result,geometry=geom,elapsed_s=time.time()-tick))
    print({k:v for k,v in result.items() if k not in ('central','column_sensitivity_samples')},flush=True);print('elapsed',time.time()-tick,flush=True)
    ann=im.copy();central=result.get('central',{})
    for c in central.get('curves',[]):cv2.polylines(ann,[np.rint(c['xy']).astype(np.int32)],False,(40,150,40),2)
    for c in central.get('crossings',[]):cv2.circle(ann,(round(a.d),round(c['y'])),5,(0,255,255),-1)
    cv2.polylines(ann,[np.rint(r['boundary']['top_curve_xy']).astype(np.int32)],False,(255,255,0),3)
    cv2.line(ann,(round(a.d),1600),(round(a.d),2999),(255,0,255),2)
    if geom['axis']:
        abc=geom['axis']['line_abc'];xy=np.array([[x,-(abc[0]*x+abc[2])/abc[1]] for x in [max(0,a.d-1400),a.d]])
        cv2.polylines(ann,[np.rint(xy).astype(np.int32)],False,(0,180,255),3)
    crop=lambda x:cv2.resize(x[1700:3000,700:3300],(1300,650))
    cv2.imencode('.jpg',np.vstack([crop(im),crop(ann)]))[1].tofile(out/f'{i:06d}.jpg')
if __name__=='__main__':main()
