import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import cv2
from common20 import *
from roi22 import masks
from groups20 import group
from geometry22 import measure as geometry
from count20 import measure

def process(id,i):
    cv2.setNumThreads(1);item=next(x for x in plan() if x['id']==id);r=record(id,i)
    with gzip.open(ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:old=json.load(f)
    assert not old['measurement']['available'];cal=read(ROOT/'calibration'/f'{id}.json');phase=next(x for x in cal['epochs'] if x['start']<=i<x['end'])
    assert i not in phase['far_guard_accepted_frames']
    cap=cv2.VideoCapture(item['source']);cap.set(1,i);ok,raw=cap.read();cap.release();assert ok and pixelsha(raw)==r['source_pixel_sha256'];im=cv2.flip(raw,1) if item['camera']=='right' else raw
    surface,mask,meta,top=masks(im,r);out=ROOT/'roi22_corrections'/id;out.mkdir(parents=True,exist_ok=True)
    ref=load('_roi22_exact_core',PARENT/'RopeEdgeSuite_R12_20261109/run_full.py');data=ref.extract_current(im,mask,r['track']);data['analysis_top_curve_xy']=top;data['mask_sha256']=np.asarray(pixelsha(mask));field=out/f'{i:06d}.npz';np.savez_compressed(field,**data)
    g=group(data,dict(r['boundary'],top_curve_xy=top),phase['d']);a=geometry(im,r,g);m=measure(g,data,r['boundary'],a['axis'],phase)
    result=dict(id=id,frame=i,source_pixel_sha256=pixelsha(raw),analysis_pixel_sha256=pixelsha(im),roi=meta,mask_sha256=pixelsha(mask),field=str(field),field_sha256=sha(field),points=len(data['points']),groups=g,geometry=a,measurement=m,
        old_count_file_sha256=sha(ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz'),producer_sha256=sha(__file__),roi_code_sha256=sha(ROOT/'roi22.py'))
    dump(out/f'{i:06d}.json',result);print(id,i,'old',old['measurement']['reasons'],'new',m['available'],m.get('N'),m.get('N_interval'),m['reasons'],flush=True);return result

if __name__=='__main__':process('dual_20260924_111500__left',310)
