import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
import argparse,time,sys,cv2
from common20 import *
from roi20 import masks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--frames',required=True);ap.add_argument('--mode',choices=['p1','f3'],default='p1');a=ap.parse_args()
    cv2.setNumThreads(1);item=next(x for x in plan() if x['id']==a.id);out=ROOT/'probes'/a.id;out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(item['source']);p1=None
    for i in map(int,a.frames.split(',')):
        dest=out/f'{a.mode}_{i:06d}.json'
        if dest.exists():print('exists',str(dest),flush=True);continue
        cap.set(1,i);ok,raw=cap.read();assert ok;r=record(a.id,i);assert pixelsha(raw)==r['source_pixel_sha256']
        image=cv2.flip(raw,1) if item['camera']=='right' else raw;assert pixelsha(image)==r['analysis_pixel_sha256']
        surface,f3mask,meta=masks(image,r);start=time.perf_counter()
        if a.mode=='p1':
            sys.path.insert(0,str(LC1))
            from lc1_p1_adapter import P1H6Adapter
            if p1 is None:p1=P1H6Adapter(surface,PITCH_PRIOR,LC1/'supplied_assets/step_0750.pt',threads=2,camera='left',surface_mode='entry_band')
            else:p1.surface=surface
            result=p1.step(image)
            if p1.last_heatmaps:np.savez_compressed(out/f'heat_{i:06d}.npz',heat=p1.last_heatmaps[0])
        else:
            sys.path.insert(0,str(BASE));import transfer_engine as e
            m=load('_res20_f3_reference',PARENT/'RopeEdgeSuite_R12_20261109/run_full.py')
            result=m.extract_current(image,f3mask,r['track']) if meta['valid'] else None
            if result is not None:
                np.savez_compressed(out/f'f3_{i:06d}.npz',**result)
                result=dict(point_count=len(result['points']),pitch=float(result['pitch']))
        elapsed=time.perf_counter()-start
        dump(dest,dict(id=a.id,frame=i,time_s=i/item['fps'],analysis_pixel_sha256=pixelsha(image),stage_record_sha256=sha(selected_run(a.id)/'frames'/f'{i:06d}.json.gz'),
            orientation='left_entering_normalized; mirror original right camera',roi=meta,result=result,elapsed_s=elapsed))
        print(a.id,i,a.mode,round(elapsed,1),result.get('status',result.get('point_count')) if result else None,flush=True)
    cap.release()
if __name__=='__main__':main()
