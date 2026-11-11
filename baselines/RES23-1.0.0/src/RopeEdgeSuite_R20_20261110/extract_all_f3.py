"""Fresh supplied-01 observation core on every current validated coil ROI."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,sys,time,subprocess,concurrent.futures as cf,cv2
from common20 import *
from roi20 import masks

def one(item):
    cv2.setNumThreads(1);sys.path.insert(0,str(BASE));import transfer_engine as e
    ref=load('_res20_exact_reference_extractor',PARENT/'RopeEdgeSuite_R12_20261109/run_full.py')
    out=ROOT/'f3_v1'/item['id'];out.mkdir(parents=True,exist_ok=True);(out/'fields').mkdir(exist_ok=True);(out/'records').mkdir(exist_ok=True)
    paths=[Path(__file__),ROOT/'roi20.py',ROOT/'common20.py',Path(ref.__file__)]+[Path(x) for x in e.core.source_dependencies()]
    hashes={str(x):sha(x) for x in paths};binding=dict(source_item=item,extractor_hashes=hashes,processing_scale=.5,pitch_prior_original_px=PITCH_PRIOR,
        full_historical_C3_F2_fusion=False,current_01_normalization_shadow_period_points=True,
        stage='d-independent raw observations; current calibrated column and final class counting follow separately')
    if (out/'binding.json').exists():assert read(out/'binding.json')==binding
    else:dump(out/'binding.json',binding)
    cap=cv2.VideoCapture(item['source']);rows=[];start=time.time()
    for i in range(item['planned_frames']):
        ok,raw=cap.read();assert ok;r=record(item['id'],i);assert pixelsha(raw)==r['source_pixel_sha256']
        image=cv2.flip(raw,1) if item['camera']=='right' else raw;assert pixelsha(image)==r['analysis_pixel_sha256']
        path=out/'records'/f'{i:06d}.json';cache=out/'fields'/f'{i:06d}.npz'
        if path.exists():
            row=read(path);assert row['analysis_pixel_sha256']==pixelsha(image)
            assert not row['called'] or (cache.exists() and sha(cache)==row['fields_sha256'])
        else:
            tick=time.time();_,mask,meta=masks(image,r);data=ref.extract_current(image,mask,r['track']) if meta['valid'] else None
            if data is not None:
                tmp=cache.with_suffix('.part.npz');np.savez_compressed(tmp,**data);tmp.replace(cache)
            row=dict(frame=i,time_s=i/item['fps'],analysis_pixel_sha256=pixelsha(image),source_pixel_sha256=r['source_pixel_sha256'],
                input_stage_record_sha256=sha(selected_run(item['id'])/'frames'/f'{i:06d}.json.gz'),
                roi=meta,mask_sha256=pixelsha(mask),called=data is not None,points=0 if data is None else len(data['points']),
                fields_sha256=sha(cache) if data is not None else None,elapsed_s=time.time()-tick)
            dump(path,row)
        rows.append(row)
        if i%20==0 or i==item['planned_frames']-1:
            progress=dict(id=item['id'],processed=i+1,planned=item['planned_frames'],called=sum(r['called'] for r in rows),elapsed_s=round(time.time()-start,1))
            dump(out/'progress.json',progress);print(progress,flush=True)
    cap.release();assert all(sha(x)==h for x,h in hashes.items());assert sha(item['source'])==item['source_sha256']
    dump(out/'compact.json',rows);dump(out/'receipt.json',dict(status='complete',frames=len(rows),F3_called_frames=sum(r['called'] for r in rows),
        skipped_frames=0,source_unchanged=True,extractor_unchanged=True,pixel_hash_verified_frames=len(rows),elapsed_s=time.time()-start))

def dispatch(item):
    log=ROOT/'f3_logs'/f"{item['id']}.log";log.parent.mkdir(exist_ok=True)
    with log.open('w',encoding='utf8') as f:
        p=subprocess.run([sys.executable,str(Path(__file__)),'--id',item['id']],stdout=f,stderr=subprocess.STDOUT)
    result=dict(id=item['id'],returncode=p.returncode,log=str(log));print(result,flush=True);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--id');ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    if a.id:one(next(x for x in plan() if x['id']==a.id))
    else:
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(dispatch,plan()))
        dump(ROOT/'f3_collection_receipt.json',dict(results=results,status='complete' if all(x['returncode']==0 for x in results) else 'failed'))
        assert all(x['returncode']==0 for x in results)
