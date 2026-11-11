"""Current-frame 01 F3 core with ROI21, exact-mask reuse and explicit gating."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,sys,time,subprocess,concurrent.futures as cf,cv2
from common20 import *
from roi21 import masks

def one(item):
    cv2.setNumThreads(1);sys.path.insert(0,str(BASE));import transfer_engine as e
    ref=load('_res20_roi21_exact_extractor',PARENT/'RopeEdgeSuite_R12_20261109/run_full.py')
    out=ROOT/'f3_v2'/item['id'];out.mkdir(parents=True,exist_ok=True);(out/'fields').mkdir(exist_ok=True);(out/'records').mkdir(exist_ok=True)
    paths=[Path(__file__),ROOT/'roi21.py',ROOT/'roi20.py',ROOT/'geometry20.py',ROOT/'common20.py',Path(ref.__file__)]+[Path(x) for x in e.core.source_dependencies()]
    hashes={str(x):sha(x) for x in paths};binding=dict(source_item=item,extractor_hashes=hashes,processing_scale=.5,pitch_prior_original_px=PITCH_PRIOR,
        current_01_core_unchanged=True,roi_version='ROI21',current_active_endpoint_prerequisite=True,exact_current_mask_cache_reuse=True)
    if (out/'binding.json').exists():assert read(out/'binding.json')==binding
    else:dump(out/'binding.json',binding)
    cap=cv2.VideoCapture(item['source']);rows=[];start=time.time()
    for i in range(item['planned_frames']):
        ok,raw=cap.read();assert ok;r=record(item['id'],i);assert pixelsha(raw)==r['source_pixel_sha256']
        im=cv2.flip(raw,1) if item['camera']=='right' else raw;assert pixelsha(im)==r['analysis_pixel_sha256']
        dest=out/'records'/f'{i:06d}.json'
        if dest.exists():
            row=read(dest);assert row['analysis_pixel_sha256']==pixelsha(im)
            assert not row['called'] or (Path(row['fields_path']).exists() and sha(row['fields_path'])==row['fields_sha256'])
        else:
            tick=time.time();_,mask,meta,top=masks(im,r);mh=pixelsha(mask);called=bool(meta['valid'] and r['track'].get('valid'))
            field=None;reused=False;origin=None
            oldrecord=ROOT/'f3_v1'/item['id']/'records'/f'{i:06d}.json';oldfield=ROOT/'f3_v1'/item['id']/'fields'/f'{i:06d}.npz'
            if called and oldrecord.exists():
                old=read(oldrecord)
                if old.get('called') and old['mask_sha256']==mh and old['analysis_pixel_sha256']==pixelsha(im) and oldfield.exists():
                    assert sha(oldfield)==old['fields_sha256'];field=oldfield;reused=True;origin=dict(record_path=str(oldrecord),record_sha256=sha(oldrecord),reason='identical_current_pixels_current_ROI_current_track_same_01_core')
            if called and field is None:
                data=ref.extract_current(im,mask,r['track']);data['analysis_top_curve_xy']=top
                data['mask_sha256']=np.asarray(mh);field=out/'fields'/f'{i:06d}.npz';tmp=field.with_suffix('.part.npz')
                np.savez_compressed(tmp,**data);tmp.replace(field);points=len(data['points'])
            elif called:
                points=old['points']
            else:points=0
            row=dict(frame=i,time_s=i/item['fps'],source_pixel_sha256=r['source_pixel_sha256'],analysis_pixel_sha256=pixelsha(im),
                input_stage_record_sha256=sha(selected_run(item['id'])/'frames'/f'{i:06d}.json.gz'),roi=meta,mask_sha256=mh,
                called=called,points=points,fields_path=None if field is None else str(field),fields_sha256=None if field is None else sha(field),
                reused_exact_mask_result=reused,reuse_provenance=origin,
                skip_reason=None if called else ('current_active_endpoint_unavailable' if not r['track'].get('valid') else 'current_boundary_ROI_unavailable'),
                elapsed_s=time.time()-tick)
            dump(dest,row)
        rows.append(row)
        if i%20==0 or i==item['planned_frames']-1:
            progress=dict(id=item['id'],processed=i+1,planned=item['planned_frames'],called=sum(x['called'] for x in rows),
                exact_mask_reused=sum(x['reused_exact_mask_result'] for x in rows),ROI_extended=sum(x['roi'].get('extension_pixels_within_current_surface',0)>0 for x in rows),elapsed_s=round(time.time()-start,1))
            dump(out/'progress.json',progress);print(progress,flush=True)
    cap.release();assert all(sha(p)==h for p,h in hashes.items());assert sha(item['source'])==item['source_sha256']
    dump(out/'compact.json',rows);dump(out/'receipt.json',dict(status='complete',frames=len(rows),pixel_hash_verified_frames=len(rows),skipped_decoded_frames=0,
        F3_called_frames=sum(x['called'] for x in rows),exact_mask_reused_frames=sum(x['reused_exact_mask_result'] for x in rows),source_unchanged=True,
        extractor_unchanged=True,producer_hashes=hashes,elapsed_s=time.time()-start))

def dispatch(item):
    log=ROOT/'f3_v2_logs'/f"{item['id']}.log";log.parent.mkdir(exist_ok=True)
    with log.open('w',encoding='utf8') as f:p=subprocess.run([sys.executable,str(Path(__file__)),'--id',item['id']],stdout=f,stderr=subprocess.STDOUT)
    result=dict(id=item['id'],returncode=p.returncode,log=str(log));print(result,flush=True);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--id');ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    if a.id:one(next(x for x in plan() if x['id']==a.id))
    else:
        jobs=sorted(plan(),key=lambda x:(not x.get('partial_source',False),x['id']))
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(dispatch,jobs))
        dump(ROOT/'f3_v2_receipt.json',dict(status='complete' if all(x['returncode']==0 for x in rows) else 'failed',videos=rows))
        assert all(x['returncode']==0 for x in rows)
