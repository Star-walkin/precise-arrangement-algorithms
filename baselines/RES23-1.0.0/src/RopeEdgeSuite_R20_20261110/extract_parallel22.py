"""Frame-chunk dispatcher for the frozen ROI21 / unchanged 01 F3 extraction.

Only scheduling changes. Existing frame records remain immutable. All fresh
records include this dispatcher's hash, current pixels and exact field hashes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import time,argparse,concurrent.futures as cf,cv2
from common20 import *
from roi21 import masks

def process(task):
    cv2.setNumThreads(1);item,lo,hi=task;id=item['id'];out=ROOT/'f3_v2'/id;cap=cv2.VideoCapture(item['source'])
    cap.set(1,lo);ref=load('_res20_parallel_same_extractor',PARENT/'RopeEdgeSuite_R12_20261109/run_full.py');done=0
    for i in range(lo,hi):
        ok,raw=cap.read();assert ok;r=record(id,i);assert pixelsha(raw)==r['source_pixel_sha256']
        dest=out/'records'/f'{i:06d}.json'
        if dest.exists():continue
        im=cv2.flip(raw,1) if item['camera']=='right' else raw;assert pixelsha(im)==r['analysis_pixel_sha256']
        tick=time.time();_,mask,meta,top=masks(im,r);mh=pixelsha(mask);called=bool(meta['valid'] and r['track'].get('valid'))
        field=None;reused=False;origin=None;oldrecord=ROOT/'f3_v1'/id/'records'/f'{i:06d}.json';oldfield=ROOT/'f3_v1'/id/'fields'/f'{i:06d}.npz'
        if called and oldrecord.exists():
            old=read(oldrecord)
            if old.get('called') and old['mask_sha256']==mh and old['analysis_pixel_sha256']==pixelsha(im) and oldfield.exists():
                assert sha(oldfield)==old['fields_sha256'];field=oldfield;reused=True;origin=dict(record_path=str(oldrecord),record_sha256=sha(oldrecord),reason='identical_current_pixels_current_ROI_current_track_same_01_core')
        if called and field is None:
            data=ref.extract_current(im,mask,r['track']);data['analysis_top_curve_xy']=top;data['mask_sha256']=np.asarray(mh)
            field=out/'fields'/f'{i:06d}.npz';tmp=field.with_suffix('.part.npz');np.savez_compressed(tmp,**data);tmp.replace(field);points=len(data['points'])
        elif called:points=old['points']
        else:points=0
        row=dict(frame=i,time_s=i/item['fps'],source_pixel_sha256=r['source_pixel_sha256'],analysis_pixel_sha256=pixelsha(im),
            input_stage_record_sha256=sha(selected_run(id)/'frames'/f'{i:06d}.json.gz'),roi=meta,mask_sha256=mh,
            called=called,points=points,fields_path=None if field is None else str(field),fields_sha256=None if field is None else sha(field),
            reused_exact_mask_result=reused,reuse_provenance=origin,skip_reason=None if called else ('current_active_endpoint_unavailable' if not r['track'].get('valid') else 'current_boundary_ROI_unavailable'),
            elapsed_s=time.time()-tick,dispatcher_sha256=sha(__file__))
        dump(dest,row);done+=1
    cap.release();return dict(id=id,lo=lo,hi=hi,new_frames=done)

def receipt(item):
    out=ROOT/'f3_v2'/item['id'];rows=[read(out/'records'/f'{i:06d}.json') for i in range(item['planned_frames'])]
    for i,row in enumerate(rows):
        r=record(item['id'],i);assert r['source_pixel_sha256']==row['source_pixel_sha256'] and r['analysis_pixel_sha256']==row['analysis_pixel_sha256']
        if row['called']:assert sha(row['fields_path'])==row['fields_sha256']
    assert sha(item['source'])==item['source_sha256']
    binding=read(out/'binding.json');assert all(sha(p)==v for p,v in binding['extractor_hashes'].items())
    dump(out/'compact.json',rows);dump(out/'receipt.json',dict(status='complete',frames=len(rows),pixel_hash_verified_frames=len(rows),skipped_decoded_frames=0,
        F3_called_frames=sum(x['called'] for x in rows),exact_mask_reused_frames=sum(x['reused_exact_mask_result'] for x in rows),source_unchanged=True,extractor_unchanged=True,
        original_binding_sha256=sha(out/'binding.json'),continuation_dispatcher_sha256=sha(__file__),producer_hashes=binding['extractor_hashes']))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=12);a=ap.parse_args();start=time.time()
    jobs=[(item,i,min(i+5,item['planned_frames'])) for item in plan() for i in range(0,item['planned_frames'],5)
        if any(not (ROOT/'f3_v2'/item['id']/'records'/f'{j:06d}.json').exists() for j in range(i,min(i+5,item['planned_frames'])))]
    # Finish near-complete videos first so downstream calibration/count/render
    # can proceed while other extraction chunks are being processed.
    remaining={x['id']:sum(1 for task in jobs if task[0]['id']==x['id']) for x in plan()};jobs.sort(key=lambda t:(remaining[t[0]['id']],t[0]['id'],t[1]))
    hashes={str(ROOT/f):sha(ROOT/f) for f in ('extract_parallel22.py','roi21.py','roi20.py','geometry20.py','common20.py')};done=[]
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(process,task):task for task in jobs}
        for future in cf.as_completed(pending):
            done.append(future.result());p=dict(chunks=len(done),total_chunks=len(jobs),fresh_frames=sum(x['new_frames'] for x in done),elapsed_s=time.time()-start,last=done[-1]);dump(ROOT/'f3_parallel_progress.json',p);print(p,flush=True)
    assert all(sha(p)==v for p,v in hashes.items())
    for item in plan():receipt(item)
    dump(ROOT/'f3_v2_receipt.json',dict(status='complete',videos=[dict(id=x['id'],returncode=0,receipt=str(ROOT/'f3_v2'/x['id']/'receipt.json')) for x in plan()],
        continuation_dispatcher_hashes=hashes,retained_exact_earlier_records=True,mathematical_extractor='unchanged_R12_extract_current_ROI21_01_core',elapsed_s=time.time()-start))
