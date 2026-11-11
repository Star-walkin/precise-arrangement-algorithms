import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,concurrent.futures as cf,collections,cv2
from common20 import *

def process(item):
    from render22 import panel,writer
    from contact_fallback20 import evaluate
    from temporal_contact20 import DistanceFilter,contact_evidence,windows
    cv2.setNumThreads(1);id=item['id'];out=ROOT/'delivery'/id;out.mkdir(parents=True,exist_ok=True);(out/'review').mkdir(exist_ok=True)
    cal=read(ROOT/'calibration'/f'{id}.json');counter=DistanceFilter();compact=[];cap=cv2.VideoCapture(item['source']);target=out/f'RES20_{id}_full.mp4';tmp=out/f'RES20_{id}_full.part.mp4';proc=writer(tmp,item['fps'])
    selections=set([0,item['planned_frames']//4,item['planned_frames']//2,3*item['planned_frames']//4,item['planned_frames']-1])
    extra={'dual_20260923_113745__left':[310,370,397,400,500],'dual_20260924_111143__right':[140,152,180],
        'dual_20260924_111500__left':[200,260,281,300],'dual_20260923_113029__right':[430,500,590],
        'dual_20260923_101517__right':[540,580,600],'dual_20260923_113305__right':[499,540,580,599],
        'dual_20260923_113029__left':[306], 'dual_20260923_101643__right':[358,359,570]}
    selections.update(i for i in extra.get(id,[]) if i<item['planned_frames']);start=time.time()
    repairs=read(ROOT/'fragments23_correction_receipt.json')['promoted']+read(ROOT/'fragments24_correction_receipt.json')['promoted']
    changed=[x['frame'] for x in repairs if x['id']==id]
    if changed:selections.update([changed[0],changed[len(changed)//2],changed[-1]])
    for i in range(item['planned_frames']):
        ok,raw=cap.read();assert ok;r=record(id,i);assert pixelsha(raw)==r['source_pixel_sha256']
        cp=ROOT/'counts_v1'/id/'frames'/f'{i:06d}.json.gz'
        with gzip.open(cp,'rt',encoding='utf8') as f:row=json.load(f)
        correction=row.get('fragments24_correction') or row.get('fragments23_correction')
        if correction:
            assert sha(correction['frontend_override_path'])==correction['frontend_override_sha256']
            with gzip.open(correction['frontend_override_path'],'rt',encoding='utf8') as f:r=json.load(f)
            assert r['source_pixel_sha256']==pixelsha(raw)
        assert row['source_pixel_sha256']==pixelsha(raw)
        row['filtered']=counter.step(row['time_s'],row['epoch'],row['measurement']);row['contact']=contact_evidence(row['measurement'],row['axis'])
        fr=read(ROOT/'f3_v2'/id/'records'/f'{i:06d}.json');data=None
        if fr['called']:
            with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
        with gzip.open(ROOT/'groups_v2'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as f:g=json.load(f)
        if row['contact']['state']=='unavailable' and row['measurement'].get('d') is None:
            with gzip.open(ROOT/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz','rt',encoding='utf8') as h:geo=json.load(h)
            row['contact']=evaluate(g['groups'],data,r['boundary'],geo)
        canvas=panel(raw,r,row,item,cal,g['groups'],data);proc.stdin.write(canvas.tobytes())
        if i in selections:cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(out/'review'/f'{i:06d}.jpg')
        m=row['measurement'];central=m.get('central') or {}
        compact.append(dict(frame=i,time_s=row['time_s'],epoch=row['epoch'],phase_direction=row['phase_direction'],
            current_F3_called=fr['called'],F3_point_count=fr['points'],ROI_extended=fr['roi'].get('extension_pixels_within_current_surface',0)>0,
            available=m['available'],status=m['status'],N=m.get('N'),N_interval=m.get('N_interval'),d=m.get('d'),d_interval=m.get('d_interval'),
            complete_F3_phase=m.get('complete_F3_phase',False),observed_grooves=m.get('observed_groove_count'),
            recovered_point_classes=sum(x.get('accepted',False) for x in central.get('unowned_point_recovery',[])),
            filtered=row['filtered'],contact=row['contact'],reasons=m['reasons'],source_pixel_sha256=row['source_pixel_sha256'],count_record_sha256=sha(cp)))
        if i%100==0:dump(out/'progress.json',dict(frames=i+1,total=item['planned_frames'],elapsed_s=time.time()-start))
    cap.release();proc.stdin.close();stderr=proc.stderr.read().decode('utf8',errors='replace');ret=proc.wait();assert ret==0,stderr;tmp.replace(target)
    assert sha(item['source'])==item['source_sha256']
    dump(out/'frames.json',compact);win=windows(compact,item['fps']);dump(out/'contact_windows.json',win)
    summary=dict(id=id,frames=len(compact),duration_s=len(compact)/item['fps'],source=item['source'],source_sha256=item['source_sha256'],
        video=str(target),video_sha256=sha(target),F3_called_frames=sum(x['current_F3_called'] for x in compact),
        available_distance_frames=sum(x['available'] for x in compact),complete_F3_phase_frames=sum(x['complete_F3_phase'] for x in compact),
        ROI_extended_frames=sum(x['ROI_extended'] for x in compact),recovered_point_classes=sum(x['recovered_point_classes'] for x in compact),
        contact_states=dict(collections.Counter(x['contact']['state'] for x in compact)),contact_windows=win,
        reason_counts=dict(collections.Counter(y for x in compact for y in x['reasons'])),
        physical_contact_confirmed_frames=0,independent_physical_contact_truth_available=False,
        per_frame_pixel_hash_verified=True,skipped_decoded_frames=0,calibration=cal,review_frames=sorted(selections),elapsed_s=time.time()-start)
    dump(out/'summary.json',summary);print(id,'rendered',len(compact),flush=True);return dict(id=id,summary=str(out/'summary.json'),video=str(target))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    hashes={str(ROOT/f):sha(ROOT/f) for f in ('finish_v4.py','render22.py','temporal_contact20.py','contact_fallback20.py')};jobs=list(plan());pending={};done_rows=[]
    retained=[]
    for item in list(jobs):
        out=ROOT/'delivery'/item['id'];sp=out/'summary.json';fp=out/'frames.json'
        if not sp.exists() or not fp.exists():continue
        rows=read(fp)
        valid=len(rows)==item['planned_frames']
        for row in rows:
            cp=ROOT/'counts_v1'/item['id']/'frames'/f"{row['frame']:06d}.json.gz"
            if not cp.exists() or sha(cp)!=row['count_record_sha256']:valid=False;break
        if valid:
            summary=read(sp);assert sha(summary['video'])==summary['video_sha256'];jobs.remove(item)
            done_rows.append(dict(id=item['id'],summary=str(sp),video=summary['video'],exact_previous_render_reuse=True,reason='same count records; existing videos already render22 or render21 without prediction frames'))
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        while jobs or pending:
            for item in list(jobs):
                if len(pending)>=a.workers:break
                if all((ROOT/'counts_v1'/item['id']/'frames'/f'{i:06d}.json.gz').exists() for i in range(item['planned_frames'])):
                    jobs.remove(item);pending[pool.submit(process,item)]=item['id']
            if not pending:time.sleep(5);continue
            done,_=cf.wait(pending,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in done:pending.pop(f);done_rows.append(f.result());dump(ROOT/'delivery_progress.json',dict(completed=done_rows,remaining=len(jobs)+len(pending)))
    assert all(sha(p)==h for p,h in hashes.items())
    dump(ROOT/'delivery_receipt.json',dict(status='complete',videos=done_rows,producer_hashes=hashes))
