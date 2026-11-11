import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,concurrent.futures as cf,collections,cv2
from common21 import *

def process(item):
    from render_final21 import panel,writer
    from final_input21 import compose
    from temporal_contact20 import DistanceFilter,contact_evidence
    from events21b import analyze
    from pose_transition21 import estimate
    cv2.setNumThreads(1);id=item['id'];out=ROOT/'delivery'/id;out.mkdir(parents=True,exist_ok=True);(out/'review').mkdir(exist_ok=True)
    cal=read(R20/'calibration'/f'{id}.json');counter=DistanceFilter();rows=compose(item)
    ev=analyze(rows,item['fps'])
    for event in ev['events']:event['pose_transition']=estimate(rows,event)
    dump(ROOT/'events'/f'{id}.json',ev)
    oldrows=read(R20/'delivery'/id/'frames.json');compact=[];cap=cv2.VideoCapture(item['source']);target=out/f'RES21_{id}_full.mp4';tmp=out/f'RES21_{id}_full.part.mp4';proc=writer(tmp,item['fps'])
    selections=set([0,item['planned_frames']//4,item['planned_frames']//2,3*item['planned_frames']//4,item['planned_frames']-1])
    for event in ev['events']:
        for t in (event['start_s'],event['entry_confirmed_s'],event['end_s'],event.get('departure_s')):
            if t is not None:selections.update([max(0,min(item['planned_frames']-1,round(t*item['fps'])+j)) for j in (-5,0,5)])
    if id.endswith('111143__right'):selections.update([10,30,60,90,120,140,150,160])
    if id.endswith('111500__left'):selections.update([200,240,247,249,255,270,280,289,290,300,310])
    start=time.time()
    for i,row_summary in enumerate(rows):
        ok,raw=cap.read();assert ok;r=frontend(id,i);assert pixelsha(raw)==r['source_pixel_sha256']==row_summary['source_pixel_sha256']
        cp=Path(row_summary['measurement_record_path']);assert sha(cp)==row_summary['measurement_record_sha256'];row=gzread(cp)
        m=row['measurement'];phase=next(x for x in cal['epochs'] if x['start']<=i<x['end'])
        row['filtered']=counter.step(row['time_s'],(row['epoch'],phase['d'] is None),m)
        row['contact']=contact_evidence(m,row['axis']) if m.get('d') is not None else oldrows[i]['contact']
        row['near_event']=ev['frames'][i]
        fr=read(R20/'f3_v2'/id/'records'/f'{i:06d}.json');data=None
        if fr['called']:
            with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
        groups=gzread(R20/'groups_v2'/id/'frames'/f'{i:06d}.json.gz')['groups']
        canvas=panel(raw,r,row,item,cal,groups,data);proc.stdin.write(canvas.tobytes())
        if i in selections:cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(out/'review'/f'{i:06d}.jpg')
        compact.append(dict(row_summary,filtered=row['filtered'],contact=row['contact'],near_event=row['near_event']))
        if i%100==0:print('render',id,i,round(time.time()-start),flush=True)
    cap.release();proc.stdin.close();stderr=proc.stderr.read().decode('utf8',errors='replace');assert proc.wait()==0,stderr;tmp.replace(target)
    assert sha(item['source'])==item['source_sha256']
    dump(out/'frames.json',compact)
    summary=dict(id=id,frames=len(compact),fps=item['fps'],source=item['source'],source_sha256=item['source_sha256'],video=str(target),video_sha256=sha(target),
        available_distance_frames=sum(x['available'] for x in compact),dynamic_distance_frames=sum(x['measurement_mode']=='current_dynamic_column' for x in compact),
        complete_F3_phase_frames=sum(x['complete_F3_phase'] for x in compact),near_events=ev['events'],
        frame_contact_states=dict(collections.Counter(x['contact']['state'] for x in compact)),event_states=dict(collections.Counter(x['near_event']['state'] for x in compact)),
        physical_contact_confirmed_frames=0,per_frame_pixel_hash_verified=True,skipped_decoded_frames=0,review_frames=sorted(selections),elapsed_s=time.time()-start)
    dump(out/'summary.json',summary);print('complete',id,flush=True);return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();jobs=list(plan());pending={};completed=[]
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        while jobs or pending:
            for item in list(jobs):
                if len(pending)>=a.workers:break
                if (ROOT/'measurements'/item['id']/'summary.json').exists() and (ROOT/'dynamic_b_receipt.json').exists():
                    jobs.remove(item);pending[pool.submit(process,item)]=item['id']
            if not pending:time.sleep(2);continue
            done,_=cf.wait(pending,timeout=2,return_when=cf.FIRST_COMPLETED)
            for f in done:pending.pop(f);completed.append(f.result())
    dump(ROOT/'delivery_receipt.json',dict(videos=completed,code_hashes={f:sha(ROOT/f) for f in ['finish_final21.py','render_final21.py','final_input21.py','events21b.py','pose_transition21.py']}))
