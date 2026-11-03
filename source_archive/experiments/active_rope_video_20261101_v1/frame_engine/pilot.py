"""Fixed two source-video pilots; no future dataset predictions are read."""
import sys,json,time,traceback,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from engine import FrameEngine,anchor_job,ROOT,HERE,js,sha,cv2,np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='pilot_001');ap.add_argument('--no-entry-smoke',action='store_true');args=ap.parse_args()
    out=HERE/args.run;out.mkdir(exist_ok=False)
    trace=js(HERE.parent/'source_trace/source_trace.json')
    chosen=['dual_20260923_101517__left'] if args.no_entry_smoke else ['dual_20260923_101517__right','L_run_20260923_01__left']
    rows=[];start=time.perf_counter()
    for ident in chosen:
        v=next(x for x in trace['videos'] if x['video_id']==ident)
        ft=js(v['frame_times_path']);times=np.asarray(ft['observation_times_s'],float)
        a=v['earliest_dataset_anchor'];at=float(a['mapped_observation_time_s'])
        eng=FrameEngine(anchor_job(v),anchor_time=at,threads=2)
        cap=cv2.VideoCapture(v['canonical_path'])
        indices=np.searchsorted(times,at+.5*np.arange(10))
        for number,index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
            if not ok:raise RuntimeError('Cannot decode '+ident+':'+str(index))
            tick=time.perf_counter();result=eng.step(frame,float(times[index]))
            row=dict(video_id=ident,frame_index=int(index),time_s=float(times[index]),
                source_sha256=v['sha256'],status=result['status'],diagnostics=result['diagnostics'],
                seeds=result['seeds'],raw_candidates=result['raw_candidates'],
                heat_statistics=None if result['heat'] is None else dict(min=float(result['heat'].min()),
                    max=float(result['heat'].max()),mean=float(result['heat'].mean()),std=float(result['heat'].std())))
            rows.append(row)
            (out/'progress.jsonl').open('a',encoding='utf8').write(json.dumps(row,ensure_ascii=False)+'\n')
            if number in (0,5,9):
                if result['heat'] is not None:np.save(out/(ident+f'__{index:05d}__heat.npy'),result['heat'])
                panel=frame.copy()
                for c in reversed(result['raw_candidates']):
                    cv2.polylines(panel,[np.rint(c['points_xy']).astype(np.int32)],False,(20,240,20),6,cv2.LINE_AA)
                cv2.imencode('.jpg',cv2.resize(panel,(1024,750)))[1].tofile(str(out/(ident+f'__{index:05d}__path.jpg')))
            print(json.dumps(dict(video=ident,index=int(index),status=result['status'],seconds=round(time.perf_counter()-tick,2),
                forward=result['diagnostics']['forward_count'],candidates=len(result['raw_candidates'])),ensure_ascii=True),flush=True)
        cap.release()
    summary=dict(planned=10*len(chosen),completed=len(rows),elapsed_seconds=time.perf_counter()-start,
        forward_count=sum(r['diagnostics']['forward_count'] for r in rows),
        no_entry=sum(r['status']=='unknown_no_entry' for r in rows),
        average_seconds=float(np.mean([r['diagnostics']['seconds'] for r in rows])),
        engine_sha256=sha(HERE/'engine.py'),source_trace_sha256=sha(HERE.parent/'source_trace/source_trace.json'),
        future_records_used=False,training=False,devices=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
