import json,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H/'all_001'
rows=json.loads((R/'measurement_progress.json').read_text(encoding='utf8'))
completed={(r['video_id'],r['stop']):r for r in rows if r['status']=='complete'}
latest={}
for line in (H/'all_001.log').read_text(encoding='utf8').splitlines():
    try:r=json.loads(line)
    except ValueError:continue
    if r.get('phase')=='measure':latest[(r['video'],r['end'])]=r['frame']
trace=json.loads((H.parent/'active_rope_video_20261101_v1/source_trace/source_trace.json').read_text(encoding='utf8'))
approx=sum(r['count'] for r in completed.values());busy=[]
for v in trace['videos']:
    size=round(20*v['nominal_fps']);n=v['frame_count_metadata']
    for start in range(0,n,size):
        stop=min(n,start+size);key=(v['video_id'],stop)
        if key not in completed and key in latest:
            approx+=latest[key]-start+1;busy.append([v['video_id'],start,latest[key],stop])
print(json.dumps(dict(minutes=round((time.time()-(R/'protocol.json').stat().st_mtime)/60,1),
    complete_chunks=len(completed),total_chunks=65,processed_frame_lower_bound=approx,total_source_frames=18017,
    failed_chunks=sum(r['status']!='complete' for r in rows),active=busy)))
