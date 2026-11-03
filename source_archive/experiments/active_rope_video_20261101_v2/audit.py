"""Stored-evidence checks; no model, detector or device is imported."""
import sys,json,gzip,hashlib,time,argparse
from pathlib import Path
import numpy as np
sys.path.append('D:/precise_arrangement/vision_prototype/.venv/Lib/site-packages')
import cv2
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=HERE/a.run
    protocol=read(run/'protocol.json');summary=read(run/'summary.json');checks=0;videos=[];started=time.perf_counter()
    for p,h in protocol['sources_sha256'].items():assert sha(p)==h;checks+=1
    assert summary['frames']==protocol['planned_source_frames']==summary['fresh_gray_detections'];checks+=1
    total_actual_forwards=total_actual_gray_calls=0
    for v in summary['results']:
        folder=run/v['video_id'];n=v['source_frames'];seen=np.zeros(n,np.uint8);measures={};changes=0
        last_sha=None;floats=0;surface_paths=0
        for chunk in sorted((folder/'chunks').iterdir()):
            receipt=read(chunk/'summary.json');assert receipt['status']=='complete';checks+=1
            assert sha(chunk/'measurements.jsonl.gz')==receipt['measurement_sha256'];checks+=1
            last_forward_total=0;last_gray_calls=0
            with gzip.open(chunk/'measurements.jsonl.gz','rt',encoding='utf8') as f:
                for text in f:
                    r=json.loads(text);i=r['source_frame_index'];seen[i]+=1;d=r['diagnostics'];surface=r['surface_tracks'];raw=r['raw_candidates']
                    assert r['gray_measurement_frame']==i and r['gray_measurement_time_s']==r['source_time_s'];checks+=1
                    assert d['fresh_gray_detection_calls']==i-receipt['warmup_start']+1;checks+=1
                    assert d['scene']['new_gray_center_measurement'] and d['fresh_surface']['fresh_detection'];checks+=1
                    assert not d['fresh_surface']['historical_boundary_coordinates_used'] and not d['fresh_surface']['old_center_fallback'];checks+=1
                    points=np.asarray([p for tr in surface for p in tr['points_xy']],np.float32).reshape(-1,2)
                    last_forward_total=d['forward_total'];last_gray_calls=d['fresh_gray_detection_calls']
                    signature=hashlib.sha256(points.tobytes()).hexdigest();assert signature==d['gray_center_sha256'];checks+=1
                    changes+=int(last_sha is not None and signature!=last_sha);last_sha=signature
                    tracks={tr['track_id']:tr for tr in surface}
                    for c in raw:
                        if c.get('surface_exact_center_track'):
                            track=tracks[c['track_id']];idx=c['surface_source_indices'];tail=np.asarray(c['points_xy'])[np.asarray(c['stage'])=='surface']
                            ref=np.asarray(track['points_xy'])[idx]
                            assert tail.shape==ref.shape and np.allclose(tail,ref,rtol=0,atol=1e-5);checks+=1;surface_paths+=1
                    for path in r['heat_paths']:
                        assert Path(path).name.startswith(f'{i:06d}_E');checks+=1
                    measures[i]=(signature,len(raw),r['source_time_s'],r['gate_candidates'],r['heat_selected_default'])
            total_actual_forwards+=last_forward_total;total_actual_gray_calls+=last_gray_calls
        assert np.all(seen==1);checks+=1
        final_count=0;previous=-float('inf')
        with gzip.open(folder/'detections.jsonl.gz','rt',encoding='utf8') as f:
            for text in f:
                r=json.loads(text);i=r['source_frame_index'];sig,nc,t,candidates,default=measures[i]
                assert i==final_count and r['display_source_frame']==i==r['gray_measurement_frame'];checks+=1
                assert r['gray_center_sha256']==sig and r['gray_age_s']==r['display_heat_age_s']==0.;checks+=1
                assert r['source_time_s']==t and t>previous;checks+=1;previous=t
                temporal=r['temporal'];assert not temporal['uses_future'];checks+=1
                assert all(tt<t for tt in temporal['consulted_accepted_history_times_s']+temporal['consulted_pending_history_times_s']);checks+=1
                if temporal['accepted']:
                    k=temporal['accepted_candidate_index'];assert candidates[k]['image_supported'];checks+=1
                    assert np.array_equal(temporal['points_xy'],r['raw_candidates'][k]['points_xy']);checks+=1
                    assert r['heatmap_entry_index']==candidates[k]['entry_index'];checks+=1
                else:assert not temporal['history_updated'];checks+=1
                if temporal['status']=='STALE':assert temporal['reference_age_s']<=1.000001;checks+=1
                final_count+=1
        assert final_count==n;checks+=1
        for p in (folder/'float_keyframes').glob('*.npz'):
            with np.load(p) as data:
                for k in data:
                    if k.startswith('entry_'):
                        x=data[k];assert x.dtype==np.float32 and x.shape==(975,2048);checks+=1
                        assert np.isfinite(x).all() and x.min()>=0 and x.max()<=1;checks+=1;floats+=1
        movie=Path(v['output_path']);assert sha(movie)==v['output_sha256'];checks+=1
        cap=cv2.VideoCapture(str(movie));fps=cap.get(cv2.CAP_PROP_FPS);decoded=0
        assert abs(fps-v['output_fps'])<.002;checks+=1
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(1280,1600,3);checks+=1;decoded+=1
        cap.release();assert decoded==n;checks+=1
        videos.append(dict(video_id=v['video_id'],frames=n,gray_measurements=n,consecutive_gray_geometry_changes=changes,
            exact_current_surface_tail_paths=surface_paths,decoded_frames=decoded,float_keyframe_arrays=floats))
        print(json.dumps(videos[-1]),flush=True)
    result=dict(status='passed',exit_code=0,checks=checks,frames=sum(v['frames'] for v in videos),videos=videos,
        actual_B_forwards_including_chunk_warmup=total_actual_forwards,actual_gray_detection_calls_including_warmup=total_actual_gray_calls,
        protocol_sha256=sha(run/'protocol.json'),summary_sha256=sha(run/'summary.json'),audit_source_sha256=sha(Path(__file__)),
        seconds=time.perf_counter()-started,identity_accuracy_evaluated=False,
        limits='Freshness/alignment/software/decoding checks are not complete target identity accuracy; static ROI/direction calibration remains.')
    (run/'audit_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
if __name__=='__main__':main()
