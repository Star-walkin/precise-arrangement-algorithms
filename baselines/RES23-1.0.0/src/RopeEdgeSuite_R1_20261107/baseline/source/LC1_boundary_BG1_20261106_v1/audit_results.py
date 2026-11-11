"""Independent saved-artifact checks; structural success is not accuracy."""
from pathlib import Path
import cv2,json,numpy as np,hashlib,collections
from run_full import ROOT,SOURCE,OLD,sha,dump

def audit(run_name='full_002'):
    run=ROOT/run_name;binding=json.loads((run/'binding.json').read_text(encoding='utf8'))
    errors=[];counts=collections.Counter();reasons=collections.Counter();total_points=0
    c=cv2.VideoCapture(str(SOURCE));N=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    assert N==601
    mask_tests=0
    for i in range(N):
        ok,a=c.read()
        if not ok:errors.append(f'frame {i}: source decode failed');continue
        p=run/'frames'/f'{i:06d}.json'
        if not p.exists():errors.append(f'frame {i}: result missing');continue
        r=json.loads(p.read_text(encoding='utf8'));region=r['region']
        if hashlib.sha256(a.tobytes()).hexdigest()!=r['source_pixel_sha256']:errors.append(f'frame {i}: pixel binding')
        if sha(OLD/f'frame_{i:06d}.json')!=r['reused_S1_record_sha256']:errors.append(f'frame {i}: S1 binding')
        if r['frame_index']!=i or r['fixed_d_original_px']!=2238.7:errors.append(f'frame {i}: coordinate binding')
        masks={}
        for name,v in r['masks'].items():
            path=run/v['path'];m=cv2.imdecode(np.fromfile(path,dtype=np.uint8),0);masks[name]=m
            if sha(path)!=v['sha256']:errors.append(f'frame {i}: {name} mask hash')
            if m.shape!=(3000,4096) or not np.all((m==0)|(m==255)):errors.append(f'frame {i}: invalid mask')
        if np.any((masks['yellow']>0)&(masks['surface']>0)):errors.append(f'frame {i}: conflicting semantic masks')
        if np.any(masks['surface'][:1500]):errors.append(f'frame {i}: outside scene search ROI')
        counts['region_passed' if region['valid'] else 'region_blocked']+=1
        if not region['valid'] and (r['F3_called'] or r['F3'] is not None or r['count_valid'] or r['display_N'] is not None):
            errors.append(f'frame {i}: gate bypass')
        if r['F3_called']:
            counts['F3_called']+=1;f=r['F3'];pts=np.asarray(f['raw_observed_points_xy'],float).reshape(-1,2)
            if len(pts):
                q=np.rint(pts).astype(int);inside=(q[:,0]>=0)&(q[:,0]<4096)&(q[:,1]>=0)&(q[:,1]<3000)
                if not np.all(inside):errors.append(f'frame {i}: observation outside image')
                elif not np.all(masks['surface'][q[:,1],q[:,0]]>0):errors.append(f'frame {i}: raw observation outside current ROI')
            total_points+=len(pts);mask_tests+=1
            if f['processing_scale']!=.5:errors.append(f'frame {i}: scale mismatch')
        if r['count_valid']:counts['count_valid']+=1
        if r['display_N'] is not None:
            counts['numeric_display']+=1
            if not r['count_valid'] or not 0<=r['display_N']<=5:errors.append(f'frame {i}: numeric gate')
        reasons.update(r['reasons'])
        if i%100==0:print('Audited',i,flush=True)
    c.release()
    for p,digest in binding['code_sha256'].items():
        if sha(p)!=digest:errors.append('run source changed: '+p)
    for p,v in json.loads((ROOT/'vendor_manifest.json').read_text(encoding='utf8')).items():
        if sha(p)!=v['sha256'] or sha(v['original'])!=v['sha256']:errors.append('vendor copy differs: '+p)
    counter=ROOT.parent/'LC1_locked_column_20261105_v1/lc1_surface.py'
    expected_counter='5db1749c537a3cd87e64ea9832f5e34142308b8d59bb49d09be1a5a8089ff159'
    if sha(counter)!=expected_counter:errors.append('external unchanged counting helper source changed')
    if sha(SOURCE)!=binding['sha256']:errors.append('source video hash changed')
    video=ROOT/'delivery/BG1_boundary_gated_full.mp4';video_count=None;video_fps=None
    if video.exists():
        vc=cv2.VideoCapture(str(video));video_count=0;video_fps=float(vc.get(cv2.CAP_PROP_FPS))
        if abs(video_fps-10)>1e-6:errors.append('video fps')
        while True:
            ok,v=vc.read()
            if not ok:break
            video_count+=1
            if v.shape[:2]!=(1080,1920):errors.append('video resolution')
        vc.release()
        if video_count!=601:errors.append('output video frame count')
    else:errors.append('delivery video missing')
    result=dict(status='passed' if not errors else 'failed',errors=errors,frames=N,counts=dict(counts),
                reason_counts=dict(reasons),checked_raw_points=total_points,observation_masks_checked=mask_tests,
                source_sha256=binding['sha256'],video_frames=video_count,video_fps=video_fps,
                video_seconds=None if not video_fps else video_count/video_fps,
                video_sha256=sha(video) if video.exists() else None,
                current_region_required_before_F3=True,no_held_previous_mask_or_count=True,
                provided_source_bytes_verified=True,absolute_accuracy=None,
                audit_scope='saved source binding, ROI containment, coordinate scale and hard gating; not manual geometric ground truth')
    dump(ROOT/'audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    if errors:raise SystemExit(1)
if __name__=='__main__':audit()
