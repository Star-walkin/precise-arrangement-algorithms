"""Independent output/state/provenance audit. Does not import production modules."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,sys,time
import cv2
import numpy as np

HERE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf8').splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def same(a,b):
    x,y=np.asarray(a),np.asarray(b)
    return x.shape==y.shape and bool(np.allclose(x,y,rtol=0,atol=1e-5))
def arc(p):return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
def sample(p,s):
    a=arc(p);return np.stack([np.interp(s,a,p[:,k]) for k in range(2)],axis=1)
def independent_lateral_p95(old,new,shift,samples):
    r,c=np.asarray(old,float),np.asarray(new,float);a=arc(r);b=arc(c)
    d=sample(r,[0,min(a[-1],min(200,max(20,a[-1]*.25)))])[1]-r[0]
    d=d/max(np.linalg.norm(d),1e-9)
    r=r+np.asarray(shift)
    offset=float((c[0]-r[0])@d);r0=max(offset,0);c0=max(-offset,0)
    length=min(a[-1]-r0,b[-1]-c0)
    if length<1:return None
    s=np.linspace(0,length,samples);rp=sample(r,r0+s);cp=sample(c,c0+s)
    tangent=np.gradient(rp,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
    normal=np.stack([-tangent[:,1],tangent[:,0]],axis=1)
    return float(np.quantile(np.abs(np.sum((cp-rp)*normal,axis=1)),.95))

def audit(run,allow_display_code_drift=False,selected_videos=None,reuse_partial_decodes=False):
    tick=time.perf_counter();errors=[];warnings=[];checks=0
    def check(ok,msg,context=None):
        nonlocal checks;checks+=1
        if not bool(ok):errors.append(dict(message=msg,context=context))
    protocol=read(run/'protocol.json');summary=read(run/'summary.json') if (run/'summary.json').exists() else None;trace=read(HERE/'source_trace/source_trace.json')
    check(sha(HERE/'source_trace/source_trace.json')==protocol['source_trace_sha256'],'source trace hash changed')
    if summary is not None:check(summary['protocol_sha256']==sha(run/'protocol.json'),'protocol hash changed')
    source_matches={p:sha(HERE/p)==h for p,h in protocol['sources_sha256'].items()}
    for p,match in source_matches.items():
        if not match and allow_display_code_drift and p in ['run_videos.py','rendering/render_panels.py']:
            warnings.append('Current '+p+' differs from pilot-time hash; permitted explicit post-pilot display/logging edit, not numerical parity proof.')
        else:check(match,'production source changed since run',p)
    if selected_videos is None:
        check(summary is not None,'whole-run summary missing')
        if summary is not None:
            check(summary['planned_videos']==summary['completed_videos']==len(protocol['videos']),'video denominator incomplete')
            check(not summary['failures'],'run reports failures')
        video_ids=protocol['videos']
    else:
        video_ids=list(selected_videos)
        check(bool(video_ids) and len(set(video_ids))==len(video_ids),'partial audit selection empty or duplicated')
        check(set(video_ids).issubset(set(protocol['videos'])),'partial audit selection outside frozen protocol')
        warnings.append('Partial completed-video audit only; does not assert whole-run completion.')
    cached_decodes={};cache_receipts=[]
    if reuse_partial_decodes:
        for receipt_path in sorted((run/'partial_audits').glob('*_independent_audit.json')):
            receipt=read(receipt_path)
            if receipt.get('status')!='passed' or receipt.get('errors') or receipt.get('protocol_sha256')!=sha(run/'protocol.json'):continue
            if set(receipt.get('source_hash_matches_current',{}))!=set(protocol['sources_sha256']):continue
            if not all(receipt['source_hash_matches_current'].values()):continue
            cache_receipts.append(dict(path=str(receipt_path),sha256=sha(receipt_path)))
            for info in receipt['videos']:
                cached_decodes[info['video_id']]=dict(video_sha256=info['video_sha256'],output_frames=info['output_frames'],receipt=str(receipt_path))
    videos={v['video_id']:v for v in trace['videos']};per_video=[];total_forwards=0;total_responses=0
    for ident in video_ids:
        check(ident in videos,'video outside explicit source allowlist',ident)
        v=videos[ident];folder=run/ident;vs=read(folder/'summary.json');times=np.asarray(read(v['frame_times_path'])['observation_times_s'],float)
        check(sha(v['frame_times_path'])==v['frame_times_sha256'],'source clock table changed',ident)
        check(np.all(np.diff(times)>0),'source observation clock not monotone',ident)
        check(vs['status']=='complete' and not vs['training'] and not vs['devices'],'invalid run scope/status',ident)
        check(vs['source_path']==v['canonical_path'] and vs['source_sha256']==v['sha256'],'source path/hash not allowlisted',ident)
        actual_movie_sha=sha(vs['output_path'])
        check(vs['output_sha256']==actual_movie_sha,'movie changed',ident)
        det=rows(folder/'detections.jsonl');render=rows(folder/'render_frames.jsonl');by_frame={r['source_frame_index']:r for r in det}
        anchor=v['earliest_dataset_anchor'];accepted_history=[];by_time={};counts=Counter();last_t=None;prev_epoch=0;last_accepted=None
        forward_count=0;candidate_count=0;accepted_count=0;rejected=0;baseline_unsupported=0;reference_only=0
        check(len(det)==vs['analysis_frames'],'analysis count mismatch',ident)
        check(len(render)==vs['output_frames'],'render count mismatch',ident)
        if det:check(det[0]['source_frame_index']==anchor['source_frame'],'first analysis is not earliest causal anchor',ident)
        for number,r in enumerate(det):
            context=f'{ident}:source_frame={r["source_frame_index"]}'
            i=r['source_frame_index'];t=r['source_time_s'];tr=r['temporal'];raw=r['raw_candidates'];cand=r['gate_candidates'];diag=r['diagnostics']
            check(r['analysis_index']==number,'analysis ordinal mismatch',context)
            check(i>=anchor['source_frame'] and t+1e-7>=anchor['mapped_observation_time_s'],'future anchor used',context)
            check(abs(t-times[i])<1e-8 and abs(r['media_time_s']-i/v['nominal_fps'])<1e-8,'media/observation clock mapping incorrect',context)
            check(abs(tr['time_s']-t)<1e-8,'gate used other clock',context)
            check(last_t is None or t>last_t,'noncausal time order',context)
            check(r['processing_elapsed_s']>=0 and math.isfinite(r['processing_elapsed_s']),'invalid processing duration',context)
            check(not tr['uses_future'] and not diag['future_records_used'] and not diag['historical_response_reused'],'future/cache response leakage claimed',context)
            check(diag['scene_anchor_id']==anchor['id'] and not diag['scene']['new_f3_measurement'],'scene source misrepresented',context)
            check(len(raw)==len(cand)<=3 and diag['candidate_count']==len(raw),'candidate count mismatch',context)
            for k,(rc,gc) in enumerate(zip(raw,cand)):
                p=np.asarray(rc['points_xy'],float);candidate_count+=1
                check(p.ndim==2 and p.shape[1]==2 and len(p)>=2 and np.isfinite(p).all(),'bad raw candidate geometry',context)
                check(same(p,gc['points_xy']) and gc['entry_index']==rc['entry_index'],'gate geometry or entrance differs from upstream candidate',context)
                states=np.asarray(rc.get('point_status',['unknown']*len(p)))
                observed=np.isin(states,['image_supported','joint_heat_image_supported','raw_a7_supported'])
                length=float(np.linalg.norm(np.diff(p,axis=0),axis=1)[observed[:-1]&observed[1:]].sum())
                check(abs(length-gc['current_raw_supported_length_px'])<1e-5,'raw image support length mismatch',context)
                check(gc['image_supported']==bool(rc.get('task_direction_allowed',False) and length>=80),'current support falsely assigned',context)
                check(abs(gc['image_score']-(1.-.1*k))<1e-8,'current-image ranking changed',context)
            baseline=0 if cand else None
            check(r['single_frame_selected_index']==baseline,'upstream no-history selection mismatch',context)
            guarded=next((k for k,c in enumerate(cand) if c['image_supported'] and arc(np.asarray(c['points_xy']))[-1]>=protocol['gate']['minimum_common_length_px']),None)
            if 'image_guard_only_selected_index' in r:check(r['image_guard_only_selected_index']==guarded,'image-only guarded comparison mismatch',context)
            if baseline is not None and not cand[baseline]['image_supported']:baseline_unsupported+=1
            reset=tr.get('reset_reason')
            if reset:
                check(reset=='large_input_time_gap' and last_t is not None and t-last_t>protocol['gate']['reset_gap_seconds'],'unexpected or missing causal reset cause',context)
                accepted_history=[];last_accepted=None;prev_epoch+=1
            accepted_times=[h['source_time_s'] for h in accepted_history]
            for ht in tr.get('consulted_accepted_history_times_s',[]):
                check(ht<t and ht in accepted_times and t-ht<=protocol['gate']['history_seconds']+1e-8,'future/rejected/expired main history used',context)
            for ht in tr.get('consulted_pending_history_times_s',[]):
                check(ht<t and ht in by_time and not by_time[ht]['temporal']['accepted'],'pending queue contains future or accepted measurement',context)
            status=tr['status'];counts[status]+=1
            check(status in ['ACCEPTED','REACQUIRED','STALE','UNKNOWN'],'invalid temporal state',context)
            if tr['accepted']:
                accepted_count+=1;k=tr['accepted_candidate_index']
                check(status in ['ACCEPTED','REACQUIRED'] and isinstance(k,int) and 0<=k<len(cand),'accepted index invalid',context)
                if isinstance(k,int) and 0<=k<len(cand):
                    check(cand[k]['image_supported'] and same(tr['points_xy'],raw[k]['points_xy']),'accepted unsupported or invented path',context)
                    check(r['heatmap_entry_index']==raw[k]['entry_index'],'selected path/heatmap entrance mismatch',context)
                    if status=='ACCEPTED' and last_accepted is not None:
                        q=tr['candidate_diagnostics'][k]['temporal']
                        check(q['passed'] and not q['reasons'],'accepted a failed gate',context)
                        value=independent_lateral_p95(last_accepted['temporal']['points_xy'],tr['points_xy'],q['predicted_translation_xy'],protocol['gate']['sample_count'])
                        check(value is not None and abs(value-q['lateral_p95_abs_px'])<1e-5,'independent common-path distance differs',context)
                        check(value is not None and value<=q['lateral_limit_px']+1e-6,'excessive accepted lateral jump',context)
                if status=='REACQUIRED':
                    check(last_accepted is not None and t-last_accepted['source_time_s']>protocol['gate']['stale_seconds'],'reacquired before prior reference expiry',context)
                    check(tr.get('identity_continuity_through_gap') is False,'reacquisition represented as same identity',context)
                    past=tr.get('consulted_pending_history_times_s',[])
                    check(len(past)>=protocol['gate']['reacquire_frames']-1 and t-past[0]>=protocol['gate']['reacquire_minimum_seconds'],'reacquisition lacks causal confirmation span',context)
                    accepted_history=[];prev_epoch+=1
                accepted_history=(accepted_history+[r])[-protocol['gate']['history_frames']:];last_accepted=r
                check(tr['history_updated'] and tr['reference_points_xy'] is None,'accepted path/reference state invalid',context)
            else:
                rejected+=1
                check(not tr['history_updated'] and tr['points_xy'] is None and tr['accepted_candidate_index'] is None,'rejected observation written as accepted',context)
                if status=='STALE':
                    check(last_accepted is not None,'stale without accepted reference',context)
                    if last_accepted:
                        age=t-last_accepted['source_time_s']
                        check(age<=protocol['gate']['stale_seconds']+1e-8 and abs(age-tr['reference_age_s'])<1e-8 and same(tr['reference_points_xy'],last_accepted['temporal']['points_xy']),'stale reference changed/expired',context)
                else:check(tr['reference_points_xy'] is None,'unknown still exposes old reference as current',context)
            check(tr['epoch']==prev_epoch and tr['temporal_history_count']==len(accepted_history),'history epoch/count inconsistent',context)
            entry_count=diag['entry_count'];forward_count+=diag['forward_count']
            check(diag['forward_count']==entry_count==len(r['seeds']),'new forward count mismatch',context)
            for seed in r['seeds']:
                check(seed['automatic'] and seed['detected_time_s']<=t+1e-8 and abs(seed['last_current_image_check_time_s']-t)<1e-7,'entry not current-image supported or from future',context)
            if entry_count:
                path=Path(r['response_path']);check(sha(path)==r['response_sha256'],'saved heat response changed',context)
                with np.load(path,allow_pickle=False) as z:
                    check(set(z.files)=={f'entry_{j}' for j in range(entry_count)}|{'bbox_xyxy','source_frame_index','observation_time_s'},'response archive entries mismatch',context)
                    check(int(z['source_frame_index'])==i and abs(float(z['observation_time_s'])-t)<1e-8,'response belongs to other frame/time',context)
                    check(np.array_equal(z['bbox_xyxy'],[0,1050,4096,3000]),'response coordinate mapping changed',context)
                    for j in range(entry_count):
                        h=z[f'entry_{j}'];total_responses+=1
                        check(h.dtype==np.float32 and h.shape==(975,2048) and np.isfinite(h).all() and h.min()>=0 and h.max()<=1,'response invalid/range stretched',context)
                check(isinstance(r['heatmap_entry_index'],int) and 0<=r['heatmap_entry_index']<entry_count,'heat entry unavailable',context)
            else:check(r['response_path'] is None and r['heatmap_entry_index'] is None and r['heat_shape'] is None,'missing response replaced with synthetic heat',context)
            by_time[t]=r;last_t=t
        for j,r in enumerate(render):
            context=f'{ident}:output_frame={j}';i=r['source_frame_index'];a=r['last_analysis_frame'];t=times[i]
            check(r['output_frame_index']==j and abs(r['source_time_s']-t)<1e-8 and abs(r['media_time_s']-i/v['nominal_fps'])<1e-8,'render clock/index mapping incorrect',context)
            if a is None:
                check(i<anchor['source_frame'] and r['display_status']=='UNKNOWN' and r['heatmap_entry_index'] is None,'future scene/result shown before anchor',context)
                expected='UNKNOWN';draw_current=False;draw_reference=False;raw_index=None;heat_available=False
            else:
                check(a<=i and a in by_frame,'render used future/nonexistent detection',context)
                source=by_frame[a];tr=source['temporal'];age=t-source['source_time_s']
                check(abs(r['detection_age_s']-age)<1e-8 and age>=0,'wrong observation age',context)
                check(r['heatmap_entry_index']==source['heatmap_entry_index'],'render heat entrance changed',context)
                expected=tr['status'];ref_age=tr.get('reference_age_s')
                if expected=='STALE' and ref_age+age>protocol['gate']['stale_seconds']:expected='UNKNOWN'
                check(r['display_status']==expected,'stale display expiration/state mismatch',context)
                draw_current=expected in ['ACCEPTED','REACQUIRED'];draw_reference=expected=='STALE'
                raw_index=tr.get('accepted_candidate_index') if draw_current else None;heat_available=source['response_path'] is not None
                if draw_reference:reference_only+=1
                if draw_current:check(age<=.75,'current display held too long between analysis updates',context)
            if 'current_geometry_drawn' in r:
                check(r['current_geometry_drawn']==draw_current and r['reference_geometry_drawn']==draw_reference,'drawn geometry not consistent with state',context)
                check(r['raw_candidate_index']==raw_index and r['heat_available']==heat_available,'render path/heat availability mismatch',context)
        check(dict(counts)==vs['counts'] and forward_count==vs['forward_count'],'video summary count mismatch',ident)
        movie=cv2.VideoCapture(vs['output_path']);check(movie.isOpened(),'output movie cannot open',ident)
        check(int(movie.get(cv2.CAP_PROP_FRAME_COUNT))==len(render) and int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))==1600 and int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))==1280,'movie dimensions/framecount wrong',ident)
        check(abs(movie.get(cv2.CAP_PROP_FPS)-5)<1e-8,'movie fps wrong',ident)
        cached=cached_decodes.get(ident)
        reuse=bool(cached and cached['video_sha256']==actual_movie_sha and cached['output_frames']==len(render))
        decoded=0
        if reuse:decoded=cached['output_frames']
        else:
            while True:
                ok,frame=movie.read()
                if not ok:break
                check(frame.shape==(1280,1600,3),'decoded output frame shape wrong',ident);decoded+=1
        movie.release();check(decoded==len(render),'output movie truncated or undecodable',ident)
        check(abs(vs['output_duration_s']-len(render)/5)<1e-8 and abs(vs['output_duration_s']-vs['source_duration_s'])<=.2+1/v['nominal_fps'],'output speed/duration changed',ident)
        total_forwards+=forward_count
        per_video.append(dict(video_id=ident,analysis_frames=len(det),output_frames=decoded,forwards=forward_count,candidates=candidate_count,
            accepted=accepted_count,rejected=rejected,states=dict(counts),upstream_selected_lacked_current_support=baseline_unsupported,
            stale_reference_display_frames=reference_only,video_sha256=vs['output_sha256'],
            decode_verification='reused_same_sha256_and_frozen_protocol' if reuse else 'all_frames_decoded_now',
            decode_receipt=cached['receipt'] if reuse else None,
            detection_log_sha256=sha(folder/'detections.jsonl'),render_log_sha256=sha(folder/'render_frames.jsonl'),video_summary_sha256=sha(folder/'summary.json')))
    return dict(status='passed' if not errors else 'failed',exit_code=int(bool(errors)),checks=checks,errors=errors,warnings=warnings,
        run=str(run),command=sys.argv,audit_source_sha256=sha(__file__),production_modules_imported=False,
        protocol_sha256=sha(run/'protocol.json'),summary_sha256=sha(run/'summary.json') if summary is not None else None,source_hash_matches_current=source_matches,
        whole_run_checked=selected_videos is None,
        checked_videos=len(per_video),forward_count=total_forwards,float_response_arrays_checked=total_responses,videos=per_video,seconds=time.perf_counter()-tick,
        reused_decode_receipts=cache_receipts,
        scope='Stored causal state, current-image support flags/length, exact accepted candidate geometry, heat entrance, numeric arrays, source clocks, rendering records and complete output-video decode.',
        limits='No manual identity truth or video identity accuracy. Historical F3 scene propagation is not fresh complete F3/SAM2 inference. Stable wrong paths can pass. Output playback5Hz and analysis2Hz do not establish compute real-time performance.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--allow-display-code-drift',action='store_true');ap.add_argument('--video',action='append');ap.add_argument('--reuse-partial-decodes',action='store_true');a=ap.parse_args()
    run=HERE/a.run;result=audit(run,a.allow_display_code_drift,a.video,a.reuse_partial_decodes)
    if a.video:
        dest=run/'partial_audits';dest.mkdir(exist_ok=True)
        path=dest/('_'.join(a.video)+'_independent_audit.json')
    else:path=run/'independent_audit.json'
    if path.exists():raise FileExistsError('preserve audit history: '+str(path))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','checks','errors','warnings','checked_videos','forward_count','float_response_arrays_checked','seconds']},ensure_ascii=False))
    raise SystemExit(result['exit_code'])

if __name__=='__main__':main()
