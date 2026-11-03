"""Descriptive temporal comparison, never a material-identity accuracy score.

Reads existing output logs only. Rejection/old reference is kept separate from
current observation, and paired displacement comparisons use identical frame
pairs. No future geometry feeds the pipeline; this is post-run accounting.
"""
from pathlib import Path
from collections import Counter
import argparse, json, sys, hashlib
import numpy as np
from audit_run import read, rows, arc, sample, independent_lateral_p95

HERE=Path(__file__).resolve().parents[1]

def quantiles(values):
    if not values:return dict(count=0,median=None,p95=None,max=None)
    a=np.asarray(values,float)
    return dict(count=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(np.max(a)))

def valid_index(r):
    if 'image_guard_only_selected_index' in r:return r['image_guard_only_selected_index']
    return next((k for k,c in enumerate(r['gate_candidates']) if c['image_supported'] and arc(np.asarray(c['points_xy'],float))[-1]>=80),None)

def compare_paths(old,new):
    a,b=np.asarray(old,float),np.asarray(new,float)
    la,lb=arc(a)[-1],arc(b)[-1]
    tangent=sample(a,[0,min(la,min(200,max(20,la*.25)))])[1]-a[0]
    tangent/=max(np.linalg.norm(tangent),1e-9)
    offset=float((b[0]-a[0])@tangent)
    overlap=min(la-max(offset,0),lb-max(-offset,0))
    if overlap<80 or overlap/max(min(la,lb),1e-9)<.55:return None
    return independent_lateral_p95(a,b,[0.,0.],96)

def summarize_video(folder):
    det=rows(folder/'detections.jsonl');render=rows(folder/'render_frames.jsonl');vs=read(folder/'summary.json')
    statuses=Counter(r['temporal']['status'] for r in det)
    reasons=Counter();raw_supported=0;guarded=0;changed=0;unchanged=0;rejected_supported=0
    paired_baseline=[];paired_temporal=[];all_baseline=[];uncomparable=0;gaps=[]
    for r in det:
        tr=r['temporal'];idx=valid_index(r);c=r['gate_candidates']
        raw_supported+=bool(c and c[0]['image_supported']);guarded+=idx is not None
        if tr['accepted']:
            changed+=tr['accepted_candidate_index']!=idx;unchanged+=tr['accepted_candidate_index']==idx
        elif idx is not None:rejected_supported+=1
        for d in tr.get('candidate_diagnostics',[]):
            temporal=d.get('temporal') or {}
            for reason in temporal.get('reasons',[]):reasons[reason]+=1
    for a,b in zip(det[:-1],det[1:]):
        dt=b['source_time_s']-a['source_time_s'];ia,ib=valid_index(a),valid_index(b)
        gaps.append(dt)
        if ia is None or ib is None:continue
        bv=compare_paths(a['gate_candidates'][ia]['points_xy'],b['gate_candidates'][ib]['points_xy'])
        if bv is not None:all_baseline.append(bv/dt)
        else:uncomparable+=1
        ta,tb=a['temporal'],b['temporal']
        if not (ta['accepted'] and tb['accepted'] and ta['epoch']==tb['epoch']):continue
        tv=compare_paths(ta['points_xy'],tb['points_xy'])
        if bv is not None and tv is not None:
            paired_baseline.append(bv/dt);paired_temporal.append(tv/dt)
    displays=Counter(r['display_status'] for r in render)
    return dict(video_id=vs['video_id'],observations=len(det),states=dict(statuses),raw_rank0_current_supported_observations=raw_supported,
        image_guard_baseline_observations=guarded,temporal_accepts_same_as_image_guard=unchanged,
        temporal_accepts_different_candidate=changed,temporal_rejections_with_image_guard_candidate=rejected_supported,
        gate_reasons_by_candidate=dict(reasons),observation_dt_s=quantiles(gaps),
        baseline_all_consecutive_current_path_lateral_px_per_s=quantiles(all_baseline),
        paired_same_adjacent_current_observation_frames=dict(image_guard_baseline_lateral_px_per_s=quantiles(paired_baseline),
            temporal_lateral_px_per_s=quantiles(paired_temporal),exclude_rejections_reacquisition_epoch_changes_and_uncomparable_paths=True),
        uncomparable_baseline_adjacent_pairs=uncomparable,
        display_frames=dict(displays),display_playback_seconds={k:v/vs['output_fps'] for k,v in displays.items()},
        processing_time_s=quantiles([r['processing_elapsed_s'] for r in det]))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--completed-only',action='store_true');args=ap.parse_args()
    run=HERE/args.run;protocol=read(run/'protocol.json');ids=protocol['videos']
    if args.completed_only:ids=[i for i in ids if (run/i/'summary.json').exists()]
    result=dict(scope='Descriptive continuity and rejection accounting; not path identity, accuracy, or a claim that rejection is improvement.',
        histories_causal_in_pipeline=True,analysis_is_post_run=True,no_production_modules_imported=True,
        nominal_three_seconds_per_projected_pitch_is_an_uncalibrated_prior=True,
        common_path_metric='p95 absolute normal displacement over shared entrance-oriented arclength; small axial entrance offset accounted for, no lateral translation correction or nearest-neighbor alignment; minimum shared80px and55% of shorter visible path.',
        paired_comparison_limit='Only identical adjacent observation pairs with current accepted paths in the same epoch are compared. Rejected frames are counted separately; stable wrong geometry can still look favorable.',
        whole_run=not args.completed_only,video_count=len(ids),videos=[summarize_video(run/i) for i in ids],command=sys.argv,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    name='temporal_comparison.json' if not args.completed_only else f'temporal_comparison_completed_{len(ids):02d}.json'
    target=run/name
    if target.exists():raise FileExistsError(str(target))
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(path=str(target),videos=len(ids))))

if __name__=='__main__':main()
