"""Compare image-only and temporal choices on an identical four-path domain."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,sys
import numpy as np
from audit_run import read,rows,arc,sample,sha
from temporal_comparison import valid_index,quantiles

HERE=Path(__file__).resolve().parents[1]

def paired_distances(paths):
    """Paths are baseline-old/new and temporal-old/new, in original pixels.

    Shared entrance-oriented arclength coordinate, with only axial entrance
    offsets. All four paths share the exact sample positions and normals.
    No lateral alignment, no nearest-point matching, no extrapolated tails.
    """
    p=[np.asarray(x,float) for x in paths];lengths=[arc(x)[-1] for x in p]
    base=p[0];length=lengths[0]
    axis=sample(base,[0,min(length,min(200,max(20,length*.25)))])[1]-base[0]
    axis/=max(np.linalg.norm(axis),1e-9)
    offsets=[float((x[0]-base[0])@axis) for x in p]
    low=max(offsets);high=min(o+l for o,l in zip(offsets,lengths))
    common=high-low
    if common<80 or common/max(min(lengths),1e-9)<.55:return None
    grid=np.linspace(low,high,96)
    sampled=[sample(x,grid-o) for x,o in zip(p,offsets)]
    tangent=np.gradient(sampled[0],axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
    normal=np.stack([-tangent[:,1],tangent[:,0]],axis=1)
    baseline=np.abs(np.sum((sampled[1]-sampled[0])*normal,axis=1))
    temporal=np.abs(np.sum((sampled[3]-sampled[2])*normal,axis=1))
    return float(np.quantile(baseline,.95)),float(np.quantile(temporal,.95)),float(common)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    run=HERE/a.run;protocol=read(run/'protocol.json');summary=read(run/'summary.json')
    assert summary['completed_videos']==summary['planned_videos']==len(protocol['videos']) and not summary['failures']
    all_base=[];all_temporal=[];all_common=[];per_video=[];states=Counter();total_guard=total_different=total_reject=0;comparison_signs=Counter()
    paired_rows=[];all_processing=[];total_observations=0
    for vid in protocol['videos']:
        detections=rows(run/vid/'detections.jsonl');base_values=[];temporal_values=[];shared_lengths=[];excluded=0
        total_observations+=len(detections)
        states.update(r['temporal']['status'] for r in detections)
        for r in detections:
            guarded=valid_index(r);tr=r['temporal'];all_processing.append(r['processing_elapsed_s'])
            total_guard+=guarded is not None
            if tr['accepted']:total_different+=tr['accepted_candidate_index']!=guarded
            elif guarded is not None:total_reject+=1
        for old,new in zip(detections[:-1],detections[1:]):
            io,inn=valid_index(old),valid_index(new);to,tn=old['temporal'],new['temporal']
            if io is None or inn is None or not (to['accepted'] and tn['accepted'] and to['epoch']==tn['epoch']):continue
            distances=paired_distances([old['gate_candidates'][io]['points_xy'],new['gate_candidates'][inn]['points_xy'],to['points_xy'],tn['points_xy']])
            if distances is None:excluded+=1;continue
            b,t,c=distances;dt=new['source_time_s']-old['source_time_s'];b/=dt;t/=dt
            base_values.append(b);temporal_values.append(t);shared_lengths.append(c)
            sign='reduced' if t<b-1e-6 else ('increased' if t>b+1e-6 else 'unchanged');comparison_signs[sign]+=1
            paired_rows.append(dict(video_id=vid,old_frame=old['source_frame_index'],new_frame=new['source_frame_index'],dt_s=dt,
                shared_arclength_px=c,baseline_px_per_s=b,temporal_px_per_s=t))
        all_base.extend(base_values);all_temporal.extend(temporal_values);all_common.extend(shared_lengths)
        per_video.append(dict(video_id=vid,matched_current_pairs=len(base_values),excluded_insufficient_four_path_common_domain=excluded,
            baseline_lateral_px_per_s=quantiles(base_values),temporal_lateral_px_per_s=quantiles(temporal_values),common_domain_length_px=quantiles(shared_lengths),
            detections_sha256=sha(run/vid/'detections.jsonl')))
    row_path=run/'temporal_common_domain_pairs.jsonl';target=run/'temporal_comparison_common_domain.json'
    if target.exists() or row_path.exists():raise FileExistsError('Preserve comparison history')
    row_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in paired_rows),encoding='utf8')
    result=dict(status='complete',version='four_path_common_domain_v1',command=sys.argv,source_sha256=sha(__file__),protocol_sha256=sha(run/'protocol.json'),
        supersedes_for_paired_distance_only='temporal_comparison.json, whose frame pairs match but each method uses its own overlapping path range',
        observations=total_observations,states=dict(states),image_guard_candidate_observations=total_guard,
        accepted_current_observations=states['ACCEPTED']+states['REACQUIRED'],accepted_different_candidate_than_guard=total_different,
        rejected_despite_current_image_guard_candidate=total_reject,matched_current_pairs=len(all_base),pair_change_counts=dict(comparison_signs),
        baseline_lateral_px_per_s=quantiles(all_base),temporal_lateral_px_per_s=quantiles(all_temporal),common_domain_length_px=quantiles(all_common),
        processing_seconds=quantiles(all_processing),paired_rows_path=str(row_path),paired_rows_sha256=sha(row_path),videos=per_video,
        method='Only identical adjacent observed-frame pairs with current accepted geometry in one reference epoch. Sample all four paths on exactly the same entrance-oriented arclength interval and baseline-old normal field. Axial start offsets only; no lateral alignment, nearest-point matching or extrapolated tails. Minimum shared80px and55% of shortest path.',
        limits='Geometric continuity only. Rejections, held references, and new reference epochs are excluded from paired displacement and counted separately; they are not correct detections. A stable wrong rope can pass. No material identity accuracy, physical speed calibration, or independent rope truth.')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['observations','states','matched_current_pairs','pair_change_counts','baseline_lateral_px_per_s','temporal_lateral_px_per_s','accepted_different_candidate_than_guard','rejected_despite_current_image_guard_candidate']},ensure_ascii=False))

if __name__=='__main__':main()
