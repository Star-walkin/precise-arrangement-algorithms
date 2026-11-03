"""Supplementary saved-record prefix audit; no predictor calls or edits."""
import argparse,json,math,sys
from pathlib import Path
import cv2,numpy as np
import whole_io as io
from audit_a6_outputs import independent_paint

def recompute_prefix(chosen,others,entry,params,full_pool=False):
    q=np.asarray(chosen['search_xy'],float);widths=np.asarray(chosen['search_widths']);ev=chosen['evidence'];w0=entry['free']['geometry']['width_px']
    unique=np.asarray(ev['neighbor_comparison_valid'],bool)&(np.asarray(ev['neighbor_advantage'])>params['trusted_unique_advantage'])
    allowed=np.asarray(chosen['direct_template_supported'],bool)&((np.asarray(chosen['angle_from_free_deg'])<3.)|unique)
    crossing=np.asarray(chosen['late_F3']['body_crossing_suspect'],bool);diverged=np.zeros(len(q),bool)
    for other in others:
        if other['candidate_id']==chosen['candidate_id']:continue
        if full_pool:
            p=np.asarray(other['xy']);direction=np.asarray(entry['free']['geometry']['direction_xy']);normal=np.array([-direction[1],direction[0]]);s=q@direction;other_s=p@direction
            if np.any(np.diff(other_s)<=0):raise ValueError('Full-pool path must progress monotonically along free axis')
            valid=(s>=other_s[0])&(s<=other_s[-1]);delta=abs(q@normal-np.interp(s,other_s,p@normal))
            right=np.clip(np.searchsorted(other_s,s,side='left'),0,len(other_s)-1);left=np.maximum(0,right-1);left=np.where(np.isclose(s,other_s[right],rtol=0.,atol=1e-6),right,left);flags=np.asarray(other['supported'],bool);supported=flags[right]&flags[left]
            diverged|=valid&supported&(delta>params['divergence_width_ratio']*widths)
        else:
            p=np.asarray(other['search_xy']);p=p[np.argsort(p[:,0])];valid=(q[:,0]>=p[0,0])&(q[:,0]<=p[-1,0]);delta=abs(q[:,1]-np.interp(q[:,0],p[:,0],p[:,1]));diverged|=valid&(delta>params['divergence_width_ratio']*widths)
    gaps=np.r_[0,np.linalg.norm(np.diff(q,axis=0),axis=1)];weak=split=0.;last=0;reason='image_support_endpoint'
    for i in range(1,len(q)):
        weak=0. if allowed[i] and not crossing[i] else weak+gaps[i];split=split+gaps[i] if diverged[i] else 0.
        if weak>params['trusted_maximum_gap_widths']*w0:reason='self_template_or_edges_no_longer_distinct';break
        if split>w0:reason='retained_branches_diverge';break
        if allowed[i] and not diverged[i]:last=i
    return max(1,last+1),reason

def reconstruct_external_free(shape,segment,surface,width):
    left=np.asarray(segment['boundary_left_xy']);right=np.asarray(segment['boundary_right_xy']);out=np.zeros(shape,np.uint8);gap=max(6.,width*.9)
    for i in range(len(left)-1):
        if max(np.linalg.norm(left[i+1]-left[i]),np.linalg.norm(right[i+1]-right[i]))>gap:continue
        cv2.fillConvexPoly(out,np.rint([left[i],right[i],right[i+1],left[i+1]]).astype(np.int32),255)
    out[surface!=0]=0;return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=io.HERE/a.run;output=run/'independent_prefix_logic_audit.json'
    if output.exists():raise ValueError('Preserve old receipt')
    start=io.read_json(run/'run_start.json');cases=[];cv2.setNumThreads(1)
    for inputid in start['input_ids']:
        rel=Path(inputid);saved=io.read_json(run/'记录'/rel.parent.name/(rel.stem+'.json'));result=saved['result'];record={'input':inputid,'errors':[],'status':result['status'],'branch_count':len(result['candidates'])};errors=record['errors']
        if not result['candidates']:record['skipped_reason']='No selected search path; main mask contract already checked';cases.append(record);continue
        chosen=next(c for c in result['candidates'] if c['candidate_id']==result['selected_candidate_id']);entries={e['entry_id']:e for e in result['entries']};entry=entries[chosen['entry_id']];params=result['diagnostics']['parameters']
        full=result.get('candidate_pool');new_policy=full is not None
        if new_policy:
            minimum=params['minimum_branch_span_widths']*entry['free']['geometry']['width_px'];others=[c for c in full if c['entry_id']==chosen['entry_id'] and c['length_px']>=minimum]
            expected_ids=[c['candidate_id'] for c in others]
            if expected_ids!=result['diagnostics'].get('prefix_competitor_ids'):errors.append('prefix_competitor_coverage_not_complete')
            record.update(prefix_scope='all_eligible_same_entry_image_generated_paths',full_pool_count=len(full),considered_candidate_ids=expected_ids,non_displayed_competitor_ids=[c['candidate_id'] for c in others if c['candidate_id'] not in {v['candidate_id'] for v in result['candidates']}])
        else:
            others=[c for c in result['candidates'] if c['entry_id']==chosen['entry_id'] and chosen['late_combined_score']-c['late_combined_score']<params['competing_score_margin']];record['prefix_scope']='legacy_display_pool_and_tail_score_margin'
        count,reason=recompute_prefix(chosen,others,entry,params,full_pool=new_policy)
        if len(entries)>1:count=0;reason='multiple_independent_free_entries'
        if count!=result['diagnostics']['trusted_search_point_count']:errors.append('trusted_search_count_does_not_follow_saved_evidence')
        if reason!=result['diagnostics']['trusted_prefix_stop_reason']:errors.append('trusted_stop_reason_mismatch')
        surface=io.read_image(io.C4/'roi/surface_masks'/rel.parent.name/rel.name,0);trusted=io.read_image(run/'01_可信前缀掩膜'/rel.parent.name/rel.name,0);best=io.read_image(run/'04_最佳外观候选掩膜'/rel.parent.name/rel.name,0);width=entry['free']['geometry']['width_px']
        if count:
            prefix=independent_paint(surface.shape,chosen['search_xy'][:count],chosen['search_widths'][:count]);external=reconstruct_external_free(surface.shape,entry['free']['free_segment'],surface,width);expected=(prefix|external)&best
        else:expected=np.zeros_like(surface)
        mismatch=int(np.count_nonzero(expected!=trusted));record.update(trusted_search_points=count,expected_trusted_pixels=int(np.count_nonzero(expected)),trusted_mask_difference_pixels=mismatch)
        if mismatch:errors.append('trusted_mask_not_saved_measured_free_plus_evidence_prefix')
        bridge=entry.get('search',{}).get('axial_reacquisition',{});record['chosen_entry_axis_reacquisition_accepted']=bool(bridge.get('accepted'))
        if bridge.get('accepted'):
            q=np.asarray(bridge['xy']);support=np.asarray(bridge['supported'],bool);false=np.diff(np.r_[False,~support,False].astype(np.int8));runs=[]
            for lo,hi in zip(np.flatnonzero(false==1),np.flatnonzero(false==-1)):
                length=float(np.linalg.norm(np.diff(q[lo:hi],axis=0),axis=1).sum())
                if length<6*width:continue
                # Inspect the middle of the gap to exclude raster overlap at
                # its two observed endpoints; a trusted prefix must not jump it.
                arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q[lo:hi],axis=0),axis=1))];central=q[lo:hi][(arc>width)&(arc<length-width)]
                green=cv2.remap(trusted,central[:,0].astype(np.float32)[None,:],central[:,1].astype(np.float32)[None,:],cv2.INTER_NEAREST).ravel()>0 if len(central) else []
                runs.append({'unsupported_gap_px':length,'central_axis_samples':len(central),'central_axis_trusted_samples':int(np.sum(green))})
                if np.any(green):errors.append('long_unknown_axis_gap_marked_trusted')
            record['long_axis_gaps']=runs
        cases.append(record)
    passed=all(not c['errors'] for c in cases);report={'scope':'Independent saved-evidence prefix/count/mask reconstruction and axial long-gap trust check. Does not decide whether appearance support tracks the true rope.','run':a.run,'passed':passed,'checked_cases':len(cases),'selected_path_cases':sum(c['branch_count']>0 for c in cases),'accepted_axis_bridge_cases':sum(c.get('chosen_entry_axis_reacquisition_accepted',False) for c in cases),'long_gap_cases':sum(bool(c.get('long_axis_gaps')) for c in cases),'cases':cases,'source_sha256':io.sha(__file__)}
    io.dump(output,report);print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False));return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
