import copy
from common20 import *
from calibrate_d import frame_fit

def duplicate_branch_audit(p1):
    candidates=p1.get('candidate_paths_audit',[])
    if len(candidates)<2:return dict(resolved=False,reason='alternatives_not_recorded')
    best=candidates[0];near=[c for c in candidates if c['rank_score']>=best['rank_score']-.05 and c.get('continuation_eligible')]
    if not near:return dict(resolved=False,reason='no_image_eligible_alternatives')
    selected=np.asarray(best['points_xy']);stage=np.asarray(best['stage']);base=selected[stage=='surface'];records=[]
    if len(base)<6:return dict(resolved=False,reason='selected_curve_missing')
    base=base[np.argsort(base[:,0])]
    for c in near:
        p=np.asarray(c['points_xy']);st=np.asarray(c['stage']);p=p[st=='surface'];p=p[np.argsort(p[:,0])]
        if len(p)<6:return dict(resolved=False,reason='alternative_curve_missing')
        lo=max(base[0,0],p[0,0]);hi=min(base[-1,0],p[-1,0])
        if hi-lo<max(40.,.5*np.ptp(base[:,0])):return dict(resolved=False,reason='alternatives_do_not_overlap_enough')
        xx=np.arange(lo,hi,4.);delta=abs(np.interp(xx,base[:,0],base[:,1])-np.interp(xx,p[:,0],p[:,1]))
        records.append(dict(entry=c['entry'],index=c['index'],q90_difference_px=float(np.percentile(delta,90))))
    okay=all(r['q90_difference_px']<=.25*PITCH_PRIOR for r in records)
    return dict(resolved=okay,reason='same_current_visual_curve_geometry' if okay else 'distinct_competing_current_curves',comparisons=records)

def fit_row(row):
    p1=row['result'];branch=duplicate_branch_audit(p1);out=frame_fit(row);out['duplicate_branch_audit']=branch
    if branch['resolved'] and 'ambiguous_H6_connection' in out['reasons']:
        out['reasons'].remove('ambiguous_H6_connection');out['eligible']=not out['reasons']
    return out

def main():
    videos=[]
    for item in plan():
        rows=[fit_row(read(f)) for f in sorted((ROOT/'paths_original'/item['id']).glob('p1_*.json'))]
        videos.append(dict(id=item['id'],frames=rows))
        print(item['id'],[(r['frame'],round(r.get('d_candidate',0)),r['reasons']) for r in rows],flush=True)
    dump(ROOT/'d_frame_audit_original.json',dict(videos=videos,calibration_locked=False,frame_profile_is_not_multi_frame_calibration=True))
if __name__=='__main__':main()
