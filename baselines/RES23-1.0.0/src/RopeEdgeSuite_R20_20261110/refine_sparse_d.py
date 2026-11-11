"""Densify an already located short far-reference window, without lowering gates."""
from common20 import *
from calibration21_dense import collect,calibrate
from calibrate_guard import observe

def main():
    id='dual_20260923_113745__left';item=next(x for x in plan() if x['id']==id);old=read(ROOT/'calibration'/f'{id}.json');phase=old['epochs'][1]
    refs=phase['far_guard_accepted_frames'];assert phase['d'] is None and len(refs)==5 and max(refs)-min(refs)>=20
    bounds=[min(refs),min(phase['end'],max(refs)+5)];pool=set(range(*bounds));candidates,_=collect(item,phase,pool);allowed=set();audit=[]
    for row in candidates:
        q=observe(id,row['frame'],dict(d=row['d'],interval=row['interval'],status='conditional_geometry_estimate'))
        passed=bool(q['available'] and q['N_interval'][0]>=5);audit.append(dict(frame=row['frame'],phase='dense_dynamic_column_screen',passed=passed,count=q))
        if passed:allowed.add(row['frame'])
    final=calibrate(item,{k:phase[k] for k in ('epoch','start','end','direction')},allowed)
    for iteration in range(3):
        if final['d'] is None:break
        rejected=[]
        for i in sorted(allowed):
            q=observe(id,i,final);passed=bool(q['available'] and q['N_interval'][0]>=5);audit.append(dict(frame=i,phase=f'fixed_column_audit_{iteration}',passed=passed,count=q))
            if not passed:rejected.append(i)
        if not rejected:break
        allowed.difference_update(rejected);final=calibrate(item,{k:phase[k] for k in ('epoch','start','end','direction')},allowed)
    else:final.update(status='unidentified',d=None,interval=None,reasons=final['reasons']+['far_frame_screen_not_converged'])
    final.update(current_F3_five_rope_audit=audit,far_guard_accepted_frames=sorted(allowed),d_original_image=final['d'],d_interval_original_image=final['interval'],
        dense_reference_window_frames=bounds,sampling_policy='all_current_frames_in_previously_located_far_window; unchanged 2-second total span and holdout gates')
    updated=dict(old,epochs=[old['epochs'][0],final],refinement=dict(code_sha256=sha(__file__),calibrator_sha256=sha(ROOT/'calibration21_dense.py'),previous_calibration_sha256=sha(ROOT/'calibration'/f'{id}.json')))
    out=ROOT/'calibration_dense_candidate';out.mkdir(exist_ok=True);dump(out/f'{id}.json',updated)
    print(final['status'],final['d'],final['interval'],final['far_guard_accepted_frames'],final.get('temporal_holdout'),flush=True)

if __name__=='__main__':main()
