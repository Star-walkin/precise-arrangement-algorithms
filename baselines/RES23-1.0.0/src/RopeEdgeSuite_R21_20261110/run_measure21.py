import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,concurrent.futures as cf,collections
from common21 import *
from measurement21 import evaluate,DynamicGate

def process(item):
    id=item['id'];out=ROOT/'measurements'/id;calpath=R20/'calibration'/f'{id}.json';cal=read(calpath);gate=DynamicGate();compact=[];mismatches=[];start=time.time()
    for i in range(item['planned_frames']):
        oldpath=R20/'counts_v1'/id/'frames'/f'{i:06d}.json.gz';old=gzread(oldpath);r=frontend(id,i)
        fp=R20/'f3_v2'/id/'records'/f'{i:06d}.json';fr=read(fp);gp=R20/'groups_v2'/id/'frames'/f'{i:06d}.json.gz';geop=R20/'geometry_v5'/id/'frames'/f'{i:06d}.json.gz'
        assert sha(fp)==old['F3_record_sha256'];assert sha(gp)==old['groups_file_sha256'];assert sha(geop)==old['geometry_file_sha256']
        assert sha(calpath)==old['calibration_file_sha256'];assert r['source_pixel_sha256']==old['source_pixel_sha256']
        data=None
        if fr['called']:
            assert sha(fr['fields_path'])==fr['fields_sha256']
            with np.load(fr['fields_path']) as z:data={k:z[k] for k in z.files}
        groups=gzread(gp)['groups'];geo=gzread(geop);phase=next(c for c in cal['epochs'] if c['start']<=i<c['end'])
        m,diag=evaluate(groups,data,r['boundary'],geo,phase,gate,i/item['fps'],phase['epoch'])
        baseline=old['measurement'];same=True
        if phase['d'] is not None:
            for key in ('available','N','N_interval','complete_F3_phase','observed_groove_count'):
                if m.get(key)!=baseline.get(key):mismatches.append(dict(frame=i,key=key,old=baseline.get(key),new=m.get(key)));same=False
        row=dict(old,measurement=m,axis=geo.get('axis'),dynamic_diagnostic=diag,R21_baseline_record_sha256=sha(oldpath),R21_fixed_baseline_equal=same)
        dst=out/'frames'/f'{i:06d}.json.gz';gzwrite(dst,row)
        c=m.get('central') or {};a=geo.get('axis') or {};b=c.get('boundary') or {};abc=a.get('line_abc')
        compact.append(dict(frame=i,time_s=i/item['fps'],epoch=phase['epoch'],phase_direction=old['phase_direction'],
            available=m['available'],N=m.get('N'),N_interval=m.get('N_interval'),d=m.get('d'),d_interval=m.get('d_interval'),
            measurement_mode=m['measurement_mode'],fixed_d=m['fixed_d'],complete_F3_phase=m.get('complete_F3_phase',False),
            observed_grooves=m.get('observed_groove_count'),current_F3_called=fr['called'],F3_point_count=fr['points'],
            current_coil_side_axis=a.get('current_coil_side_measured',False),axis_slope=-abc[0]/abc[1] if abc else None,
            axis_uncertainty_px=a.get('uncertainty_px'),axis_width_px=a.get('width_px'),pitch=c.get('local_pitch_px'),
            boundary_occluded=b.get('occluded'),boundary_local=b.get('locally_observed'),boundary_uncertainty_px=b.get('uncertainty_px'),
            boundary_anchor_hull=b.get('inside_current_anchor_hull'),contact_gap=c.get('contact_projected_gap_N'),reasons=m['reasons'],
            source_pixel_sha256=old['source_pixel_sha256'],measurement_record_sha256=sha(dst)))
        if i%100==0:print(id,i,'elapsed',round(time.time()-start),flush=True)
    dump(out/'frames.json',compact)
    summary=dict(id=id,frames=len(compact),numeric=sum(r['available'] for r in compact),dynamic=sum(r['measurement_mode']=='current_dynamic_column' for r in compact),
        fixed_measurement_regressions=mismatches,baseline_preserved=not mismatches,reason_counts=dict(collections.Counter(k for r in compact if not r['available'] for k in r['reasons'])),elapsed_s=time.time()-start)
    dump(out/'summary.json',summary);print(summary,flush=True);return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--targets',nargs='*');a=ap.parse_args()
    jobs=[v for v in plan() if not a.targets or v['id'] in a.targets]
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:result=list(pool.map(process,jobs))
    dump(ROOT/'measurement_receipt.json',dict(videos=result,code_hashes={f:sha(ROOT/f) for f in ['measurement21.py','run_measure21.py','common21.py']},all_fixed_values_unchanged=all(v['baseline_preserved'] for v in result)))
