"""BG6.2: local brightness-reference contamination fix, chronological rerun.

Only additional rejections are introduced. Already computed spatial outputs
are reused from the same source frame when every original ROI pixel is kept;
no new ROI or curve is fabricated. GV1 and counter thresholds stay frozen.
"""
import sys,time,gzip,json,copy
import cv2
from common import *
from local_foreground_v2 import local_count_guard
from geometry_gv1 import assess
from run_full import temporal_step,e

def refine(im,r):
    reg=r['region'];proposal=r.get('region_BG5') or record(r['frame_index'])['region_BG5']
    local=local_count_guard(im,proposal);prior=reg['identity_guard'];prior_reasons=list(reg['reasons'])
    reg['identity_guard_v1']=prior;reg['local_count_guard_v2']=local
    combined=copy.deepcopy(local)
    combined['foreground_polygons_xy']=prior.get('foreground_polygons_xy',[])+local.get('foreground_polygons_xy',[])
    combined['rejected_column_ranges_xy']=prior.get('rejected_column_ranges_xy',[])
    combined['valid']=prior['valid'] and local['valid'];reg['identity_guard']=combined
    reg['version']='BG6.2_current_local_foreground_guard'
    reg['reasons']=list(dict.fromkeys(prior_reasons+local['reasons']));reg['valid']=reg['valid'] and local['valid']
    if not reg['valid']:
        reg['edge_y']=None;reg['top_curve_xy']=[];reg['top_curve_supported_segments']=[]
        reg['region_status']='current_count_neighborhood_unavailable'
        r['BG4']=None;r['observation']=None;r['fresh_F3']=dict(called=False,point_count=0,raw_points=[],BG3_curve_count=0)
    r['region']=reg;r['region_BG6']=reg;r['region_gate_passed']=reg['valid']
    r['geometry_GV1']=assess(r['S1_tracker_full'],reg,read(ROOT/'d_calibration/receipt.json'))
    r['version']='RES-1.3/BG6.2-GV1';return r

def main():
    cv2.setNumThreads(1);prior=ROOT/'runs/dual_20260923_113745';out=ROOT/'runs/dual_20260923_113745_refined'
    out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir()
    bind=read(prior/'binding.json');bind.update(algorithm='RES-1.3/BG6.2-GV1',refinement_code_sha256=sha(__file__),
        local_foreground_code_sha256=sha(ROOT/'local_foreground_v2.py'),
        refinement_reason='reference contamination observed in full first-pass review; cap brightness reference and test first6 pitches',
        additional_development_frames=[368,388,392,400],first_pass_binding_sha256=sha(prior/'binding.json'))
    dump(out/'binding.json',bind);cap=cv2.VideoCapture(bind['source']);rows=[];counter=e.Counter();tick=time.perf_counter()
    for i in range(600):
        deadline=time.monotonic()+600
        while not (prior/'frames'/f'{i:06d}.json.gz').exists():
            assert time.monotonic()<deadline;time.sleep(.2)
        ok,im=cap.read();assert ok;r=record(i,prior);assert pixelsha(im)==r['source_pixel_sha256']
        oldvalid=r['region']['valid'];oldF3=r['provenance_R13']['F3_mode'];r=refine(im,r)
        r['refinement_provenance']=dict(first_pass_record_sha256=sha(prior/'frames'/f'{i:06d}.json.gz'),
            local_guard_code_sha256=sha(ROOT/'local_foreground_v2.py'),boundary_first_pass_valid=oldvalid,
            F3_first_pass_mode=oldF3,ROI_changed_in_refinement=False,
            F3_mode='source_bound_first_pass_reuse_no_ROI_change' if r['region']['valid'] else 'not_called_refined_boundary_rejected')
        # Restore spatial observation rather than feeding first-pass posterior.
        r['observation']=r['pre_GV1_observation'] if r['region']['valid'] else None
        r['temporal']=temporal_step(counter,r)
        path=out/'frames'/f'{i:06d}.json.gz'
        with gzip.open(path.with_suffix('.part'),'wt',encoding='utf-8',compresslevel=3) as f:json.dump(plain(r),f,ensure_ascii=False,allow_nan=False)
        path.with_suffix('.part').replace(path)
        rows.append(dict(frame=i,time_s=i/10,tracking_valid=r['s1']['valid'],region_valid=r['region']['valid'],
            foreground_rejected='broad_foreground_occludes_count_interface' in r['region']['reasons'],
            geometry_valid=r['geometry_GV1']['conditional_projection_valid'],geometry_reasons=r['geometry_GV1']['reasons'],
            count_state=r['temporal']['state'],value=r['temporal']['value'],display_N=r['temporal']['display_N'],
            F3_mode=r['refinement_provenance']['F3_mode'],points=r['fresh_F3']['point_count'],
            old_region_valid=r['baseline_RES12']['region_valid'],old_value=r['baseline_RES12']['count']['value'],
            raw_gap=r['geometry_GV1'].get('raw_gap_px')))
        if i%50==0 or i==599:
            v=dict(processed=len(rows),planned=600,foreground_rejected=sum(q['foreground_rejected'] for q in rows),
                region_valid=sum(q['region_valid'] for q in rows),projection_pass=sum(q['geometry_valid'] for q in rows),
                numeric=sum(q['value'] is not None for q in rows),elapsed_s=round(time.perf_counter()-tick,1))
            dump(out/'progress.json',v);print(v,flush=True)
    cap.release();dump(out/'compact.json',rows)
    assert read(prior/'receipt.json')['status']=='complete'
    assert sha(bind['source'])==bind['source_sha256'];e.verify_baseline()
    dump(out/'receipt.json',dict(status='complete',frames=600,source_unchanged=True,baseline_unchanged=True,
        new_local_guards=600,unchanged_spatial_reuse_only=True,elapsed_s=time.perf_counter()-tick,
        d_calibration_valid=False,absolute_count_accuracy_validated=False))
if __name__=='__main__':main()
