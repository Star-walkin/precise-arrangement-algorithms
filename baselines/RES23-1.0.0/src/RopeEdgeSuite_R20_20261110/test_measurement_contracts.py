"""Adversarial contracts: observed counting, missing rows, contact and mirroring."""
import numpy as np
from common20 import *
from count20 import e
from temporal_contact20 import DistanceFilter,contact_evidence,windows

def run():
    passed=[]
    # The measured phase must equal 5.5 pitches, rather than rounding the
    # five interior grooves into a claimed integer distance.
    pitch=12.;surface=np.full((300,300),255,np.uint8);exclusion=np.zeros_like(surface)
    counter=e.local.SurfaceObserver(surface,pitch)
    crosses=[dict(y=float(y)) for y in np.arange(90,260,12)]
    edge=dict(edge_y=102.,edge_current_frame_confirmed=True)
    q,why,_=counter._count(150.,168.,crosses,edge,exclusion)
    assert q['count_valid'] and abs(q['continuous_pitch_distance']-5.5)<1e-8,(q,why)
    assert q['count_observed_grooves']==5
    passed.append('noninteger_measured_phase_and_interior_grooves_are_distinct')
    missing=[x for x in crosses if x['y']!=138.]
    q,why,_=counter._count(150.,168.,missing,edge,exclusion)
    assert not q['count_valid'] and q['continuous_pitch_distance'] is None
    assert 'unsupported_count_gap_gt_1_6_pitch' in why
    passed.append('missing_whole_groove_does_not_become_complete_phase')
    exclusion[120:138,150]=255
    q,why,_=counter._count(150.,168.,crosses,edge,exclusion)
    assert not q['count_valid'] and 'unknown_or_excluded_region_on_count_segment' in why
    passed.append('occlusion_blocks_observed_phase_even_with_fitted_curves')
    # Formula conversion and column conversion are tested together using
    # independent sampled points, including the image's last pixel.
    abc=np.array([-.17,.985,-1760.]);flipped=np.array([-abc[0],abc[1],abc[2]+4095*abc[0]])
    for d in (0.,1430.5,2100.,4095.):
        y=-(abc[0]*d+abc[2])/abc[1]
        assert abs(flipped@np.array([4095-d,y,1.]))<1e-8
    passed.append('right_camera_line_and_column_return_to_original_pixels')
    filt=DistanceFilter();good=dict(available=True,N=2.,N_interval=[1.8,2.2],d=2100)
    for i in range(10):filt.step(i*.1,0,good)
    missing=dict(available=False,d=2100,reasons=['current_evidence_missing'])
    assert filt.step(1.1,0,missing)['available']
    assert not filt.step(1.5,0,missing)['available']
    assert not filt.step(1.1,1,missing)['available']
    assert contact_evidence(missing,None)['state']=='unavailable'
    passed.append('short_prediction_expires_and_never_confirms_contact_or_crosses_epoch')
    base=dict(available=True,N=.5,N_interval=[.2,.8],contact_gap_interval=[-.2,.3],central=dict(boundary=dict(locally_observed=True,occluded=False)))
    axis=dict(current_coil_side_measured=True)
    out=contact_evidence(base,axis)
    assert out['state']=='visual_contact_candidate' and not out['physical_contact_confirmed']
    base['central']['boundary'].update(locally_observed=False,occluded=True)
    assert contact_evidence(base,axis)['state']=='near_edge_uncertain'
    passed.append('hidden_boundary_cannot_confirm_visual_or_material_contact')
    states=['visual_contact_candidate']*3+['unavailable']+['visual_contact_candidate']*3
    result=windows([dict(contact=dict(state=s)) for s in states],10)
    assert len(result)==2 and result[0]['end_frame']==2 and result[1]['start_frame']==4
    passed.append('contact_windows_do_not_bridge_missing_current_evidence')
    dump(ROOT/'measurement_contract_tests.json',dict(passed=passed,count=len(passed),test_sha256=sha(__file__)))
    print('PASS',len(passed),flush=True)

if __name__=='__main__':run()
