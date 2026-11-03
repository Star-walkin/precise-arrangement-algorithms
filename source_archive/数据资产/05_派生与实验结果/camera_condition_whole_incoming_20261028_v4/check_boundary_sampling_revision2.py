"""Pure synthetic checks for A4 entry/probe/domain handling; no real labels."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import cv2
import boundary_occlusion_revision2 as b

HERE=Path(__file__).resolve().parent

def main():
    tests={}
    shape=(800,900);S=np.zeros(shape,np.uint8);S[100:760,200:850]=255
    origin=np.array([20.,20.]);d=np.array([1.,1.])/np.sqrt(2);normal=np.array([-d[1],d[0]]);end=np.array([600.,600.]);pitch=20.;width=10.
    refs=[{'curve_id':f'h{i}','source':'synthetic','sampled_points_xy':np.c_[np.arange(100.,851.,5.),np.full(151,y)].tolist()} for i,y in enumerate([250.,300.,350.,450.])]
    groups,diag=b._seed_rows(refs,origin,d,normal,end,width,pitch,S)
    assert diag['retained_before_old_lower']>0 and diag['entry_sampling_domain']['lower_s_px']<diag['entry_sampling_domain']['old_lower_s_px']
    assert all(g[0]['extrapolated_x_px']<=6*pitch for g in groups)
    tests['pre_endpoint_crossings_restored']={'passed':True,'restored':diag['retained_before_old_lower'],'entry':diag['entry_sampling_domain']}

    S2=np.full(shape,255,np.uint8);xs=np.arange(20.,781.,2.);ys=.001*(xs-400)**2+80
    refs2=[{'curve_id':'one_visual_row_two_crossings','source':'synthetic','sampled_points_xy':np.c_[xs,ys].tolist()}]
    groups2,diag2=b._seed_rows(refs2,np.array([0.,120.]),np.array([1.,0.]),np.array([0.,1.]),np.array([600.,120.]),10.,20.,S2)
    assert len(set(g[0]['visual_row_id'] for g in groups2))==1 and len(groups2)>=2
    assert np.ptp([g[0]['free_s'] for g in groups2])>300
    tests['distinct_probes_one_visual_row']={'passed':True,'probes':len(groups2),'s':[g[0]['free_s'] for g in groups2]}

    img=np.full((*shape,3),170,np.uint8);img[147:153,:,:]=55
    ref3=[{'curve_id':'parallel','source':'synthetic','sampled_points_xy':[[float(x),150.] for x in range(20,850,3)]}]
    geo={'status':'observed','origin_xy':[20.,150.],'direction_xy':[1.,0.],'width_px':10.,'visible_end_xy':[400.,150.]}
    out=b.analyze_boundary_occlusion(img,S2,np.zeros(shape,np.uint8),np.zeros(shape,np.uint8),ref3,geo,pitch=20.)
    assert out['rows'] and all(r['original_evidence_state']=='unknown_original_evidence' and r['transition_score']==0 for r in out['rows'])
    assert out['diagnostics']['crossing_evidence']['chain_count']==0
    tests['parallel_not_two_sided_interruption']={'passed':True,'states':[r['near_tangent_state'] for r in out['rows']]}

    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    fields={'x0':0,'y0':0,'original_gray':g,'normalized_gray':(g-100)/30,'gradient_x':cv2.Sobel(g,cv2.CV_32F,1,0)/8,'gradient_y':cv2.Sobel(g,cv2.CV_32F,0,1)/8}
    no_domain=np.zeros(shape,np.uint8)
    p=b._profile(fields,np.array([400.,150.]),np.array([1.,0.]),np.array([.7,.7]),20.,S2,10.,no_domain)
    assert p['raw_available'] and not p['domain_available'] and not p['available'] and not p['original_groove_support']
    tests['outside_domain_not_evidence']={'passed':True,'raw_available':p['raw_available'],'domain_fraction':p['search_domain_fraction']}

    def row(i,visual,state,near=False):return {'row_id':f'p{i}','visual_row_id':visual,'free_axis_s_px':i*20.,'near_parallel_overlap':near,'geometry_kind':'tangential_strip_approach' if near else 'within_reference_span','original_evidence_state':state}
    rows=[row(0,'a','raw_interruption_supported'),row(1,'a','raw_interruption_supported'),row(2,'b','raw_continuous')]
    b._assign_crossing_evidence(rows);assert not rows[-1]['interruption_then_continuity_chain']
    rows=[row(0,'a','raw_interruption_supported'),row(1,'b','raw_interruption_supported'),row(2,'c','raw_continuous')]
    b._assign_crossing_evidence(rows);assert rows[-1]['interruption_then_continuity_chain']
    rows.insert(2,row(1.5,'u','unknown_original_evidence'));b._assign_crossing_evidence(rows);assert not rows[-1]['interruption_then_continuity_chain']
    tests['chain_unique_rows_and_unknown_break']={'passed':True}

    many=[]
    for i in range(100):
        for k in ('within_reference_span','tangential_strip_approach','finite_reference_extrapolation'):
            many.append([{'visual_row_id':str(i),'geometry_kind':k,'free_s':float(i*10)}])
    kept,omitted=b._fair_probe_budget(many,64,500.)
    assert len(kept)==64 and len(omitted)==236 and len(set(b._probe_family(g[0]['geometry_kind']) for g in kept))==3
    tests['fixed_budget_fair_families']={'passed':True,'kept':len(kept),'omitted':len(omitted)}

    old=HERE.parent/'camera_condition_whole_incoming_20261028_v3/boundary_occlusion.py'
    spec=importlib.util.spec_from_file_location('a3_boundary_frozen_readonly',old);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert b.PARAMETERS==mod.PARAMETERS
    tests['original_image_thresholds_unchanged']={'passed':True}
    report={'scope':'synthetic mechanism checks only, not real-image accuracy','passed':True,'tests':tests,'source_sha256':hashlib.sha256((HERE/'boundary_occlusion_revision2.py').read_bytes()).hexdigest()}
    path=HERE/'boundary_sampling_synthetic_revision2.json'
    if path.exists():raise ValueError('Never overwrite check receipt')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
