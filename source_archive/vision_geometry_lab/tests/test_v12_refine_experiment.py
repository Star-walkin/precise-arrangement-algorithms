import numpy as np
from vision_geometry_lab.v12_refine_experiment import gate_decisions,_shifted_reference,_resampled_inputs


def test_gate_rejects_damage_or_no_actual_cv_acceptance():
    def summary(coverage,precision):
        return [dict(scope='downstream',cohort='all',split='train',covered_fraction=coverage,prediction_precision_at_5px=precision)]
    source=dict(summaries={'original_v11':summary(1.,1.),'bright_ridge':summary(.97,1.),'paired_edges':summary(1.,1.)},
                counts={'bright_ridge':{'accepted_points':10},'paired_edges':{'accepted_points':0}})
    protocol={'gate':{'train_coverage_max_drop':.01,'train_precision_max_drop':.01}}
    assert not any(r['passed'] for r in gate_decisions(source,protocol))
    source['counts']['paired_edges']['accepted_points']=5
    assert gate_decisions(source,protocol)[1]['passed']


def test_gt_stress_is_directional_and_width_scaled_not_a_fixed_translation():
    source=[[[0.,10.],[100.,10.]]]
    plus=np.asarray(_shifted_reference(source,20.,.45)[0])
    minus=np.asarray(_shifted_reference(source,20.,-.45)[0])
    assert np.allclose(plus[:,1],19.) and np.allclose(minus[:,1],1.)
    assert source==[[[0.,10.],[100.,10.]]]


def test_resampling_control_preserves_gaps_and_original_candidate_points():
    result={'diagnostics':{'records':[{'segment_index':i,'input_xy':p} for i,ps in
             enumerate([[[0,0],[1,1]],[[4,4],[5,5]]]) for p in ps]}}
    assert _resampled_inputs(result)==[[[0,0],[1,1]],[[4,4],[5,5]]]
