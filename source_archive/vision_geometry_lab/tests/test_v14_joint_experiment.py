import numpy as np
from vision_geometry_lab.v14_joint_experiment import reference_support_pass,seed_downstream_targets,location
from pathlib import Path


def test_reference_support_gate_preserves_full_denominator():
    def row(fraction):
        return {'boundary_reference_support':{'edges':{
            k:{'scorable_fraction_of_trusted':fraction} for k in ('edge_flange','edge_surface')}}}
    assert reference_support_pass([row(.95)],.9)
    assert not reference_support_pass([row(.95),row(.89)],.9)
    assert not reference_support_pass([row(None)],.9)
    assert not reference_support_pass([],.9)
    assert not reference_support_pass([{'boundary_reference_support':{'edges':{}}}],.9)


def test_downstream_masks_score_only_and_do_not_mutate_targets():
    targets=dict(center_valid=np.ones(4,bool),width_valid=np.ones(4,bool),edge_valid=np.ones((2,4),bool),
                 edge_rows=np.array([[1]*4,[4]*4],float))
    geometry=dict(u_first_widths=-1,step_widths=.5,shape_hw=[8,4])
    selected=seed_downstream_targets(targets,geometry)
    assert selected['center_valid'].tolist()==[False,False,True,True]
    assert not selected['edge_valid'][:,:2].any()
    assert targets['center_valid'].all() and targets['edge_valid'].all()
    assert np.array_equal(selected['edge_rows'],targets['edge_rows'])


def test_development_is_not_warm_started_single_location():
    base=Path('trial')
    assert location(base,'single','joint')!=location(base,'train','joint')
    assert location(base,'holdout','joint')==location(base,'train','joint')
