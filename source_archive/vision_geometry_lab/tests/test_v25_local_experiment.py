import numpy as np
import pytest
from vision_geometry_lab.v25_local_experiment import commit_segments,training_plan,rollout

def test_commit_uses_prediction_arclength_not_endpoint_chord():
    output=commit_segments([[[0,0],[2,0],[2,6]]],1.)
    assert np.allclose(output,[[[0,0],[2,0],[2,2]]])

def test_plan_identical_exposures_and_no_held_out():
    index=dict(train_sample_ids=[str(i) for i in range(10)],train_keys={},entries={})
    for sid in index['train_sample_ids']:
        index['train_keys'][sid]=[sid+'a',sid+'b']
        for key in index['train_keys'][sid]:index['entries'][key]=dict(split='train')
    plan=training_plan(index)
    assert len(plan)==600
    assert all(sum(key.startswith(sid) for batch in plan for key in batch)==120 for sid in index['train_sample_ids'])
    index['entries']['0a']['split']='holdout'
    with pytest.raises(ValueError):training_plan(index)

def test_auto_stops_without_gt_and_does_not_bridge_missing():
    def predict(x,valid,g):
        return dict(segments=[],reason='missing',diagnostics={}),None,None,0.,0.
    out=rollout(np.zeros((400,400,3),np.uint8),[[50,200],[100,200]],10,predict,'high')
    assert out['segments']==[] and len(out['steps'])==1 and not out['GT_used']

def test_rollout_advances_only_own_prediction_and_keeps_unknown_identity():
    origins=[]
    def predict(x,valid,g):
        a=np.array(g['origin_xy']);origins.append(a)
        segment=np.stack([a,a+[20,0],a+[40,0]]).tolist()
        return dict(segments=[segment],reason='candidate',diagnostics={'stopped_on_unreliable_connection':False}),None,None,0.,0.
    out=rollout(np.zeros((400,400,3),np.uint8),[[50,200],[100,200]],10,predict,'high',maximum_steps=2)
    assert np.allclose(origins,[[100,200],[140,200]])
    assert len(out['segments'])==2 and not out['identity_confirmed']

def test_jump_to_neighbor_not_committed():
    def predict(x,valid,g):
        a=np.array(g['origin_xy'])+[0,10]
        return dict(segments=[np.stack([a,a+[40,0]]).tolist()],reason='candidate',diagnostics={}),None,None,0.,0.
    out=rollout(np.zeros((400,400,3),np.uint8),[[50,200],[100,200]],10,predict,'high')
    assert not out['segments'] and not out['steps'][0]['accepted']
