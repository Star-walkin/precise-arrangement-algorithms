"""Finite observed-domain direction contracts; not contact/identity tests."""
import copy
import numpy as np
from common import HERE,dump,sha
from run_h5 import task_direction_feasibility

def entry(xs,group='same_A7',sign=1):
    p=np.stack([xs,np.full(len(xs),50.)],axis=1)
    return dict(seed=dict(a7_candidate_id=group,xy=[[100,100],[100+sign*12,100]]),
        free=dict(points_xy=p.tolist(),point_status=['entry_condition']*3+['image_supported']*(len(xs)-3),
        protect_support=[True]*len(xs),supported=[True]*len(xs)))

mask=np.zeros((100,300),np.uint8);mask[:,200:]=1
good=entry(np.arange(50.,281.,3.));out=entry(np.arange(180.,9.,-3.),sign=-1)
pair=[copy.deepcopy(good),copy.deepcopy(out)]
task_direction_feasibility(pair,mask)
assert pair[0]['task_direction']['selection_allowed'] and not pair[1]['task_direction']['selection_allowed']
assert not pair[0]['task_direction']['target_identity_confirmed']
short=entry(np.arange(50.,199.,3.))
pair=[short,copy.deepcopy(out)];task_direction_feasibility(pair,mask)
assert all(e['task_direction']['selection_allowed'] for e in pair)
other=copy.deepcopy(out);other['seed']['a7_candidate_id']='unrelated'
pair=[copy.deepcopy(good),other];task_direction_feasibility(pair,mask)
assert all(e['task_direction']['selection_allowed'] for e in pair)
single=[copy.deepcopy(out)];task_direction_feasibility(single,mask)
assert single[0]['task_direction']['selection_allowed']
weak=copy.deepcopy(good);weak['free']['supported']=[False]*len(weak['free']['supported'])
pair=[weak,copy.deepcopy(out)];task_direction_feasibility(pair,mask)
assert all(e['task_direction']['selection_allowed'] for e in pair)
dump(HERE/'direction_contract_check.json',dict(status='passed',checks=['observed_paired_opposites',
    'no_infinite_extrapolation','different_A7_not_filtered','single_entry_unchanged',
    'B_without_original_corroboration_insufficient'],source_sha256=sha(HERE/'run_h5.py'),identity_verified=False))
print('5 finite-direction contracts passed')
