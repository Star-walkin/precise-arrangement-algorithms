"""Synthetic forward/backward contracts only; no training or field data."""
import unittest

from vision_geometry_lab.tests.test_v11_path_model import run_torch


class JointBoundaryModelV14Tests(unittest.TestCase):
    def test_shared_initialization_and_primary_outputs_are_bit_identical(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import build_model as old_build
from vision_geometry_lab.v14_joint_boundary_model import build_model
torch.set_num_threads(2)
torch.manual_seed(71); before=torch.random.get_rng_state().clone()
new=build_model(20260929,2).eval()
assert torch.equal(before,torch.random.get_rng_state())
old=old_build(20260929,2).eval(); again=build_model(20260929,2).eval()
assert all(torch.equal(value,new.state_dict()[key]) for key,value in old.state_dict().items())
assert all(torch.equal(value,again.state_dict()[key]) for key,value in new.state_dict().items())
x=torch.rand(2,8,33,67);x[:,7]=1;x[:,7,:,6:9]=0
with torch.inference_mode(): a=old(x);b=new(x)
assert torch.equal(a['center_logits'],b['center_logits'])
assert torch.equal(a['slope'],b['slope'])
assert b['edge_logits'].shape==(2,2,33,67)
assert set(b)=={'center_logits','slope','edge_logits'}
''')

    def test_full_strip_and_inference_uses_image_mask_for_both_sides(self):
        run_torch('''
import json
import torch
from vision_geometry_lab.v14_joint_boundary_model import build_model,image_masked_edge_probabilities,architecture_metadata
torch.set_num_threads(2)
x=torch.rand(1,8,256,512);x[:,7]=1;x[:,7,:,30:35]=0
with torch.inference_mode(): out=build_model(7,2).eval()(x)
assert out['edge_logits'].shape==(1,2,256,512)
p=image_masked_edge_probabilities(out,x[:,7])
assert (p[:,:,:,30:35]==0).all()
assert torch.allclose(p.sum(2)[:,:,:30],torch.ones(1,2,30),atol=1e-6)
try: image_masked_edge_probabilities(out,{'row_valid':x[:,7]})
except ValueError: pass
else: raise AssertionError('annotation dictionary accepted for inference')
metadata=architecture_metadata(2);json.dumps(metadata,allow_nan=False)
assert not metadata['visibility_head'] and metadata['no_physical_event_output']
''')

    def test_load_baseline_keeps_new_head_and_rejects_incomplete_mapping(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import build_model as old_build
from vision_geometry_lab.v14_joint_boundary_model import build_model,load_shared_baseline
torch.set_num_threads(2)
old=old_build(11,2);new=build_model(12,2)
head={k:v.clone() for k,v in new.edge_head.state_dict().items()}
load_shared_baseline(new,old.state_dict())
assert all(torch.equal(value,new.state_dict()[key]) for key,value in old.state_dict().items())
assert all(torch.equal(value,new.edge_head.state_dict()[key]) for key,value in head.items())
bad=dict(old.state_dict());bad.pop(next(iter(bad)))
try: load_shared_baseline(new,bad)
except ValueError: pass
else: raise AssertionError('incomplete baseline accepted')
''')

    def test_unknown_partial_edges_nan_safe_and_missing_side_has_zero_gradient(self):
        run_torch('''
import torch
from vision_geometry_lab.v14_joint_boundary_model import joint_path_loss
center=torch.zeros(1,4,3,requires_grad=True);slope=torch.zeros(1,3,requires_grad=True)
rows=torch.zeros(1,2,4,3);rows[0,0,:2,:2]=1
columns=torch.zeros(1,2,3);columns[0,0,:2]=1
valid=rows.bool() & columns[:,:,None,:].bool()
raw=torch.full((1,2,4,3),float('nan'));raw[valid]=0
edges=raw.requires_grad_();targets=torch.full_like(edges,float('nan'));targets[valid]=1
batch={'target_distribution':torch.full_like(center,float('nan')),'row_valid':torch.zeros_like(center),
 'column_valid':torch.zeros(1,3),'slope_target':torch.full_like(slope,float('nan')),'slope_valid':torch.zeros_like(slope),
 'edge_target_distribution':targets,'edge_row_valid':rows,'edge_column_valid':columns}
r=joint_path_loss({'center_logits':center,'slope':slope,'edge_logits':edges},batch)
assert r['valid_edge_columns'].item()==2 and r['valid_edge_columns_by_side'].tolist()==[2,0]
assert torch.isfinite(r['loss']);r['loss'].backward()
assert torch.isfinite(edges.grad).all() and (edges.grad[~valid]==0).all()
assert (center.grad==0).all() and (slope.grad==0).all()
''')

    def test_edge_ce_averages_all_side_columns_not_side_means(self):
        run_torch('''
import math
import torch
from vision_geometry_lab.v14_joint_boundary_model import joint_path_loss
center=torch.zeros(1,4,3);slope=torch.zeros(1,3);edges=torch.zeros(1,2,4,3,requires_grad=True)
rows=torch.zeros_like(edges);rows[0,0,:,0]=1;rows[0,1,:2,:]=1
columns=torch.tensor([[[1,0,0],[1,1,1]]])
batch={'target_distribution':torch.ones_like(center),'row_valid':torch.ones_like(center),'column_valid':torch.ones(1,3),
 'slope_target':torch.zeros_like(slope),'slope_valid':torch.ones_like(slope),
 'edge_target_distribution':torch.ones_like(edges)*7,'edge_row_valid':rows,'edge_column_valid':columns}
r=joint_path_loss({'center_logits':center,'slope':slope,'edge_logits':edges},batch)
expected=(math.log(4)+3*math.log(2))/4
assert abs(r['edge_ce_loss'].item()-expected)<1e-6
assert abs(r['loss'].item()-(math.log(4)+.5*expected))<1e-6
assert r['valid_edge_columns'].item()==4
''')

    def test_all_unknown_nan_batch_is_zero_with_zero_gradients(self):
        run_torch('''
import torch
from vision_geometry_lab.v14_joint_boundary_model import joint_path_loss
c=torch.full((2,4,3),float('nan'),requires_grad=True);s=torch.full((2,3),float('nan'),requires_grad=True)
e=torch.full((2,2,4,3),float('nan'),requires_grad=True)
b={'target_distribution':torch.full_like(c,float('nan')),'row_valid':torch.zeros_like(c),'column_valid':torch.zeros(2,3),
 'slope_target':torch.full_like(s,float('nan')),'slope_valid':torch.zeros_like(s),
 'edge_target_distribution':torch.full_like(e,float('nan')),'edge_row_valid':torch.zeros_like(e),'edge_column_valid':torch.zeros(2,2,3)}
r=joint_path_loss({'center_logits':c,'slope':s,'edge_logits':e},b)
assert r['loss'].item()==0 and r['edge_ce_loss'].item()==0
r['loss'].backward()
assert (c.grad==0).all() and (s.grad==0).all() and (e.grad==0).all()
''')

    def test_center_only_synthetic_masks_equal_original_loss_and_gradients_exactly(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import path_loss
from vision_geometry_lab.v14_joint_boundary_model import joint_path_loss
torch.manual_seed(23)
c=torch.randn(1,4,5,requires_grad=True);s=torch.randn(1,5,requires_grad=True)
c2=c.detach().clone().requires_grad_();s2=s.detach().clone().requires_grad_()
e=torch.full((1,2,4,5),float('nan'),requires_grad=True)
b={'target_distribution':torch.rand_like(c),'row_valid':torch.ones_like(c),'column_valid':torch.ones(1,5),
 'slope_target':torch.zeros_like(s),'slope_valid':torch.ones_like(s),
 'edge_target_distribution':torch.full_like(e,float('nan')),'edge_row_valid':torch.zeros_like(e),'edge_column_valid':torch.zeros(1,2,5)}
old=path_loss({'center_logits':c,'slope':s},b)
new=joint_path_loss({'center_logits':c2,'slope':s2,'edge_logits':e},b)
assert torch.equal(old['loss'],new['loss'])
old['loss'].backward();new['loss'].backward()
assert torch.equal(c.grad,c2.grad) and torch.equal(s.grad,s2.grad) and (e.grad==0).all()
''')

    def test_weight_zero_is_original_tensor_with_no_edge_target_dependency(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import path_loss
from vision_geometry_lab.v14_joint_boundary_model import joint_path_loss
c=torch.randn(1,4,5,requires_grad=True);s=torch.randn(1,5,requires_grad=True)
b={'target_distribution':torch.rand_like(c),'row_valid':torch.ones_like(c),'column_valid':torch.ones(1,5),
 'slope_target':torch.zeros_like(s),'slope_valid':torch.ones_like(s)}
out={'center_logits':c,'slope':s}
old=path_loss(out,b);new=joint_path_loss(out,b,edge_weight=0)
assert torch.equal(old['loss'],new['loss'])
assert torch.equal(torch.autograd.grad(old['loss'],c,retain_graph=True)[0],torch.autograd.grad(new['loss'],c)[0])
assert new['edge_ce_loss'].item()==0
''')

    def test_edge_loss_reaches_edge_head_and_shared_trunk_not_center_head(self):
        run_torch('''
import torch
from vision_geometry_lab.v14_joint_boundary_model import build_model,joint_path_loss
torch.set_num_threads(2)
x=torch.rand(1,8,32,64);x[:,7]=1
model=build_model(13,2);out=model(x)
out['center_logits'].retain_grad()
target=torch.zeros_like(out['edge_logits']);target[:,:,8,:]=1
b={'target_distribution':torch.full_like(out['center_logits'],float('nan')),'row_valid':torch.zeros_like(out['center_logits']),
 'column_valid':torch.zeros(1,64),'slope_target':torch.full_like(out['slope'],float('nan')),'slope_valid':torch.zeros_like(out['slope']),
 'edge_target_distribution':target,'edge_row_valid':torch.ones_like(target),'edge_column_valid':torch.ones(1,2,64)}
joint_path_loss(out,b)['loss'].backward()
assert (out['center_logits'].grad==0).all()
assert model.edge_head.weight.grad.abs().sum()>0
assert model.encoders[0].layers[0].weight.grad.abs().sum()>0
assert model.center_head.weight.grad.abs().sum()==0
''')

    def test_bad_edge_masks_targets_and_shape_rejected(self):
        run_torch('''
import copy
import torch
from vision_geometry_lab.v14_joint_boundary_model import joint_path_loss
c=torch.zeros(1,3,2);s=torch.zeros(1,2);e=torch.zeros(1,2,3,2)
good={'target_distribution':torch.ones_like(c),'row_valid':torch.ones_like(c),'column_valid':torch.ones(1,2),
 'slope_target':torch.zeros_like(s),'slope_valid':torch.ones_like(s),
 'edge_target_distribution':torch.ones_like(e),'edge_row_valid':torch.ones_like(e),'edge_column_valid':torch.ones(1,2,2)}
for kind in ('fraction','no_rows','zero_mass','nan_target','negative','bad_column_shape','nan_logit'):
 b=copy.deepcopy(good)
 logits=e.clone()
 if kind=='fraction': b['edge_row_valid'][0,0,0,0]=.2
 elif kind=='no_rows': b['edge_row_valid'][0,0,:,0]=0
 elif kind=='zero_mass': b['edge_target_distribution'][0,0,:,0]=0
 elif kind=='nan_target': b['edge_target_distribution'][0,0,0,0]=float('nan')
 elif kind=='negative': b['edge_target_distribution'][0,0,0,0]=-1
 elif kind=='bad_column_shape': b['edge_column_valid']=torch.ones(1,2)
 elif kind=='nan_logit': logits[0,0,0,0]=float('nan')
 try: joint_path_loss({'center_logits':c,'slope':s,'edge_logits':logits},b)
 except ValueError: pass
 else: raise AssertionError(kind+' accepted')
''')


if __name__ == '__main__':
    unittest.main()
