"""Synthetic-only V11 model/loss contracts; never open field data."""
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = ROOT / 'vision_training/.venv/Scripts/python.exe'


def run_torch(code, timeout=60):
    result = subprocess.run([str(TRAIN_PY), '-X', 'utf8', '-c', code], cwd=ROOT,
        capture_output=True, text=True, timeout=timeout,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise AssertionError(result.stdout + '\n' + result.stderr)
    return result.stdout


class SpatialPathModelV11Tests(unittest.TestCase):
    def test_initialization_reproducible_without_changing_caller_rng(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import build_model
torch.set_num_threads(2)
torch.manual_seed(71);before=torch.random.get_rng_state().clone()
a=build_model(20260928,8)
assert torch.equal(before,torch.random.get_rng_state())
b=build_model(20260928,8);c=build_model(9,8)
assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
assert any(not torch.equal(a.state_dict()[k],c.state_dict()[k]) for k in a.state_dict())
''')

    def test_full_strip_shape_bounded_slope_and_legal_columns(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import build_model,mask_normalized_probabilities
torch.set_num_threads(2)
model=build_model(20260928).eval()
inputs=torch.rand(1,8,256,512);inputs[:,7]=1
inputs[:,7,:,30:35]=0
with torch.inference_mode(): out=model(inputs)
assert set(out)=={'center_logits','slope'}
assert out['center_logits'].shape==(1,256,512)
assert out['slope'].shape==(1,512)
assert (out['slope'].abs()<=3.000001).all()
assert (out['slope'][:,30:35]==0).all()
p=mask_normalized_probabilities(out,inputs[:,7])
assert (p[:,:,30:35]==0).all()
assert torch.allclose(p.sum(1)[:,:30],torch.ones(1,30),atol=1e-6)
assert torch.isfinite(p).all()
''')

    def test_odd_spatial_shapes_preserve_aligned_output(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import build_model
torch.set_num_threads(2)
x=torch.rand(2,8,33,67);x[:,7]=1
out=build_model(8,2)(x)
assert out['center_logits'].shape==(2,33,67)
assert out['slope'].shape==(2,67)
''')

    def test_unknown_nan_logits_targets_rows_columns_have_zero_gradients(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import path_loss,mask_normalized_probabilities
torch.set_num_threads(2)
rows=torch.tensor([[[1.,1.,0.],[1.,0.,0.],[0.,0.,0.],[0.,0.,0.]]])
columns=torch.tensor([[1.,1.,0.]])
valid=rows.bool()&columns[:,None].bool()
raw=torch.zeros(1,4,3);raw[~valid]=float('nan')
logits=raw.requires_grad_();slope=torch.tensor([[0.,0.,float('nan')]],requires_grad=True)
target=torch.full((1,4,3),float('nan'));target[valid]=1.
batch={'target_distribution':target,'row_valid':rows,'column_valid':columns,
 'slope_target':torch.tensor([[.2,-.1,float('nan')]]),'slope_valid':columns}
loss=path_loss({'center_logits':logits,'slope':slope},batch)
assert torch.isfinite(loss['loss']);loss['loss'].backward()
assert (logits.grad[~valid]==0).all() and torch.isfinite(logits.grad).all()
assert slope.grad[0,2].item()==0 and torch.isfinite(slope.grad).all()
p=mask_normalized_probabilities({'center_logits':logits},batch)
assert (p[~valid]==0).all()
assert torch.allclose(p.sum(1),columns,atol=1e-6)
''')

    def test_fully_unknown_batch_is_exact_zero_not_nan(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import path_loss,mask_normalized_probabilities
logits=torch.full((2,5,7),float('nan'),requires_grad=True)
slope=torch.full((2,7),float('nan'),requires_grad=True)
batch={'target_distribution':torch.full_like(logits,float('nan')),'row_valid':torch.zeros_like(logits),
 'column_valid':torch.zeros(2,7),'slope_target':torch.full_like(slope,float('nan')),'slope_valid':torch.zeros_like(slope)}
result=path_loss({'center_logits':logits,'slope':slope},batch)
assert result['loss'].item()==0
result['loss'].backward()
assert (logits.grad==0).all() and (slope.grad==0).all()
assert (mask_normalized_probabilities(logits,batch)==0).all()
''')

    def test_ce_softmass_normalization_and_loss_weights(self):
        run_torch('''
import math
import torch
from vision_geometry_lab.v11_path_model import path_loss
logits=torch.zeros(1,2,3,requires_grad=True);slope=torch.zeros(1,3,requires_grad=True)
batch={'target_distribution':torch.tensor([[[2.,0.,9.],[0.,7.,6.]]]),
 'row_valid':torch.ones_like(logits),'column_valid':torch.tensor([[1.,1.,0.]]),
 'slope_target':torch.tensor([[1.,-1.,99.]]),'slope_valid':torch.tensor([[1.,1.,0.]])}
result=path_loss({'center_logits':logits,'slope':slope},batch)
assert abs(result['center_loss'].item()-math.log(2))<1e-6
assert abs(result['slope_loss'].item()-.5)<1e-6
assert abs(result['loss'].item()-(math.log(2)+.05))<1e-6
assert result['valid_columns'].item()==2
assert result['valid_slope_columns'].item()==2
batch['distribution']=batch.pop('target_distribution')
assert torch.equal(path_loss({'center_logits':logits,'slope':slope},batch)['loss'],result['loss'])
''')

    def test_slope_loss_does_not_backpropagate_into_center_logits(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import build_model,path_loss
torch.set_num_threads(2)
inputs=torch.rand(1,8,32,64);inputs[:,7]=1
out=build_model(23,2)(inputs);out['center_logits'].retain_grad()
batch={'target_distribution':torch.full_like(out['center_logits'],float('nan')),
 'row_valid':torch.zeros_like(out['center_logits']),'column_valid':torch.zeros(1,64),
 'slope_target':torch.ones(1,64),'slope_valid':torch.ones(1,64)}
loss=path_loss(out,batch);loss['loss'].backward()
assert (out['center_logits'].grad==0).all()
''')

    def test_invalid_supervised_targets_and_masks_rejected(self):
        run_torch('''
import copy
import torch
from vision_geometry_lab.v11_path_model import path_loss,build_model
logits=torch.zeros(1,3,2);slope=torch.zeros(1,2)
good={'target_distribution':torch.ones_like(logits),'row_valid':torch.ones_like(logits),
 'column_valid':torch.ones(1,2),'slope_target':torch.zeros(1,2),'slope_valid':torch.ones(1,2)}
for kind in ('mask_fraction','no_rows','zero_target','nan_target','negative_target','bad_slope'):
    batch=copy.deepcopy(good)
    if kind=='mask_fraction': batch['row_valid'][0,0,0]=.2
    elif kind=='no_rows': batch['row_valid'][:,:,0]=0
    elif kind=='zero_target': batch['target_distribution'][:,:,0]=0
    elif kind=='nan_target': batch['target_distribution'][0,0,0]=float('nan')
    elif kind=='negative_target': batch['target_distribution'][0,0,0]=-1.
    elif kind=='bad_slope': batch['slope_target'][0,0]=float('nan')
    try: path_loss({'center_logits':logits,'slope':slope},batch)
    except ValueError: pass
    else: raise AssertionError(kind+' accepted')
for shape in ((1,4,32,32),(1,8,8,8)):
    try: build_model(0,2)(torch.zeros(shape))
    except ValueError: pass
    else: raise AssertionError('bad input accepted')
''')

    def test_tiny_synthetic_curve_can_reduce_masked_path_loss(self):
        run_torch('''
import torch
from vision_geometry_lab.v11_path_model import build_model,path_loss
torch.set_num_threads(2)
g=torch.Generator().manual_seed(20260928)
H,W=32,64
x=torch.rand(1,8,H,W,generator=g)*.05;x[:,7]=1
yy=torch.arange(H)[:,None].float();xx=torch.arange(W)[None].float()
center=10.+.06*xx+.001*xx*xx
ridge=torch.exp(-.5*((yy-center)/1.1)**2)
x[0,:3]=ridge[None];x[0,3]=ridge;x[0,4,:,:8]=ridge[:,:8]
target=(ridge/ridge.sum(0,keepdim=True))[None]
batch={'target_distribution':target,'row_valid':torch.ones_like(target),'column_valid':torch.ones(1,W),
 'slope_target':(.06+.002*xx),'slope_valid':torch.ones(1,W)}
model=build_model(17,2);opt=torch.optim.Adam(model.parameters(),lr=.01)
before=path_loss(model(x),batch)['loss'].item()
for _ in range(35):
    opt.zero_grad();loss=path_loss(model(x),batch)['loss'];loss.backward();opt.step()
after=path_loss(model(x),batch)['loss'].item()
assert after < before*.8,(before,after)
''', timeout=90)


if __name__ == '__main__':
    unittest.main()
