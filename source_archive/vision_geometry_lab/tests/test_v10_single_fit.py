"""Loss contract checks run via the dedicated torch environment, no field images."""
from pathlib import Path
import os
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = ROOT / 'vision_training/.venv/Scripts/python.exe'


def run_torch(code):
    result = subprocess.run([str(TRAIN_PY), '-X', 'utf8', '-c', code], cwd=ROOT,
        capture_output=True, text=True, timeout=60,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise AssertionError(result.stdout + '\n' + result.stderr)


class SingleFitV10Tests(unittest.TestCase):
    def test_unknown_target_is_not_background_and_has_zero_gradient(self):
        run_torch('''
import torch
from vision_geometry_lab.v10_single_fit import paired_loss
for arm in ('masked_bce','balanced_bce_dice'):
    outputs={k:torch.zeros(1,1,2,3,requires_grad=True) for k in ('body_logits','center_logits')}
    target=torch.tensor([[[[1.,0.,float('nan')],[.5,0.,float('nan')]]]])
    mask=torch.tensor([[[[1.,1.,0.],[1.,1.,0.]]]])
    batch={k:v for h in ('body','center') for k,v in ((h+'_target',target),(h+'_valid_mask',mask))}
    out=paired_loss(outputs,batch,arm);out['loss'].backward()
    assert torch.isfinite(out['loss'])
    for value in outputs.values():
        assert (value.grad[mask==0]==0).all()
        assert torch.isfinite(value.grad).all()
''')

    def test_zero_supervision_has_exact_zero_loss_and_grad(self):
        run_torch('''
import torch
from vision_geometry_lab.v10_single_fit import paired_loss
for arm in ('masked_bce','balanced_bce_dice'):
    outputs={k:torch.randn(2,1,3,4,requires_grad=True) for k in ('body_logits','center_logits')}
    batch={k:v for h in ('body','center') for k,v in ((h+'_target',torch.full((2,1,3,4),float('nan'))),(h+'_valid_mask',torch.zeros(2,1,3,4)))}
    loss=paired_loss(outputs,batch,arm)['loss'];loss.backward()
    assert loss.item()==0
    assert all((v.grad==0).all() for v in outputs.values())
''')

    def test_balancing_resists_negative_pixel_duplication(self):
        run_torch('''
import torch
from vision_geometry_lab.v10_single_fit import _balanced_head
torch.set_num_threads(1)
for neg_count in (1,20,100):
    logits=torch.zeros(1,1,1,neg_count+1,requires_grad=True)
    target=torch.zeros_like(logits);target[...,0]=1
    loss,bce,dice=_balanced_head(logits,target,torch.ones_like(logits))
    assert abs(bce.item()-0.69314718)<1e-6
    bce.backward()
    assert abs(logits.grad[...,0].item()+.25)<1e-6
    assert abs(logits.grad[...,1:].sum().item()-.25)<1e-6
''')

    def test_known_negative_head_and_bad_arm(self):
        run_torch('''
import torch
from vision_geometry_lab.v10_single_fit import _balanced_head,paired_loss
z=torch.zeros(1,1,2,2,requires_grad=True)
loss,bce,dice=_balanced_head(z,torch.zeros_like(z),torch.ones_like(z))
assert dice.item()==0 and loss.item()==bce.item()
loss.backward();assert (z.grad>0).all()
try: paired_loss({}, {}, 'invented')
except ValueError: pass
else: raise AssertionError('invalid arm accepted')
''')

    def test_only_the_fixed_train_record_is_selected(self):
        run_torch('''
from vision_geometry_lab.v10_single_fit import select_train_sample,SAMPLE_ID
good={'sample_id':SAMPLE_ID,'split':'train'}
bad={'sample_id':'not_target','split':'holdout','image_path':'does_not_exist.png'}
assert select_train_sample({'samples':[bad,good]}) is good
for entries in ([],[dict(good,split='holdout')],[good,good]):
    try: select_train_sample({'samples':entries})
    except ValueError: pass
    else: raise AssertionError('bad sample set accepted')
''')


if __name__ == '__main__':
    unittest.main()
