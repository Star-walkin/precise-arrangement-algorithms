"""Pre-training numerical contracts; real arrays exclusively frozen training10."""
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_heatmap import (HERE,ROOT,DEFAULTS,load,dump,sha,seed_all,cfg_for,sample,
                           shared_hashes,legacy_checks,read_training)
from paired_train import pair_tensor_sample
from train import load_model
from network import load_network
from preprocessing import weighted_lcn,prepare_input


def main():
    tick=time.perf_counter();torch.set_num_threads(DEFAULTS['threads']);seed_all(DEFAULTS['seed'])
    result={'legacy_loss_and_geometry':legacy_checks(DEFAULTS)}
    cfg=cfg_for('D');h,w=141,219
    yy,xx=np.mgrid[:h,:w]
    marker=np.zeros((h,w),np.float32);marker[45:60,55:90]=1
    a={'image':np.clip(.3+.15*np.sin(xx/4)+.05*np.cos(yy/7),0,1).astype(np.float32),
       'seed':marker,'target':marker,'valid':np.ones((h,w),bool),'roi':np.ones((h,w),bool),
       'stage':np.ones((h,w),np.uint8)*2,'center_target':marker.copy(),
       'center_valid':np.ones((h,w),bool),'center_stage':np.ones((h,w),np.uint8)*2,
       'neighbor_mask':marker.astype(bool),'background_mask':~marker.astype(bool)}
    for j in range(8):
        torch.manual_seed(j);c=sample(a,cfg_for('C'),True,random.Random(j))
        torch.manual_seed(j);d=sample(a,cfg_for('D'),True,random.Random(j))
        torch.manual_seed(j);old=pair_tensor_sample(a,DEFAULTS,True,random.Random(j))
        assert shared_hashes(c)==shared_hashes(d)
        assert torch.equal(c['x'][:,:2],old['x'])
        assert torch.equal(c['center_target'],old['center_target'])
        assert torch.equal(c['center_valid'],old['center_valid'])
        assert torch.equal(c['neighbor_mask'],old['neighbor_mask'])
        assert torch.equal(c['background_mask'],old['background_mask'])
        assert torch.count_nonzero(c['x'][:,2])==0
        assert not torch.any(d['x'][:,2:3][~d['roi']])
    result['eight_paired_augmentations_shared_input_target_masks_exact']=True
    result['eight_first_two_channels_exact_legacy']=True
    result['lcn_does_not_draw_rng_or_change_sampling']=True
    mask=torch.ones(h,w,dtype=torch.bool);mask[:10]=False;mask[:,200:]=False
    const=torch.full((h,w),.4);z,meta=weighted_lcn(const,mask,cfg,True)
    assert float(z[mask].abs().max())<1e-4
    corrupted=const.clone();corrupted[~mask]=999
    z2,_=weighted_lcn(corrupted,mask,cfg)
    assert torch.equal(z,z2)
    realzero=const.clone();realzero[40:70,50:90]=0
    z0,m0=weighted_lcn(realzero,mask,cfg,True)
    assert torch.all(m0['arrays']['W'][40:70,50:90]>.8)
    assert z0[55,70]<0
    assert float(z0.min())>=-3 and float(z0.max())<=3
    assert torch.isfinite(z0).all()
    result.update({'constant_image_no_edge_halo':True,'outside_valid_domain_cannot_change_lcn':True,
                   'true_zero_gray_kept_in_local_statistics':True,'fixed_clip_float_and_finite':True,
                   'constant_max_abs_z':float(z[mask].abs().max())})
    # Synthetic separable-kernel receipt supports independent OpenCV comparison.
    with open(HERE/'lcn_reference_fixture.npz','wb') as f:
        np.savez_compressed(f,gray=realzero.numpy(),mask=mask.numpy(),z=z0.numpy(),
                            W=m0['arrays']['W'].numpy(),mean=m0['arrays']['local_mean'].numpy(),std=m0['arrays']['local_std'].numpy())
    protocol=load(HERE/'protocol.json');path=protocol['source_B_model']['path']
    oldnet=load_model(path);net=load_network(path)
    items,arrays,_=read_training(protocol['legacy_index']['path'])
    a=arrays[items[0]['id']]
    c=sample(a,cfg_for('C'));d=sample(a,cfg_for('D'));legacy_b=pair_tensor_sample(a,DEFAULTS)
    assert torch.equal(c['x'][:,:2],legacy_b['x']) and shared_hashes(c)==shared_hashes(d)
    with torch.inference_mode():
        yp=oldnet(legacy_b['x']);yc=net(c['x']);yd=net(d['x'])
    assert torch.equal(yp,yc) and torch.equal(yp,yd),'initial function must be bitwise B'
    result['full_training_ROI_initial_C_D_equal_B_bitwise']=True
    result['initial_max_absolute_difference']=float((yp-yd).abs().max())
    # New auxiliary path can learn; seed remains channel1 and affects distant output.
    torch.manual_seed(5);x=torch.randn(1,3,257,289)*.1;x[:,1]=0;x[:,1,15:25,15:25]=1;x.requires_grad_(True)
    net.zero_grad(set_to_none=True);pred=net(x);pred[0,0,200:220,220:240].mean().backward()
    seed_grad=float(x.grad[0,1,15:25,15:25].abs().sum())
    aux_grad=float(net.enc1.layers[0].auxiliary.weight.grad.abs().sum())
    assert seed_grad>0 and aux_grad>0
    result.update({'seed_index1_far_gradient':seed_grad,'zero_initialized_auxiliary_learnable_gradient':aux_grad,
                   'expanded_parameters':sum(p.numel() for p in net.parameters()),
                   'old_parameters':sum(p.numel() for p in oldnet.parameters())})
    # API is exactly the non-augmented training pipeline, including full ROI/padding.
    public,pm=prepare_input(a['image'],a['seed'],cfg_for('D'),a['roi'])
    assert torch.equal(public,d['x'])
    result['inference_prepare_input_equals_training_sample']=True
    result['source_sha256']={p:sha(HERE/p) for p in ['network.py','preprocessing.py','train_heatmap.py']}
    result.update({'status':'passed','exit_code':0,'elapsed_seconds':time.perf_counter()-tick,
                   'command':sys.argv,'real_image_read_scope':'only fixed training10 caches',
                   'reference_model_sha256':sha(path),'scope':'implementation contracts, not recognition accuracy'})
    dump(HERE/'numerical_checks.json',result);dump(HERE/'config.json',DEFAULTS)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
