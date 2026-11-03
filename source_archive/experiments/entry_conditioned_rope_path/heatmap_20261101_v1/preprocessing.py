"""Full-search-domain weighted LCN; float-only model input, no display conversion.

Photometry operates only on nonnegative gray before this module. M is the full
image search domain (possibly warped), never SAM2 or sparse supervision. Pixels
of genuine gray zero inside M participate in the local mean and variance.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

DEFAULTS = {'variant': 'D', 'normalization_std_floor': .08, 'pad_multiple': 32,
            'lcn_sigma_model_px': 16., 'lcn_sigma_original_px': 32.,
            'lcn_std_floor': 2.5/255., 'lcn_truncate': 4., 'lcn_clip': 3.,
            'lcn_epsilon': 1e-6, 'lcn_border': 'constant-zero masked normalization'}


def weighted_lcn(gray, valid, cfg, diagnostics=False):
    """Return z (H,W) and small statistics; optional full intermediate tensors."""
    assert gray.ndim == 2 and valid.shape == gray.shape and valid.any()
    sigma=float(cfg['lcn_sigma_model_px']);radius=int(math.ceil(sigma*cfg['lcn_truncate']))
    offsets=torch.arange(-radius,radius+1,dtype=gray.dtype,device=gray.device)
    kernel=torch.exp(-.5*(offsets/sigma).square());kernel=kernel/kernel.sum()
    mask=valid.to(gray.dtype)
    fields=torch.stack([mask,gray*mask,gray.square()*mask])[:,None]
    blur=F.conv2d(fields,kernel[None,None,None,:],padding=(0,radius))
    blur=F.conv2d(blur,kernel[None,None,:,None],padding=(radius,0))[:,0]
    weight=blur[0];den=weight.clamp_min(cfg['lcn_epsilon'])
    mean=blur[1]/den;variance=(blur[2]/den-mean.square()).clamp_min(0)
    std=variance.sqrt();floor=float(cfg['lcn_std_floor'])
    raw_z=(gray-mean)/std.clamp_min(floor)
    z=raw_z.clamp(-cfg['lcn_clip'],cfg['lcn_clip']);z[~valid]=0
    meta={'sigma_model_px':sigma,'sigma_original_px':cfg['lcn_sigma_original_px'],
          'truncate':cfg['lcn_truncate'],'kernel_size':2*radius+1,
          'border':cfg['lcn_border'],'std_floor':floor,
          'std_floor_fraction':float((std[valid]<floor).float().mean()),
          'clip_fraction':float((raw_z[valid].abs()>cfg['lcn_clip']).float().mean()),
          'W_min_in_domain':float(weight[valid].min()),
          'mean':float(z[valid].mean()),'std':float(z[valid].std(unbiased=False)),
          'minimum':float(z[valid].min()),'maximum':float(z[valid].max())}
    if diagnostics:meta['arrays']={'W':weight,'local_mean':mean,'local_std':std,'unclipped_z':raw_z}
    return z,meta


def input_from_tensors(gray, seed, valid, cfg=None):
    cfg={**DEFAULTS,**(cfg or {})}
    assert gray.ndim==2 and gray.shape==seed.shape==valid.shape
    assert torch.isfinite(gray).all() and 0<=gray.min() and gray.max()<=1
    assert torch.isfinite(seed).all() and valid.any()
    mean=gray[valid].mean();std=gray[valid].std(unbiased=False).clamp_min(cfg['normalization_std_floor'])
    raw=(gray-mean)/std;raw[~valid]=0
    if cfg['variant']=='D':aux,lcnmeta=weighted_lcn(gray,valid,cfg)
    else:
        assert cfg['variant']=='C'
        aux=torch.zeros_like(gray);lcnmeta={'enabled':False,'auxiliary_exact_zero':True}
    h,w=gray.shape;m=cfg['pad_multiple'];pad=(0,(-w)%m,0,(-h)%m)
    x=F.pad(torch.stack([raw,seed.clamp(0,1),aux]),pad)[None]
    return x,{'unpadded_shape':[h,w],'mean':float(mean),'std':float(std),
              'raw_gray_min':float(gray[valid].min()),'raw_gray_max':float(gray[valid].max()),
              'valid_pixels':int(valid.sum()),'true_zero_valid_pixels':int(((gray==0)&valid).sum()),
              'lcn':lcnmeta,'channel_order':['raw_z','entry_seed','lcn_aux'],
              'padding':list(pad),'neutral_fill':0.0,'variant':cfg['variant']}


def prepare_input(gray, seed, cfg=None, roi=None):
    """Public inference API: numpy HxW float gray[0,1], seed -> tensor/meta.

    The caller crops/downsamples once in the established original-coordinate
    mapping. LCN is calculated on that full search ROI before neutral padding.
    cfg can be checkpoint['config']; roi defaults to all input pixels.
    """
    gray=np.asarray(gray,dtype=np.float32);seed=np.asarray(seed,dtype=np.float32)
    valid=np.ones(gray.shape,bool) if roi is None else np.asarray(roi,bool)
    return input_from_tensors(torch.from_numpy(gray.copy()),torch.from_numpy(seed.copy()),
                              torch.from_numpy(valid.copy()),cfg)
