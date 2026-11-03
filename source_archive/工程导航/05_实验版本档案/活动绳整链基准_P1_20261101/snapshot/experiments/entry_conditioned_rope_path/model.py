"""Small entry-conditioned U-Net; pure PyTorch, no acquisition/device imports.

The seed only marks the initial free entry. Its appearance descriptor and
coordinates are broadcast to the bottleneck so distant pixels receive the
entry condition even outside the ordinary convolutional receptive field.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),
            nn.GroupNorm(math.gcd(4,cout),cout),nn.SiLU(inplace=True),
            nn.Conv2d(cout,cout,3,padding=1,bias=False),
            nn.GroupNorm(math.gcd(4,cout),cout),nn.SiLU(inplace=True))
    def forward(self,x):return self.layers(x)


class EntryUNet(nn.Module):
    def __init__(self,base=8):
        super().__init__();self.base=base
        self.enc1=ConvBlock(2,base);self.enc2=ConvBlock(base,base*2)
        self.enc3=ConvBlock(base*2,base*4);self.bottleneck=ConvBlock(base*4,base*8)
        # Learned entry appearance + x/y/second moments/seed-presence descriptor.
        self.condition=nn.Sequential(nn.Linear(base+6,base*8),nn.SiLU(),nn.Linear(base*8,base*8))
        self.condition_fuse=nn.Conv2d(base*16+2,base*8,1)
        self.dec3=ConvBlock(base*12,base*4);self.dec2=ConvBlock(base*6,base*2)
        self.dec1=ConvBlock(base*3,base);self.output=nn.Conv2d(base,1,1)

    @staticmethod
    def grids(x):
        h,w=x.shape[-2:]
        xx=torch.linspace(-1,1,w,device=x.device,dtype=x.dtype)[None,None,None,:].expand(1,1,h,w)
        yy=torch.linspace(-1,1,h,device=x.device,dtype=x.dtype)[None,None,:,None].expand(1,1,h,w)
        return xx,yy

    def forward(self,x):
        assert x.ndim==4 and x.shape[1]==2
        seed=x[:,1:2].clamp(0,1)
        e1=self.enc1(x);e2=self.enc2(F.max_pool2d(e1,2))
        e3=self.enc3(F.max_pool2d(e2,2));z=self.bottleneck(F.max_pool2d(e3,2))
        xx,yy=self.grids(seed);mass=seed.sum((2,3),keepdim=True);den=mass.clamp_min(1)
        cx=(seed*xx).sum((2,3),keepdim=True)/den;cy=(seed*yy).sum((2,3),keepdim=True)/den
        dx,dy=xx-cx,yy-cy
        moments=[cx,cy,(seed*dx.square()).sum((2,3),keepdim=True)/den,
            (seed*dy.square()).sum((2,3),keepdim=True)/den,
            (seed*dx*dy).sum((2,3),keepdim=True)/den,
            mass/(mass+1)]
        appearance=(e1*seed).sum((2,3),keepdim=True)/den
        descriptor=torch.cat([appearance,*moments],dim=1).flatten(1)
        cond=self.condition(descriptor)[:,:,None,None].expand(-1,-1,*z.shape[-2:])
        relative=F.interpolate(torch.cat([dx.expand(x.shape[0],-1,-1,-1),dy.expand(x.shape[0],-1,-1,-1)],1),
            size=z.shape[-2:],mode='bilinear',align_corners=False)
        z=self.condition_fuse(torch.cat([z,cond,relative],1))
        z=self.dec3(torch.cat([F.interpolate(z,size=e3.shape[-2:],mode='bilinear',align_corners=False),e3],1))
        z=self.dec2(torch.cat([F.interpolate(z,size=e2.shape[-2:],mode='bilinear',align_corners=False),e2],1))
        z=self.dec1(torch.cat([F.interpolate(z,size=e1.shape[-2:],mode='bilinear',align_corners=False),e1],1))
        return torch.sigmoid(self.output(z))


def balanced_mse(pred,target,valid,threshold=.5):
    valid=valid.bool();hi=valid&(target>=threshold);lo=valid&(target<threshold)
    err=(pred-target).square();nh,nl=hi.sum(),lo.sum()
    high=(err*hi).sum()/nh.clamp_min(1);low=(err*lo).sum()/nl.clamp_min(1)
    present=(nh>0).to(err.dtype)+(nl>0).to(err.dtype)
    return (high+low)/present.clamp_min(1),{'high_mse':float(high.detach()),'low_mse':float(low.detach()),
        'high_pixels':int(nh),'low_pixels':int(nl)}

def stage_balanced_mse(pred,target,valid,stage,threshold=.5):
    """One weighted MSE: equal phases, each with equal high/low response mass.

    A phase enters the training mean only when it contains valid positive
    target pixels. Empty/unknown phases never create a fabricated target.
    """
    overall,detail=balanced_mse(pred,target,valid,threshold)
    active=[];rows={}
    for s in range(1,5):
        vv=valid.bool()&(stage==s)
        term,dd=balanced_mse(pred,target,vv,threshold)
        used=bool(dd['high_pixels']>0)
        if used:active.append(term)
        rows[str(s)]={**dd,'used_in_loss':used,'balanced_mse':float(term.detach()) if vv.any() else None}
    loss=torch.stack(active).mean() if active else pred.sum()*0
    return loss,{**detail,'overall_balanced_mse':float(overall.detach()),'active_positive_stages':len(active),'stages':rows}


def checks(base=8):
    torch.manual_seed(20261029)
    net=EntryUNet(base);x=torch.randn(1,2,257,289)*.15
    x[:,1]=0;x[:,1,15:25,15:25]=1;x.requires_grad_(True)
    y=net(x);assert y.shape==(1,1,257,289) and torch.isfinite(y).all()
    far=y[0,0,200:220,220:240].mean();far.backward()
    grad=float(x.grad[0,1,15:25,15:25].abs().sum());assert grad>1e-10
    with torch.no_grad():
        zero=x.detach().clone();zero[:,1]=0
        moved=x.detach().clone();moved[:,1]=torch.roll(moved[:,1],100,-1)
        yz,ym=net(zero),net(moved)
        zero_delta=float((y-yz).abs().mean());moved_delta=float((y-ym).abs().mean())
        far_moved_delta=float((y-ym)[0,0,200:220,220:240].abs().mean())
    assert far_moved_delta>1e-8
    p=torch.tensor([[[[.1,.4,.8,.9]]]],requires_grad=True)
    t=torch.tensor([[[[0.,1.,0.,1.]]]]);v=torch.tensor([[[[1,1,0,0]]]],dtype=torch.bool)
    loss,_=balanced_mse(p,t,v);loss.backward();assert torch.equal(p.grad[...,2:],torch.zeros_like(p.grad[...,2:]))
    other=t.clone();other[...,2:]=99;again,_=balanced_mse(p.detach(),other,v)
    assert torch.equal(loss.detach(),again)
    sp=torch.tensor([[[[.2,.4,.8,.9,.7]]]],requires_grad=True)
    st=torch.tensor([[[[1.,0.,1.,0.,1.]]]])
    sv=torch.tensor([[[[1,1,1,1,0]]]],dtype=torch.bool)
    ss=torch.tensor([[[[1,1,3,3,2]]]],dtype=torch.uint8)
    sl,sd=stage_balanced_mse(sp,st,sv,ss);sl.backward()
    assert abs(float(sl.detach())-.4125)<1e-6 and sp.grad[0,0,0,4]==0
    # 100 copies of free pixels must not outweigh two attached pixels.
    longp=torch.cat([sp.detach()[...,:2].repeat(1,1,1,100),sp.detach()[...,2:4]],-1)
    longt=torch.cat([st[...,:2].repeat(1,1,1,100),st[...,2:4]],-1)
    longs=torch.cat([ss[...,:2].repeat(1,1,1,100),ss[...,2:4]],-1)
    same,_=stage_balanced_mse(longp,longt,torch.ones_like(longp,dtype=torch.bool),longs)
    assert abs(float(same)-float(sl.detach()))<1e-6
    changed=st.clone();changed[...,4]=99
    empty,_=stage_balanced_mse(sp.detach(),changed,sv,ss);assert torch.equal(empty,sl.detach())
    return {'odd_output_shape':list(y.shape),'parameters':sum(p.numel() for p in net.parameters()),
      'far_output_seed_gradient_nonzero':True,'far_output_seed_gradient_abs_sum':grad,
      'untrained_zero_seed_mean_abs_change':zero_delta,'untrained_moved_seed_mean_abs_change':moved_delta,
      'untrained_far_moved_seed_mean_abs_change':far_moved_delta,'ignore_gradient_zero':True,
      'ignore_target_invariance':True,'stage_equal_weight_independent_of_pixel_count':True,
      'empty_stage_and_ignored_target_do_not_affect_loss':True,'stage_loss_ignore_gradient_zero':True,
      'scope':'architecture and numerical condition influence only, not rope identity accuracy'}
