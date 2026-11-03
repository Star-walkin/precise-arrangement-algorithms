"""H7 additive LCN input adapter. Frozen two-channel U-Net is read, never edited.

The old first convolution still receives exactly the original two channels.
A separate bias-free, zero-initialized convolution reads auxiliary channel 2.
This preserves the original convolution's arithmetic and seed at index 1.
"""
from pathlib import Path
import importlib.util
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location('h7_frozen_legacy_model', ROOT/'model.py')
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)


class AdditiveInputConv(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.auxiliary = nn.Conv2d(1, original.out_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.auxiliary.weight)

    def forward(self, x):
        return self.original(x[:, :2]) + self.auxiliary(x[:, 2:3])


class HeatmapUNet(legacy.EntryUNet):
    def __init__(self, base=8):
        super().__init__(base)
        self.enc1.layers[0] = AdditiveInputConv(self.enc1.layers[0])

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1] == 3
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


def load_network(path):
    """Load either frozen legacy B or a completed H7 checkpoint, CPU/eval mode."""
    cp = torch.load(path, map_location='cpu', weights_only=False)
    net = HeatmapUNet(cp['base_channels'])
    state = dict(cp['model_state'])
    old_key = 'enc1.layers.0.weight'
    if old_key in state:
        state['enc1.layers.0.original.weight'] = state.pop(old_key)
        state['enc1.layers.0.auxiliary.weight'] = torch.zeros_like(net.enc1.layers[0].auxiliary.weight)
    net.load_state_dict(state, strict=True)
    net.config = cp.get('config', {})
    net.checkpoint_step = cp.get('step')
    net.eval()
    return net


def save_network(path, net, cfg, step, extra=None):
    torch.save({'model_state': net.state_dict(), 'base_channels': net.base,
                'config': cfg, 'step': step, 'extra': extra or {}}, path)


stage_balanced_mse = legacy.stage_balanced_mse
