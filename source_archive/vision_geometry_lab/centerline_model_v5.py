"""Seed-conditioned two-head centreline baseline.

This module is intentionally isolated from the live camera/control stack.  Its
synthetic command is a mechanical smoke test only: it does not establish field
accuracy and it does not replace the missing human annotations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from PIL import Image,ImageDraw,ImageFilter
from .centerline_review_v5 import validate_annotation as validate_review_annotation
from .centerline_quality_v5 import load_label_quality


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=Path(__file__).resolve().parent/'outputs/centerline_v5/synthetic_model'
DEFAULT_MANIFEST_OUTPUT=Path(__file__).resolve().parent/'outputs/centerline_v5/manifest_model'
INPUT_SEMANTICS=('red','green','blue','free_span_seed_heatmap')
OUTPUT_HEADS=('body_logits','center_logits')
RASTER_ENCODING_VERSION=2


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json_sha256(value):
    payload=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                       allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(payload,'utf-8');os.replace(temporary,path)


class _ConvBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(input_channels,output_channels,3,padding=1,bias=False),
            nn.GroupNorm(1,output_channels),nn.SiLU(inplace=True),
            nn.Conv2d(output_channels,output_channels,3,padding=1,bias=False),
            nn.GroupNorm(1,output_channels),nn.SiLU(inplace=True))

    def forward(self,value):
        return self.layers(value)


class TinyCenterlineUNet(nn.Module):
    """Four channels in (RGB + free-span seed), two independent logit heads."""
    def __init__(self,base_channels=8):
        super().__init__()
        if base_channels<2:
            raise ValueError('base_channels must be at least 2')
        self.base_channels=int(base_channels)
        base=self.base_channels
        self.encoder1=_ConvBlock(4,base)
        self.encoder2=_ConvBlock(base,base*2)
        self.bridge=_ConvBlock(base*2,base*4)
        self.decoder2=_ConvBlock(base*4+base*2,base*2)
        self.decoder1=_ConvBlock(base*2+base,base)
        self.body_head=nn.Conv2d(base,1,1)
        self.center_head=nn.Conv2d(base,1,1)

    def forward(self,value):
        if value.ndim!=4 or value.shape[1]!=4:
            raise ValueError('Expected N x 4 x H x W input (RGB + seed heatmap)')
        first=self.encoder1(value)
        second=self.encoder2(F.max_pool2d(first,2,ceil_mode=True))
        bridge=self.bridge(F.max_pool2d(second,2,ceil_mode=True))
        up2=F.interpolate(bridge,size=second.shape[-2:],mode='bilinear',align_corners=False)
        decoded2=self.decoder2(torch.cat([up2,second],dim=1))
        up1=F.interpolate(decoded2,size=first.shape[-2:],mode='bilinear',align_corners=False)
        decoded1=self.decoder1(torch.cat([up1,first],dim=1))
        return {'body_logits':self.body_head(decoded1),
                'center_logits':self.center_head(decoded1)}


def masked_two_head_loss(outputs,body_target,center_target,valid_mask=None,center_weight=1.0,
                         body_valid_mask=None,center_valid_mask=None):
    """BCE over explicit per-head valid pixels; unknown pixels have zero gradient."""
    body_logits=outputs['body_logits'];center_logits=outputs['center_logits']
    if valid_mask is not None:
        if body_valid_mask is None: body_valid_mask=valid_mask
        if center_valid_mask is None: center_valid_mask=valid_mask
    if body_valid_mask is None or center_valid_mask is None:
        raise ValueError('Provide valid_mask or both body_valid_mask and center_valid_mask')
    tensors=(body_logits,center_logits,body_target,center_target,body_valid_mask,center_valid_mask)
    if any(tensor.ndim!=4 or tensor.shape[1]!=1 for tensor in tensors):
        raise ValueError('Logits, targets, and valid masks must be N x 1 x H x W')
    if any(tensor.shape!=body_logits.shape for tensor in tensors[1:]):
        raise ValueError('Logits, targets, and valid masks must have identical shapes')
    if not math.isfinite(float(center_weight)) or float(center_weight)<0:
        raise ValueError('center_weight must be finite and non-negative')
    for name,mask in (('body_valid_mask',body_valid_mask),('center_valid_mask',center_valid_mask)):
        if not torch.isfinite(mask).all() or torch.any((mask!=0)&(mask!=1)):
            raise ValueError(f'{name} must be a binary 0/1 mask')
    body_valid=body_valid_mask.to(dtype=body_logits.dtype)
    center_valid=center_valid_mask.to(dtype=center_logits.dtype)
    body_safe=torch.where(body_valid>0,body_target,torch.zeros_like(body_target)).to(body_logits.dtype)
    center_safe=torch.where(center_valid>0,center_target,torch.zeros_like(center_target)).to(center_logits.dtype)
    for name,target,mask in (('body_target',body_safe,body_valid),
                             ('center_target',center_safe,center_valid)):
        observed=target[mask>0]
        if observed.numel() and (not torch.isfinite(observed).all() or torch.any((observed<0)|(observed>1))):
            raise ValueError(f'{name} must be finite and in [0,1] wherever valid_mask > 0')
    body_weight=body_valid.sum();center_valid_weight=center_valid.sum()
    body_loss=(F.binary_cross_entropy_with_logits(body_logits,body_safe,reduction='none')*body_valid).sum()/body_weight.clamp_min(1.)
    center_loss=(F.binary_cross_entropy_with_logits(center_logits,center_safe,reduction='none')*center_valid).sum()/center_valid_weight.clamp_min(1.)
    total=body_loss+float(center_weight)*center_loss
    return {'loss':total,'body_loss':body_loss,'center_loss':center_loss,
            'body_valid_weight':body_weight.detach(),
            'center_valid_weight':center_valid_weight.detach(),
            'valid_weight':torch.maximum(body_weight,center_valid_weight).detach()}


def split_scene_groups(group_count,validation_fraction=.25,seed=20260926):
    """Split whole synthetic scenes; variants from one scene never cross folds."""
    if group_count<2:
        raise ValueError('At least two scene groups are required')
    if not 0<validation_fraction<1:
        raise ValueError('validation_fraction must be between 0 and 1')
    groups=[f'scene_{index:03d}' for index in range(int(group_count))]
    shuffled=list(groups);random.Random(int(seed)).shuffle(shuffled)
    validation_count=min(len(groups)-1,max(1,round(len(groups)*validation_fraction)))
    validation=sorted(shuffled[:validation_count]);training=sorted(shuffled[validation_count:])
    return training,validation


def _uniform(generator,low,high):
    return low+(high-low)*float(torch.rand((),generator=generator))


def synthetic_curve_sample(group_index,variant_index,size=64,seed=20260926):
    """Generate a transformed target curve plus visually similar neighbouring ropes."""
    if size<16:
        raise ValueError('Synthetic image size must be at least 16')
    scene=torch.Generator().manual_seed(int(seed)+10007*int(group_index))
    variation=torch.Generator().manual_seed(int(seed)+10007*int(group_index)+97*int(variant_index)+1)
    offset=_uniform(scene,-.22,.22);slope=_uniform(scene,-.18,.18)
    amplitude=_uniform(scene,.05,.16);frequency=_uniform(scene,2.2,4.4)
    phase=_uniform(scene,-math.pi,math.pi);quadratic=_uniform(scene,-.08,.08)
    rope_radius=_uniform(scene,.035,.065);neighbor_gap=_uniform(scene,.16,.29)
    angle=math.radians(_uniform(variation,-22.,22.));scale=_uniform(variation,.78,1.22)
    brightness=_uniform(variation,.65,1.35);contrast=_uniform(variation,.75,1.3)
    polarity=-1. if _uniform(variation,0.,1.)<.35 else 1.

    axis=torch.linspace(-1.,1.,int(size));yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    cosine=math.cos(angle);sine=math.sin(angle)
    xr=(cosine*xx+sine*yy)/scale;yr=(-sine*xx+cosine*yy)/scale
    centre=offset+slope*xr+amplitude*torch.sin(frequency*xr+phase)+quadratic*xr.square()
    distance=(yr-centre).abs()
    body=(distance<=rope_radius).float()
    center=torch.exp(-.5*(distance/(rope_radius*.32)).square())
    seed_gate=torch.sigmoid((-xr-.32)*35.)
    seed_heatmap=torch.exp(-.5*(distance/(rope_radius*.7)).square())*seed_gate
    if torch.any(center>0): center=center/center.max()
    if torch.any(seed_heatmap>0): seed_heatmap=seed_heatmap/seed_heatmap.max()

    neighbor=torch.zeros_like(body)
    for sign in (-1.,1.):
        neighbor=torch.maximum(neighbor,torch.exp(-.5*((yr-centre-sign*neighbor_gap)/(rope_radius*.8)).square()))
    noise=torch.randn((int(size),int(size)),generator=variation)*.035
    background=.28+.11*(yy+1.)+.035*torch.sin(9.*xx+phase)+noise
    rope_soft=torch.exp(-.5*(distance/(rope_radius*.72)).square())
    luminance=(background+polarity*.48*rope_soft-.24*neighbor)*contrast*brightness
    rgb=torch.stack([luminance+.025*xx,luminance+.015*yy,luminance-.02*xx],dim=0).clamp(0,1)

    valid=torch.ones_like(body)
    # A deterministic artificial occlusion stands for explicitly unidentifiable
    # pixels. Targets remain present there to prove the loss mask, but do not train.
    if variant_index%2:
        left=int(size*_uniform(variation,.38,.58));span=max(2,int(size*_uniform(variation,.10,.20)))
        valid[:,left:min(size,left+span)]=0
        rgb[:, :, left:min(size,left+span)]=.45+noise[:,left:min(size,left+span)].unsqueeze(0)
    if variant_index%3==2:
        valid[int(size*.82):,:]=0
    input_tensor=torch.cat([rgb,seed_heatmap.unsqueeze(0)],dim=0)
    return {'input':input_tensor.float(),'body_target':body.unsqueeze(0),
            'center_target':center.unsqueeze(0),'valid_mask':valid.unsqueeze(0),
            'group_id':f'scene_{int(group_index):03d}','variant_index':int(variant_index)}


class SyntheticCurveDataset(Dataset):
    def __init__(self,groups,variants_per_group,size=64,seed=20260926):
        self.samples=[(int(group.split('_')[-1]),variant)
                      for group in groups for variant in range(int(variants_per_group))]
        self.size=int(size);self.seed=int(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        group,variant=self.samples[index]
        return synthetic_curve_sample(group,variant,self.size,self.seed)


def _validated_points(value,name,shape):
    points=np.asarray(value,float)
    if points.size==0:
        return np.empty((0,2),float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError(f'{name} must be a finite list of [x,y] points')
    height,width=shape
    if np.any(points[:,0]<0) or np.any(points[:,0]>=width) or np.any(points[:,1]<0) or np.any(points[:,1]>=height):
        raise ValueError(f'{name} contains a point outside image_shape_hw')
    return points


def inspect_manifest(manifest_path):
    """Validate the fixed annotation protocol and classify usable/skipped rows."""
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text('utf-8'))
    if manifest.get('schema_version')!=1 or manifest.get('task')!='seed_conditioned_new_winding_centerline':
        raise ValueError('Unsupported manifest schema_version/task')
    samples=manifest.get('samples')
    if not isinstance(samples,list):
        raise ValueError('manifest.samples must be a list')
    identifiers=[sample.get('sample_id') for sample in samples]
    if (any(not isinstance(value,str) or re.fullmatch(r'[A-Za-z0-9_.-]+',value) is None
            for value in identifiers) or len(set(identifiers))!=len(identifiers)):
        raise ValueError('sample_id must be unique and contain only letters, digits, dot, underscore, or hyphen')
    group_splits={}
    for sample in samples:
        split=sample.get('split');group=sample.get('group_id')
        if split not in ('train','holdout') or not isinstance(group,str) or not group:
            raise ValueError(f"Invalid split/group_id for {sample.get('sample_id')}")
        group_splits.setdefault(group,set()).add(split)
    leaked={group:sorted(splits) for group,splits in group_splits.items() if len(splits)>1}
    if leaked:
        raise ValueError(f'group_id leakage across train/holdout: {leaked}')

    usable={'train':[],'holdout':[]};inference_records=[];skipped=[]
    for sample in samples:
        sample_id=sample['sample_id']
        try:
            image_path=Path(sample['image_path']);annotation_path=Path(sample['annotation_path'])
            if not image_path.is_absolute() or image_path.suffix.lower()!='.png':
                raise ValueError('image_path must be an absolute PNG path')
            if not annotation_path.is_absolute():
                raise ValueError('annotation_path must be absolute')
            shape=sample.get('image_shape_hw')
            if not (isinstance(shape,list) and len(shape)==2 and all(isinstance(v,int) and v>0 for v in shape)):
                raise ValueError('image_shape_hw must be [H,W] positive integers')
            roi=sample.get('roi_xyxy')
            if not (isinstance(roi,list) and len(roi)==4 and all(isinstance(v,int) for v in roi)):
                raise ValueError('roi_xyxy must contain four integers')
            x0,y0,x1,y1=roi
            if not (0<=x0<x1<=shape[1] and 0<=y0<y1<=shape[0]):
                raise ValueError('roi_xyxy is outside image_shape_hw')
            source=sample.get('source')
            if not isinstance(source,dict) or any(key not in source for key in ('video_path','frame_index','time_s','camera')):
                raise ValueError('source must include video_path/frame_index/time_s/camera')
            if not image_path.is_file():
                raise FileNotFoundError('missing_image')
            if _file_sha256(image_path)!=sample.get('image_sha256'):
                raise ValueError('image_sha256 mismatch')
            with Image.open(image_path) as image:
                if image.size!=(shape[1],shape[0]):
                    raise ValueError('decoded PNG shape differs from image_shape_hw')
            if not annotation_path.is_file():
                raise FileNotFoundError('missing_annotation')
            annotation_sha256=_file_sha256(annotation_path)
            label=validate_review_annotation(json.loads(annotation_path.read_text('utf-8-sig')),sample)
            quality=load_label_quality(sample)
            if label.get('schema_version')!=1 or label.get('sample_id')!=sample_id:
                raise ValueError('label schema_version/sample_id mismatch')
            if label.get('image_sha256')!=sample['image_sha256']:
                raise ValueError('label image_sha256 mismatch')
            if label.get('coordinate_system')!='full_image_xy' or label.get('reviewed') is not True:
                raise ValueError('label must be reviewed in full_image_xy coordinates')
            status=label.get('status')
            if status not in ('clear','partial','unidentifiable','absent'):
                raise ValueError('invalid label status')
            polylines=label.get('polylines')
            if not isinstance(polylines,dict) or any(key not in polylines for key in
                    ('centerline','edge_flange','edge_surface','free_seed')):
                raise ValueError('label.polylines is incomplete')
            parsed={key:_validated_points(polylines[key],f'polylines.{key}',shape)
                    for key in ('centerline','edge_flange','edge_surface','free_seed')}
            for name,points in parsed.items():
                if len(points) and (np.any(points[:,0]<x0) or np.any(points[:,0]>=x1) or
                                    np.any(points[:,1]<y0) or np.any(points[:,1]>=y1)):
                    raise ValueError(f'polylines.{name} contains a point outside roi_xyxy')
            polygons=[]
            if not isinstance(label.get('ignore_polygons'),list):
                raise ValueError('ignore_polygons must be a list')
            for index,polygon in enumerate(label['ignore_polygons']):
                parsed_polygon=_validated_points(polygon,f'ignore_polygons[{index}]',shape)
                if len(parsed_polygon)<3:
                    raise ValueError('ignore polygon must contain at least three points')
                polygons.append(parsed_polygon)
            quality_polygons=[]
            for index,polygon in enumerate(quality['precision_exclude_polygons']):
                parsed_polygon=_validated_points(
                    polygon,f'quality.precision_exclude_polygons[{index}]',shape)
                if len(parsed_polygon)<3:
                    raise ValueError('quality precision-exclude polygon must contain at least three points')
                quality_polygons.append(parsed_polygon)
            record=dict(sample);record['label']=label;record['parsed_polylines']=parsed
            record['parsed_ignore_polygons']=polygons
            record['parsed_quality_exclude_polygons']=quality_polygons
            record['quality_sha256']=quality['quality_sha256']
            record['quality_regions']=quality['regions']
            if quality.get('supervision_polylines') is not None:
                record['parsed_supervision_polylines']={
                    key:_validated_points(points,f'quality.supervision_polylines.{key}',shape)
                    for key,points in quality['supervision_polylines'].items()}
            record['annotation_sha256']=annotation_sha256
            record['free_seed_sha256']=_canonical_json_sha256(polylines['free_seed'])
            if len(parsed['free_seed'])>=2:
                inference_records.append(record)
            if status=='unidentifiable':
                skipped.append({'sample_id':sample_id,'split':sample['split'],
                                'reason':'unidentifiable_has_no_target_supervision'})
                continue
            if status=='absent' and any(len(points) for points in parsed.values()):
                skipped.append({'sample_id':sample_id,'split':sample['split'],
                                'reason':'absent_status_must_not_contain_target_polylines'})
                continue
            if status in ('clear','partial') and (len(parsed['centerline'])<2 or len(parsed['free_seed'])<2):
                skipped.append({'sample_id':sample_id,'split':sample['split'],
                                'reason':'missing_centerline_or_free_seed'})
                continue
            usable[sample['split']].append(record)
        except FileNotFoundError as error:
            skipped.append({'sample_id':sample_id,'split':sample['split'],'reason':str(error)})
        except (KeyError,TypeError,ValueError,json.JSONDecodeError) as error:
            skipped.append({'sample_id':sample_id,'split':sample['split'],
                            'reason':'invalid_sample_or_label: '+str(error)})
    return {'manifest':manifest,'manifest_sha256':_file_sha256(manifest_path),
            'usable':usable,'inference_records':inference_records,'skipped':skipped,
            'group_split':{group:next(iter(splits)) for group,splits in group_splits.items()}}


def _local_points(points,roi):
    return [(float(x-roi[0]),float(y-roi[1])) for x,y in points]


def _normalize_nonempty_line_map(array):
    array=np.asarray(array,dtype=np.float32)
    peak=float(array.max()) if array.size else 0.
    if not math.isfinite(peak):
        raise ValueError('Line heatmap contains a non-finite value')
    return array/peak if peak>0 else np.zeros_like(array,dtype=np.float32)


def _polyline_image(size,points,width=1,blur=0.,normalize_peak=False):
    image=Image.new('L',size,0)
    if len(points)>=2:
        ImageDraw.Draw(image).line(points,fill=255,width=max(1,int(width)),joint='curve')
    if blur>0:
        image=image.filter(ImageFilter.GaussianBlur(float(blur)))
    result=np.asarray(image,dtype=np.float32)/255.
    return _normalize_nonempty_line_map(result) if normalize_peak else result


def _resample_polyline(points,count=64):
    points=np.asarray(points,float)
    distances=np.linalg.norm(np.diff(points,axis=0),axis=1)
    cumulative=np.r_[0.,np.cumsum(distances)]
    if cumulative[-1]<=0:
        return np.repeat(points[:1],count,axis=0)
    positions=np.linspace(0,cumulative[-1],count)
    return np.column_stack([np.interp(positions,cumulative,points[:,axis]) for axis in (0,1)])


def _resize_plane(array,size,resample):
    image=Image.fromarray(np.rint(np.clip(array,0,1)*255).astype(np.uint8))
    return np.asarray(image.resize((size,size),resample=resample),dtype=np.float32)/255.


def _resize_exclusion_mask_conservative(exclusion,output_size):
    """Map explicit unknown pixels so no excluded native pixel becomes valid after resize."""
    exclusion=np.asarray(exclusion,dtype=np.float32)
    if exclusion.ndim!=2 or not np.isfinite(exclusion).all():
        raise ValueError('exclusion mask must be a finite HxW array')
    pooled=F.adaptive_max_pool2d(torch.from_numpy(exclusion)[None,None],
                                 (int(output_size),int(output_size)))
    # PIL's bilinear target resize can mix a boundary source pixel into the
    # adjacent model pixel.  One output-pixel guard keeps that interpolation
    # footprint unknown instead of silently turning it back into supervision.
    pooled=F.max_pool2d(pooled,kernel_size=3,stride=1,padding=1)
    return (pooled[0,0].numpy()>0).astype(np.float32)


def manifest_native_input(record):
    """Return native local RGB and human free-seed only; never read target curves."""
    x0,y0,x1,y1=record['roi_xyxy'];width=x1-x0;height=y1-y0
    with Image.open(record['image_path']) as source:
        rgb=np.asarray(source.convert('RGB').crop((x0,y0,x1,y1)),dtype=np.float32)/255.
    free_seed=record['parsed_polylines']['free_seed']
    local_seed=_local_points(free_seed,record['roi_xyxy'])
    short=max(1,min(width,height));seed_blur=max(1.,short*.008)
    seed=_polyline_image((width,height),local_seed,width=max(1,round(short*.006)),
                         blur=seed_blur,normalize_peak=True)
    return rgb,seed


def rasterize_manifest_record(record,output_size):
    """Build local RGB/seed inputs and independent sparse body/centre supervision."""
    x0,y0,x1,y1=record['roi_xyxy'];width=x1-x0;height=y1-y0
    rgb,seed=manifest_native_input(record)
    # RGB/free_seed above always use the original input. Explicitly confirmed
    # target cutoffs must apply BEFORE Gaussian drawing / body polygon filling,
    # otherwise an uncertain tail can contaminate pixels on the trusted side.
    parsed=dict(record['parsed_polylines']);status=record['label']['status']
    parsed.update(record.get('parsed_supervision_polylines',{}))
    local={key:_local_points(points,record['roi_xyxy']) for key,points in parsed.items()}
    short=max(1,min(width,height))
    body=np.zeros((height,width),np.float32);center=np.zeros_like(body)
    body_valid=np.zeros_like(body);center_valid=np.zeros_like(body)
    if status=='absent':
        body_valid[:]=1;center_valid[:]=1
    else:
        center=_polyline_image((width,height),local['centerline'],1,max(1.,short*.006),
                               normalize_peak=True)
        center_valid=_polyline_image((width,height),local['centerline'],
                                     max(3,round(short*.06)),0.)
        if len(local['edge_flange'])>=2 and len(local['edge_surface'])>=2:
            first=_resample_polyline(local['edge_flange']);second=_resample_polyline(local['edge_surface'])
            polygon=[tuple(point) for point in np.vstack([first,second[::-1]])]
            body_image=Image.new('L',(width,height),0);ImageDraw.Draw(body_image).polygon(polygon,fill=255)
            body=np.asarray(body_image,dtype=np.float32)/255.
            dilation=max(3,int(round(short*.04))|1)
            body_valid=np.asarray(body_image.filter(ImageFilter.MaxFilter(dilation)),dtype=np.float32)/255.
    exclusion=Image.new('L',(width,height),0);drawer=ImageDraw.Draw(exclusion)
    for polygon in (*record['parsed_ignore_polygons'],
                    *record.get('parsed_quality_exclude_polygons',[])):
        drawer.polygon(_local_points(polygon,record['roi_xyxy']),fill=255)
    exclusion=np.asarray(exclusion,dtype=np.float32)/255.
    resampling=Image.Resampling
    rgb=np.asarray(Image.fromarray(np.rint(rgb*255).astype(np.uint8)).resize(
        (output_size,output_size),resample=resampling.BILINEAR),dtype=np.float32)/255.
    seed=_resize_plane(seed,output_size,resampling.BILINEAR)
    body=_resize_plane(body,output_size,resampling.BILINEAR)
    center=_resize_plane(center,output_size,resampling.BILINEAR)
    seed=_normalize_nonempty_line_map(seed)
    center=_normalize_nonempty_line_map(center)
    body_valid=_resize_plane(body_valid,output_size,resampling.NEAREST)
    center_valid=_resize_plane(center_valid,output_size,resampling.NEAREST)
    keep=1.-_resize_exclusion_mask_conservative(exclusion,output_size)
    body_valid*=keep;center_valid*=keep
    return {'input':torch.from_numpy(np.concatenate([rgb.transpose(2,0,1),seed[None]],axis=0)).float(),
            'body_target':torch.from_numpy(body[None]).float(),
            'center_target':torch.from_numpy(center[None]).float(),
            'body_valid_mask':torch.from_numpy(body_valid[None]).float(),
            'center_valid_mask':torch.from_numpy(center_valid[None]).float(),
            'sample_id':record['sample_id'],'group_id':record['group_id']}


class ManifestCurveDataset(Dataset):
    def __init__(self,records,output_size=128,cache_samples=False):
        self.records=list(records);self.output_size=int(output_size)
        self.cache_samples=bool(cache_samples)
        self._cache=([rasterize_manifest_record(record,self.output_size)
                      for record in self.records] if self.cache_samples else None)

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        if self._cache is not None:
            return self._cache[index]
        return rasterize_manifest_record(self.records[index],self.output_size)


def manifest_supervision_summary(records,output_size=128):
    summary={'body_valid_sample_count':0,'center_valid_sample_count':0,
             'body_positive_sample_count':0,'center_positive_sample_count':0}
    for record in records:
        raster=rasterize_manifest_record(record,int(output_size))
        body_valid_mask=raster['body_valid_mask'].numpy()>0
        center_valid_mask=raster['center_valid_mask'].numpy()>0
        body_valid=bool(body_valid_mask.any())
        center_valid=bool(center_valid_mask.any())
        body_positive=bool(((raster['body_target'].numpy()>0)*body_valid_mask).any())
        center_positive=bool(((raster['center_target'].numpy()>0)*center_valid_mask).any())
        summary['body_valid_sample_count']+=int(body_valid)
        summary['center_valid_sample_count']+=int(center_valid)
        summary['body_positive_sample_count']+=int(body_positive)
        summary['center_positive_sample_count']+=int(center_positive)
    summary['body_head_supervised']=summary['body_positive_sample_count']>0
    summary['center_head_supervised']=summary['center_positive_sample_count']>0
    summary['supervision_complete']=summary['body_head_supervised'] and summary['center_head_supervised']
    return summary


def manifest_resize_audit(samples,output_size):
    rows=[]
    for sample in samples:
        roi=sample.get('roi_xyxy')
        if not (isinstance(roi,list) and len(roi)==4):
            continue
        width=roi[2]-roi[0];height=roi[3]-roi[1]
        if width<=0 or height<=0:
            continue
        rows.append({'sample_id':sample.get('sample_id'),'native_roi_hw':[height,width],
                     'model_input_hw':[int(output_size),int(output_size)],
                     'scale_xy':[float(output_size/width),float(output_size/height)]})
    return rows


def _batch_losses(model,batch,center_weight,forward_timer=None):
    started=time.perf_counter()
    outputs=model(batch['input'])
    if forward_timer is not None:
        forward_timer['seconds']+=time.perf_counter()-started
    if 'body_valid_mask' in batch:
        losses=masked_two_head_loss(outputs,batch['body_target'],batch['center_target'],None,
                                    center_weight,batch['body_valid_mask'],batch['center_valid_mask'])
    else:
        losses=masked_two_head_loss(outputs,batch['body_target'],batch['center_target'],
                                    batch['valid_mask'],center_weight)
    return outputs,losses


def _model_metrics(model,loader,center_weight,interpretation):
    totals=dict(body_loss_sum=0.,center_loss_sum=0.,body_intersection=0.,body_sum=0.,
                center_intersection=0.,center_sum=0.,body_valid_pixels=0.,center_valid_pixels=0.)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs,losses=_batch_losses(model,batch,center_weight)
            body_valid=(batch.get('body_valid_mask',batch.get('valid_mask'))>0)
            center_valid=(batch.get('center_valid_mask',batch.get('valid_mask'))>0)
            body_prediction=(torch.sigmoid(outputs['body_logits'])>=.5)&body_valid
            body_truth=(batch['body_target']>=.5)&body_valid
            center_prediction=(torch.sigmoid(outputs['center_logits'])>=.5)&center_valid
            center_truth=(batch['center_target']>=.5)&center_valid
            body_weight=float(losses['body_valid_weight']);center_valid_weight=float(losses['center_valid_weight'])
            totals['body_loss_sum']+=float(losses['body_loss'])*body_weight
            totals['center_loss_sum']+=float(losses['center_loss'])*center_valid_weight
            totals['body_intersection']+=float((body_prediction&body_truth).sum())
            totals['body_sum']+=float(body_prediction.sum()+body_truth.sum())
            totals['center_intersection']+=float((center_prediction&center_truth).sum())
            totals['center_sum']+=float(center_prediction.sum()+center_truth.sum())
            totals['body_valid_pixels']+=float(body_valid.sum())
            totals['center_valid_pixels']+=float(center_valid.sum())
    def dice(intersection,total):
        return None if total<=0 else 2.*intersection/total
    body_loss=(None if totals['body_valid_pixels']<=0 else
               totals['body_loss_sum']/totals['body_valid_pixels'])
    center_loss=(None if totals['center_valid_pixels']<=0 else
                 totals['center_loss_sum']/totals['center_valid_pixels'])
    combined=(None if body_loss is None and center_loss is None else
              (0. if body_loss is None else body_loss)+
              float(center_weight)*(0. if center_loss is None else center_loss))
    return {'masked_two_head_loss':combined,'body_masked_loss':body_loss,
            'center_masked_loss':center_loss,
            'body_dice_at_0_5':dice(totals['body_intersection'],totals['body_sum']),
            'center_dice_at_0_5':dice(totals['center_intersection'],totals['center_sum']),
            'body_valid_pixels':int(totals['body_valid_pixels']),
            'center_valid_pixels':int(totals['center_valid_pixels']),
            'interpretation':interpretation}


def _synthetic_metrics(model,loader,center_weight):
    return _model_metrics(model,loader,center_weight,
        'synthetic grouped-holdout mechanical diagnostic only; not field accuracy')


def train_synthetic(args):
    torch.set_num_threads(int(args.threads));torch.manual_seed(int(args.seed))
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    train_groups,validation_groups=split_scene_groups(args.groups,args.validation_fraction,args.seed)
    if set(train_groups)&set(validation_groups):
        raise RuntimeError('Group leakage detected')
    training=SyntheticCurveDataset(train_groups,args.variants,args.size,args.seed)
    validation=SyntheticCurveDataset(validation_groups,args.variants,args.size,args.seed)
    generator=torch.Generator().manual_seed(int(args.seed)+11)
    loader=DataLoader(training,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=generator)
    validation_loader=DataLoader(validation,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=TinyCenterlineUNet(args.base_channels).cpu()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    history=[];iterator=iter(loader);started=time.perf_counter();model.train()
    for step in range(int(args.steps)):
        try: batch=next(iterator)
        except StopIteration:
            iterator=iter(loader);batch=next(iterator)
        optimizer.zero_grad(set_to_none=True)
        _,losses=_batch_losses(model,batch,args.center_weight)
        losses['loss'].backward();optimizer.step()
        history.append({'step':step+1,'loss':float(losses['loss'].detach()),
                        'body_loss':float(losses['body_loss'].detach()),
                        'center_loss':float(losses['center_loss'].detach())})
    elapsed=time.perf_counter()-started
    metrics=_synthetic_metrics(model,validation_loader,args.center_weight)
    checkpoint=output/'checkpoint.pt';temporary=output/'checkpoint.pt.tmp'
    torch.save({'format':'centerline_v5_seed_conditioned_two_head','schema_version':1,
                'raster_encoding_version':RASTER_ENCODING_VERSION,
                'line_map_encoding':'nonempty Gaussian line normalized to peak 1 after model resize; empty stays zero',
                'model_config':{'input_channels':4,'base_channels':int(args.base_channels),
                                'heads':list(OUTPUT_HEADS),
                                'input_size_hw':[int(args.size),int(args.size)]},
                'state_dict':model.state_dict(),
                'input_semantics':list(INPUT_SEMANTICS),
                'output_semantics':{'body':'target rope-body probability',
                                    'center':'target rope-centre heatmap'},
                'training_source':'synthetic_mechanical_smoke_only',
                'train_group_ids':train_groups,
                'body_head_supervised':True,'center_head_supervised':True,
                'body_valid_sample_count':len(training),
                'center_valid_sample_count':len(training),
                'field_validation_performed':False},temporary)
    os.replace(temporary,checkpoint)
    report={'schema_version':1,'status':'synthetic_mechanical_smoke_only',
            'raster_encoding_version':RASTER_ENCODING_VERSION,
            'data':{'source':'procedurally generated curves, neighbouring ropes, and nuisances',
                    'real_annotation_count':0,'split_unit':'synthetic_scene_group',
                    'train_groups':train_groups,'validation_groups':validation_groups,
                    'groups_disjoint':not bool(set(train_groups)&set(validation_groups)),
                    'variants_per_group':int(args.variants)},
            'training':{'device':'cpu','steps_completed':int(args.steps),
                        'elapsed_seconds':elapsed,'threads':int(args.threads),
                        'image_size':[int(args.size),int(args.size)],
                        'base_channels':int(args.base_channels),'loss_history':history,
                        'body_head_supervised':True,'center_head_supervised':True,
                        'body_valid_sample_count':len(training),
                        'center_valid_sample_count':len(training)},
            'synthetic_holdout_mechanical_metrics':metrics,
            'artifacts':{'checkpoint':str(checkpoint),'checkpoint_sha256':_file_sha256(checkpoint)},
            'claim_limits':{'field_accuracy_established':False,'production_ready':False,
                            'real_training_completed':False,
                            'reason':'real labelled target/valid/ignore regions are not yet available'}}
    _write_json(output/'report.json',report)
    print(json.dumps({'report':str(output/'report.json'),'checkpoint':str(checkpoint),
                      'elapsed_seconds':elapsed,'status':report['status']}))
    return report


def train_manifest(args):
    """Train only eligible ``split=train`` labels; inspect holdout once afterwards."""
    command_started=time.perf_counter()
    torch.set_num_threads(int(args.threads));torch.manual_seed(int(args.seed))
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    inspected=inspect_manifest(args.manifest)
    train_records=inspected['usable']['train'];holdout_records=inspected['usable']['holdout']
    supervision=manifest_supervision_summary(train_records,args.size)
    base_report={'schema_version':1,'raster_encoding_version':RASTER_ENCODING_VERSION,
                 'manifest':str(Path(args.manifest).resolve()),
                 'manifest_sha256':inspected['manifest_sha256'],
                 'data':{'split_source':'manifest split field; no random frame split',
                         'train_sample_ids':[record['sample_id'] for record in train_records],
                         'holdout_sample_ids':[record['sample_id'] for record in holdout_records],
                         'train_groups':sorted({record['group_id'] for record in train_records}),
                         'holdout_groups':sorted({record['group_id'] for record in holdout_records}),
                         'train_annotation_sha256_by_sample':{
                             record['sample_id']:record['annotation_sha256'] for record in train_records},
                         'holdout_annotation_sha256_by_sample':{
                             record['sample_id']:record['annotation_sha256'] for record in holdout_records},
                         'train_quality_sha256_by_sample':{
                             record['sample_id']:record['quality_sha256'] for record in train_records},
                         'holdout_quality_sha256_by_sample':{
                             record['sample_id']:record['quality_sha256'] for record in holdout_records},
                         'skipped_samples':inspected['skipped'],
                         'holdout_used_for_training':False,
                         'holdout_used_for_tuning_by_code':False,
                         'resize_audit':manifest_resize_audit(inspected['manifest']['samples'],args.size)},
                 'supervision':supervision,
                 'claim_limits':{'field_accuracy_established':False,'production_ready':False,
                                 'reason':'requires independent reviewed field acceptance after annotation'}}
    if not train_records:
        report=dict(base_report,status='blocked_no_reviewed_labels',
                    training={'steps_completed':0,'elapsed_seconds':0.,'device':'cpu'},
                    holdout_diagnostic=None,artifacts={'checkpoint':None})
        _write_json(output/'report.json',report)
        print(json.dumps({'report':str(output/'report.json'),'status':report['status']}))
        return report
    if not supervision['body_head_supervised'] and not supervision['center_head_supervised']:
        report=dict(base_report,status='blocked_no_positive_target_supervision',
                    training={'steps_completed':0,'elapsed_seconds':0.,'device':'cpu'},
                    holdout_diagnostic=None,artifacts={'checkpoint':None})
        _write_json(output/'report.json',report)
        print(json.dumps({'report':str(output/'report.json'),'status':report['status']}))
        return report

    cache_started=time.perf_counter()
    training=ManifestCurveDataset(train_records,args.size,args.cache_samples)
    loader=DataLoader(training,batch_size=args.batch_size,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(int(args.seed)+23))
    holdout_loader=(DataLoader(ManifestCurveDataset(holdout_records,args.size,args.cache_samples),
                               batch_size=args.batch_size,shuffle=False,num_workers=0)
                    if holdout_records else None)
    cache_preparation_seconds=time.perf_counter()-cache_started
    model=TinyCenterlineUNet(args.base_channels).cpu()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    history=[];iterator=iter(loader);started=time.perf_counter();model.train()
    forward_timer={'seconds':0.};batch_fetch_seconds=0.
    for step in range(int(args.steps)):
        fetch_started=time.perf_counter()
        try: batch=next(iterator)
        except StopIteration:
            iterator=iter(loader);batch=next(iterator)
        batch_fetch_seconds+=time.perf_counter()-fetch_started
        optimizer.zero_grad(set_to_none=True)
        _,losses=_batch_losses(model,batch,args.center_weight,forward_timer)
        if float(losses['body_valid_weight']+losses['center_valid_weight'])<=0:
            raise ValueError('Eligible training batch unexpectedly has no supervised pixels')
        losses['loss'].backward();optimizer.step()
        history.append({'step':step+1,'loss':float(losses['loss'].detach()),
                        'body_loss':float(losses['body_loss'].detach()),
                        'center_loss':float(losses['center_loss'].detach()),
                        'body_valid_pixels':int(losses['body_valid_weight']),
                        'center_valid_pixels':int(losses['center_valid_weight'])})
    elapsed=time.perf_counter()-started
    holdout=(_model_metrics(model,holdout_loader,args.center_weight,
             'reviewed manifest holdout diagnostic; not used for training or tuning by this command')
             if holdout_loader is not None else None)
    checkpoint=output/'checkpoint.pt';temporary=output/'checkpoint.pt.tmp'
    torch.save({'format':'centerline_v5_seed_conditioned_two_head','schema_version':1,
                'raster_encoding_version':RASTER_ENCODING_VERSION,
                'line_map_encoding':'nonempty Gaussian line normalized to peak 1 after model resize; empty stays zero',
                'model_config':{'input_channels':4,'base_channels':int(args.base_channels),
                                'heads':list(OUTPUT_HEADS),
                                'input_size_hw':[int(args.size),int(args.size)]},
                'state_dict':model.state_dict(),
                'input_semantics':list(INPUT_SEMANTICS),
                'output_semantics':{'body':'target rope-body probability',
                                    'center':'target rope-centre heatmap'},
                'training_manifest_sha256':inspected['manifest_sha256'],
                'training_annotation_sha256_by_sample':{
                    record['sample_id']:record['annotation_sha256'] for record in train_records},
                'train_quality_sha256_by_sample':{
                    record['sample_id']:record['quality_sha256'] for record in train_records},
                'training_source':'reviewed_manifest_annotations',
                'train_group_ids':sorted({record['group_id'] for record in train_records}),
                'body_head_supervised':supervision['body_head_supervised'],
                'center_head_supervised':supervision['center_head_supervised'],
                'body_valid_sample_count':supervision['body_valid_sample_count'],
                'center_valid_sample_count':supervision['center_valid_sample_count'],
                'body_positive_sample_count':supervision['body_positive_sample_count'],
                'center_positive_sample_count':supervision['center_positive_sample_count'],
                'field_validation_performed':False},temporary)
    os.replace(temporary,checkpoint)
    training_status=('manifest_training_complete_unvalidated' if supervision['supervision_complete']
                     else 'partial_supervision')
    report=dict(base_report,status=training_status,
                raster_encoding_version=RASTER_ENCODING_VERSION,
                training={'device':'cpu','steps_completed':int(args.steps),
                          'elapsed_seconds':elapsed,'threads':int(args.threads),
                          'cache_samples':bool(args.cache_samples),
                          'cache_preparation_seconds':cache_preparation_seconds,
                          'batch_fetch_seconds':batch_fetch_seconds,
                          'model_forward_seconds':forward_timer['seconds'],
                          'command_elapsed_before_report_seconds':time.perf_counter()-command_started,
                          'resize_hw':[int(args.size),int(args.size)],
                          'base_channels':int(args.base_channels),'loss_history':history},
                holdout_diagnostic=holdout,
                artifacts={'checkpoint':str(checkpoint),'checkpoint_sha256':_file_sha256(checkpoint)})
    _write_json(output/'report.json',report)
    print(json.dumps({'report':str(output/'report.json'),'checkpoint':str(checkpoint),
                      'elapsed_seconds':elapsed,'status':report['status']}))
    return report


def load_model(checkpoint_path):
    checkpoint=torch.load(Path(checkpoint_path),map_location='cpu',weights_only=True)
    if checkpoint.get('format')!='centerline_v5_seed_conditioned_two_head':
        raise ValueError('Unsupported checkpoint format')
    if checkpoint.get('raster_encoding_version')!=RASTER_ENCODING_VERSION:
        raise ValueError('Legacy checkpoint uses an incompatible line-map encoding; retrain it with raster_encoding_version=2')
    config=checkpoint.get('model_config',{})
    if config.get('input_channels')!=4 or config.get('heads')!=['body_logits','center_logits']:
        raise ValueError('Checkpoint input/head semantics mismatch')
    input_size=config.get('input_size_hw')
    if not (isinstance(input_size,list) and len(input_size)==2 and
            all(isinstance(value,int) and value>=16 for value in input_size)):
        raise ValueError('Checkpoint lacks valid model_config.input_size_hw; regenerate it before inference')
    model=TinyCenterlineUNet(int(config['base_channels'])).cpu()
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    return model,checkpoint


def _resize_native_model_input(rgb,seed,input_size_hw):
    model_height,model_width=map(int,input_size_hw)
    resampling=Image.Resampling
    rgb_uint8=np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)
    resized_rgb=np.asarray(Image.fromarray(rgb_uint8,'RGB').resize(
        (model_width,model_height),resample=resampling.BILINEAR),dtype=np.float32)/255.
    seed_image=Image.fromarray(np.rint(np.clip(seed,0,1)*255).astype(np.uint8),'L')
    resized_seed=np.asarray(seed_image.resize((model_width,model_height),
        resample=resampling.BILINEAR),dtype=np.float32)/255.
    return resized_rgb,_normalize_nonempty_line_map(resized_seed)


def _predict_native_input(model,rgb,seed,input_size_hw):
    resized_rgb,resized_seed=_resize_native_model_input(rgb,seed,input_size_hw)
    tensor=np.concatenate([resized_rgb.transpose(2,0,1),resized_seed[None]],axis=0)
    started=time.perf_counter()
    with torch.inference_mode():
        outputs=model(torch.from_numpy(tensor).unsqueeze(0))
        body=torch.sigmoid(outputs['body_logits'])[0,0].numpy().astype(np.float32,copy=False)
        center=torch.sigmoid(outputs['center_logits'])[0,0].numpy().astype(np.float32,copy=False)
    return body,center,(time.perf_counter()-started)*1000.


def _write_prediction(output,body,center):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,body_probability=body.astype(np.float32,copy=False),
                        center_probability=center.astype(np.float32,copy=False))
    return output


def infer_npz(args):
    torch.set_num_threads(int(args.threads));started=time.perf_counter()
    model,checkpoint=load_model(args.checkpoint)
    with np.load(args.input_npz,allow_pickle=False) as source:
        if 'rgb' not in source or 'seed_heatmap' not in source:
            raise ValueError('Input NPZ must contain rgb and seed_heatmap arrays')
        rgb=np.asarray(source['rgb']);seed=np.asarray(source['seed_heatmap'])
    if rgb.ndim!=3 or rgb.shape[2]!=3 or seed.shape!=rgb.shape[:2]:
        raise ValueError('Expected rgb HxWx3 and seed_heatmap HxW with matching dimensions')
    if not args.sample_id or len(args.image_sha256)!=64 or any(char not in '0123456789abcdefABCDEF' for char in args.image_sha256):
        raise ValueError('sample_id and a 64-character hexadecimal image_sha256 are required')
    rx0,ry0,rx1,ry1=args.roi_xyxy
    if not (rx0<rx1 and ry0<ry1 and rx1-rx0==rgb.shape[1] and ry1-ry0==rgb.shape[0]):
        raise ValueError('roi_xyxy dimensions must match the local RGB/seed arrays')
    rgb=rgb.astype(np.float32);seed=seed.astype(np.float32)
    if rgb.size and rgb.max()>1.: rgb/=255.
    if not np.isfinite(rgb).all() or not np.isfinite(seed).all():
        raise ValueError('Inference input must be finite')
    if rgb.min()<0 or rgb.max()>1 or seed.min()<0 or seed.max()>1:
        raise ValueError('Normalized RGB and seed_heatmap must be in [0,1]')
    input_size_hw=checkpoint['model_config']['input_size_hw']
    body,center,inference_ms=_predict_native_input(model,rgb,seed,input_size_hw)
    output=_write_prediction(args.output_npz,body,center)
    elapsed=time.perf_counter()-started
    training_source=checkpoint.get('training_source','unknown')
    report={'schema_version':1,'raster_encoding_version':checkpoint['raster_encoding_version'],
            'status':'unvalidated_model_inference',
            'sample_id':args.sample_id,'image_sha256':args.image_sha256,
            'roi_xyxy':[int(value) for value in args.roi_xyxy],
            'input':str(Path(args.input_npz)),'input_sha256':_file_sha256(args.input_npz),
            'checkpoint':str(Path(args.checkpoint)),'checkpoint_sha256':_file_sha256(args.checkpoint),
            'output':str(output),'output_sha256':_file_sha256(output),
            'probability_arrays':{'body_probability':'model-resolution HxW float32 spanning native ROI',
                                  'center_probability':'model-resolution HxW float32 spanning native ROI'},
            'native_roi_hw':[int(rgb.shape[0]),int(rgb.shape[1])],
            'model_input_hw':[int(input_size_hw[0]),int(input_size_hw[1])],
            'probability_map_hw':[int(body.shape[0]),int(body.shape[1])],
            'scale_xy':[float(input_size_hw[1]/rgb.shape[1]),float(input_size_hw[0]/rgb.shape[0])],
            'inference_ms':inference_ms,'total_elapsed_seconds':elapsed,
            'train_group_ids':checkpoint.get('train_group_ids',[]),
            'train_quality_sha256_by_sample':checkpoint.get('train_quality_sha256_by_sample',{}),
            'human_free_seed_used':args.free_seed_source=='human_polyline',
            'free_seed_source':args.free_seed_source,
            'ground_truth_center_or_edges_used_as_input':False,
            'checkpoint_training_source':training_source,
            'real_annotation_trained':training_source=='reviewed_manifest_annotations',
            'body_head_supervised':bool(checkpoint.get('body_head_supervised',False)),
            'center_head_supervised':bool(checkpoint.get('center_head_supervised',False)),
            'body_valid_sample_count':int(checkpoint.get('body_valid_sample_count',0)),
            'center_valid_sample_count':int(checkpoint.get('center_valid_sample_count',0)),
            'supervision_complete':bool(checkpoint.get('body_head_supervised',False) and
                                        checkpoint.get('center_head_supervised',False)),
            'field_accuracy_established':bool(checkpoint.get('field_validation_performed',False)),
            'hardware_access':False,'force_command':None}
    report_path=Path(args.report) if args.report else output.with_suffix('.json')
    _write_json(report_path,report);print(json.dumps({'report':str(report_path),'output':str(output)}))
    return report


def infer_manifest(args):
    """Export reviewed free-seed cases without feeding target centre/edges to the model."""
    torch.set_num_threads(int(args.threads))
    model,checkpoint=load_model(args.checkpoint)
    inspected=inspect_manifest(args.manifest);output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    records=[record for record in inspected['inference_records']
             if args.split=='all' or record['split']==args.split]
    checkpoint_hash=_file_sha256(args.checkpoint);input_size_hw=checkpoint['model_config']['input_size_hw']
    training_source=checkpoint.get('training_source','unknown');exported=[]
    for record in records:
        started=time.perf_counter();rgb,seed=manifest_native_input(record)
        input_path=output/'inputs'/f"{record['sample_id']}.npz"
        input_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(input_path,rgb=rgb.astype(np.float32,copy=False),
                            seed_heatmap=seed.astype(np.float32,copy=False))
        body,center,inference_ms=_predict_native_input(model,rgb,seed,input_size_hw)
        prediction_path=_write_prediction(output/'predictions'/f"{record['sample_id']}.npz",body,center)
        native_hw=[int(rgb.shape[0]),int(rgb.shape[1])]
        metadata={'schema_version':1,'raster_encoding_version':checkpoint['raster_encoding_version'],
                  'status':'unvalidated_model_inference',
                  'sample_id':record['sample_id'],'group_id':record['group_id'],
                  'manifest_split':record['split'],'image_sha256':record['image_sha256'],
                  'annotation_sha256':record['annotation_sha256'],
                  'free_seed_sha256':record['free_seed_sha256'],
                  'quality_sha256':record['quality_sha256'],
                  'roi_xyxy':record['roi_xyxy'],'native_roi_hw':native_hw,
                  'model_input_hw':[int(input_size_hw[0]),int(input_size_hw[1])],
                  'probability_map_hw':[int(body.shape[0]),int(body.shape[1])],
                  'scale_xy':[float(input_size_hw[1]/rgb.shape[1]),
                              float(input_size_hw[0]/rgb.shape[0])],
                  'probability_arrays':{'body_probability':'model-resolution HxW float32 spanning native ROI',
                                        'center_probability':'model-resolution HxW float32 spanning native ROI'},
                  'input_npz':str(input_path),'input_sha256':_file_sha256(input_path),
                  'output_npz':str(prediction_path),'output_sha256':_file_sha256(prediction_path),
                  'checkpoint':str(Path(args.checkpoint)),'checkpoint_sha256':checkpoint_hash,
                  'inference_ms':inference_ms,
                  'total_elapsed_seconds':time.perf_counter()-started,
                  'train_group_ids':checkpoint.get('train_group_ids',[]),
                  'train_quality_sha256_by_sample':checkpoint.get('train_quality_sha256_by_sample',{}),
                  'human_free_seed_used':True,'free_seed_source':'human_polyline',
                  'free_seed_point_count':int(len(record['parsed_polylines']['free_seed'])),
                  'ground_truth_center_or_edges_used_as_input':False,
                  'checkpoint_training_source':training_source,
                  'real_annotation_trained':training_source=='reviewed_manifest_annotations',
                  'body_head_supervised':bool(checkpoint.get('body_head_supervised',False)),
                  'center_head_supervised':bool(checkpoint.get('center_head_supervised',False)),
                  'body_valid_sample_count':int(checkpoint.get('body_valid_sample_count',0)),
                  'center_valid_sample_count':int(checkpoint.get('center_valid_sample_count',0)),
                  'supervision_complete':bool(checkpoint.get('body_head_supervised',False) and
                                              checkpoint.get('center_head_supervised',False)),
                  'field_accuracy_established':bool(checkpoint.get('field_validation_performed',False)),
                  'hardware_access':False,'force_command':None}
        metadata_path=prediction_path.with_suffix('.json');_write_json(metadata_path,metadata)
        exported.append({'sample_id':record['sample_id'],'split':record['split'],
                         'prediction':str(prediction_path),'metadata':str(metadata_path)})
    selected_samples=[sample for sample in inspected['manifest']['samples']
                      if args.split=='all' or sample['split']==args.split]
    exported_ids={item['sample_id'] for item in exported}
    reason_by_id={item['sample_id']:item['reason'] for item in inspected['skipped']}
    skipped=[{'sample_id':sample['sample_id'],'split':sample['split'],
              'reason':reason_by_id.get(sample['sample_id'],'no_reviewed_free_seed')}
             for sample in selected_samples if sample['sample_id'] not in exported_ids]
    summary={'schema_version':1,'raster_encoding_version':checkpoint['raster_encoding_version'],
             'status':('complete' if exported else 'blocked_no_reviewed_free_seed'),
             'manifest':str(Path(args.manifest).resolve()),
             'manifest_sha256':inspected['manifest_sha256'],'requested_split':args.split,
             'checkpoint':str(Path(args.checkpoint)),'checkpoint_sha256':checkpoint_hash,
             'checkpoint_training_source':training_source,
             'train_quality_sha256_by_sample':checkpoint.get('train_quality_sha256_by_sample',{}),
             'real_annotation_trained':training_source=='reviewed_manifest_annotations',
             'body_head_supervised':bool(checkpoint.get('body_head_supervised',False)),
             'center_head_supervised':bool(checkpoint.get('center_head_supervised',False)),
             'body_valid_sample_count':int(checkpoint.get('body_valid_sample_count',0)),
             'center_valid_sample_count':int(checkpoint.get('center_valid_sample_count',0)),
             'supervision_complete':bool(checkpoint.get('body_head_supervised',False) and
                                         checkpoint.get('center_head_supervised',False)),
             'model_input_hw':[int(input_size_hw[0]),int(input_size_hw[1])],
             'exported':exported,'skipped':skipped,
             'ground_truth_center_or_edges_used_as_input':False,
             'hardware_access':False,'force_command':None}
    _write_json(output/'summary.json',summary)
    print(json.dumps({'summary':str(output/'summary.json'),'exported':len(exported),
                      'skipped':len(skipped),'status':summary['status']}))
    return summary


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    train=commands.add_parser('train-synthetic',help='Run a synthetic mechanical smoke test only')
    train.add_argument('--output-dir',default=str(DEFAULT_OUTPUT));train.add_argument('--steps',type=int,default=12)
    train.add_argument('--groups',type=int,default=8);train.add_argument('--variants',type=int,default=4)
    train.add_argument('--validation-fraction',type=float,default=.25);train.add_argument('--size',type=int,default=64)
    train.add_argument('--base-channels',type=int,default=8);train.add_argument('--batch-size',type=int,default=2)
    train.add_argument('--learning-rate',type=float,default=1e-3);train.add_argument('--center-weight',type=float,default=1.)
    train.add_argument('--seed',type=int,default=20260926);train.add_argument('--threads',type=int,default=2)
    train.set_defaults(action=train_synthetic)
    manifest=commands.add_parser('train-manifest',help='Train from reviewed schema-v1 labels only')
    manifest.add_argument('--manifest',required=True);manifest.add_argument('--output-dir',default=str(DEFAULT_MANIFEST_OUTPUT))
    manifest.add_argument('--steps',type=int,default=100);manifest.add_argument('--size',type=int,default=512)
    manifest.add_argument('--base-channels',type=int,default=8);manifest.add_argument('--batch-size',type=int,default=2)
    manifest.add_argument('--learning-rate',type=float,default=1e-3);manifest.add_argument('--center-weight',type=float,default=1.)
    manifest.add_argument('--seed',type=int,default=20260926);manifest.add_argument('--threads',type=int,default=2)
    manifest.add_argument('--cache-samples',action='store_true',
                          help='Rasterize each used manifest sample once in memory; no augmentation is changed')
    manifest.set_defaults(action=train_manifest)
    infer=commands.add_parser('infer',help='Infer from NPZ containing rgb and seed_heatmap')
    infer.add_argument('--checkpoint',required=True);infer.add_argument('--input-npz',required=True)
    infer.add_argument('--output-npz',required=True);infer.add_argument('--report');infer.add_argument('--threads',type=int,default=2)
    infer.add_argument('--sample-id',required=True);infer.add_argument('--image-sha256',required=True)
    infer.add_argument('--roi-xyxy',type=int,nargs=4,required=True)
    infer.add_argument('--free-seed-source',choices=('human_polyline','external_detector'),required=True)
    infer.set_defaults(action=infer_npz)
    batch=commands.add_parser('infer-manifest',help='Export every reviewed sample with a valid human free_seed')
    batch.add_argument('--manifest',required=True);batch.add_argument('--checkpoint',required=True)
    batch.add_argument('--output-dir',required=True);batch.add_argument('--split',choices=('all','train','holdout'),default='all')
    batch.add_argument('--threads',type=int,default=2);batch.set_defaults(action=infer_manifest)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    for name in ('steps','groups','variants','size','base_channels','batch_size','threads'):
        if hasattr(args,name) and getattr(args,name)<=0:
            raise ValueError(f'{name} must be positive')
    return args.action(args)


if __name__=='__main__':
    main()
