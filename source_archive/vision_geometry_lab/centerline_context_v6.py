"""Aspect-preserving seed-conditioned centreline experiment.

V6 is an offline CPU experiment only.  It consumes native ROI RGB plus the
human free-span seed, applies one shared fit-and-pad transform, and predicts
rope-body and centre heatmaps.  It never imports the live camera/control stack.
"""
from pathlib import Path
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from PIL import Image,ImageDraw,ImageFilter

from .centerline_model_v5 import (
    INPUT_SEMANTICS,OUTPUT_HEADS,RASTER_ENCODING_VERSION,
    TinyCenterlineUNet,_canonical_json_sha256,_file_sha256,_local_points,
    _normalize_nonempty_line_map,_polyline_image,_resample_polyline,
    inspect_manifest,manifest_native_input,masked_two_head_loss)


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=Path(__file__).resolve().parent/'outputs/centerline_v6/manifest_model'
CHECKPOINT_FORMAT='centerline_context_v6_letterbox_two_head'
PREPROCESS_VERSION=1
DEFAULT_TARGET_HW=(512,1024)
ARCHITECTURES=('shallow','context')


def _write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(payload,'utf-8');os.replace(temporary,path)


def _positive_hw(value,name):
    if (not isinstance(value,(tuple,list)) or len(value)!=2 or
            any(isinstance(item,bool) or not isinstance(item,int) or item<=0 for item in value)):
        raise ValueError(f'{name} must be [height,width] positive integers')
    return int(value[0]),int(value[1])


def letterbox_geometry(native_hw,target_hw=DEFAULT_TARGET_HW):
    """Return a deterministic centred fit-and-pad transform.

    Pixel centres follow ``(native + .5) * scale - .5 + pad``.  Separate
    rounded x/y scales are recorded because integer content dimensions cannot
    preserve an arbitrary aspect ratio perfectly.
    """
    native_height,native_width=_positive_hw(native_hw,'native_hw')
    target_height,target_width=_positive_hw(target_hw,'target_hw')
    scale=min(target_width/native_width,target_height/native_height)
    content_width=max(1,min(target_width,int(round(native_width*scale))))
    content_height=max(1,min(target_height,int(round(native_height*scale))))
    left=(target_width-content_width)//2;top=(target_height-content_height)//2
    right=target_width-content_width-left;bottom=target_height-content_height-top
    return {'native_hw':[native_height,native_width],
            'model_input_hw':[target_height,target_width],
            'content_hw':[content_height,content_width],
            'content_xyxy':[left,top,left+content_width,top+content_height],
            'padding_ltrb':[left,top,right,bottom],
            'scale_xy':[content_width/native_width,content_height/native_height],
            'mapping':'model_xy=(native_local_xy+0.5)*scale_xy-0.5+pad_left_top'}


def full_image_xy_to_model_xy(points,roi_xyxy,geometry):
    points=np.asarray(points,dtype=np.float64)
    if points.size==0:return np.empty((0,2),dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('points must be a finite Nx2 array')
    x0,y0,_,_=map(float,roi_xyxy);left,top,_,_=geometry['content_xyxy']
    scale=np.asarray(geometry['scale_xy'],dtype=np.float64)
    return (points-np.array([x0,y0])+.5)*scale-.5+np.array([left,top])


def model_xy_to_full_image_xy(points,roi_xyxy,geometry):
    points=np.asarray(points,dtype=np.float64)
    if points.size==0:return np.empty((0,2),dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('points must be a finite Nx2 array')
    x0,y0,_,_=map(float,roi_xyxy);left,top,_,_=geometry['content_xyxy']
    scale=np.asarray(geometry['scale_xy'],dtype=np.float64)
    return (points-np.array([left,top])+.5)/scale-.5+np.array([x0,y0])


def _resize_plane(array,content_hw,resample):
    content_height,content_width=_positive_hw(content_hw,'content_hw')
    image=Image.fromarray(np.rint(np.clip(array,0,1)*255).astype(np.uint8),'L')
    return np.asarray(image.resize((content_width,content_height),resample=resample),
                      dtype=np.float32)/255.


def _place_plane(content,geometry,value=0.):
    height,width=geometry['model_input_hw'];left,top,right,bottom=geometry['content_xyxy']
    canvas=np.full((height,width),float(value),dtype=np.float32)
    if content.shape!=(bottom-top,right-left):
        raise ValueError('content plane shape does not match letterbox geometry')
    canvas[top:bottom,left:right]=content
    return canvas


def _place_rgb(content,geometry,value=0.):
    height,width=geometry['model_input_hw'];left,top,right,bottom=geometry['content_xyxy']
    canvas=np.full((height,width,3),float(value),dtype=np.float32)
    if content.shape!=(bottom-top,right-left,3):
        raise ValueError('content RGB shape does not match letterbox geometry')
    canvas[top:bottom,left:right]=content
    return canvas


def _resize_exclusion_conservative(exclusion,content_hw):
    exclusion=np.asarray(exclusion,dtype=np.float32)
    if exclusion.ndim!=2 or not np.isfinite(exclusion).all():
        raise ValueError('exclusion mask must be a finite HxW array')
    target_height,target_width=_positive_hw(content_hw,'content_hw')
    pooled=F.adaptive_max_pool2d(torch.from_numpy(exclusion)[None,None],
                                 (target_height,target_width))
    # Guard the one-output-pixel footprint touched by bilinear target resize.
    pooled=F.max_pool2d(pooled,kernel_size=3,stride=1,padding=1)
    return (pooled[0,0].numpy()>0).astype(np.float32)


def letterbox_native_input(rgb,seed,target_hw=DEFAULT_TARGET_HW):
    """Shared training/inference RGB+seed transform; target curves are absent."""
    rgb=np.asarray(rgb,dtype=np.float32);seed=np.asarray(seed,dtype=np.float32)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or seed.shape!=rgb.shape[:2]:
        raise ValueError('Expected native RGB HxWx3 and matching seed HxW')
    if not np.isfinite(rgb).all() or not np.isfinite(seed).all():
        raise ValueError('RGB and seed must be finite')
    if np.any((rgb<0)|(rgb>1)) or np.any((seed<0)|(seed>1)):
        raise ValueError('RGB and seed must be normalized to [0,1]')
    geometry=letterbox_geometry(rgb.shape[:2],target_hw)
    content_height,content_width=geometry['content_hw'];resampling=Image.Resampling
    rgb_uint8=np.rint(rgb*255).astype(np.uint8)
    resized_rgb=np.asarray(Image.fromarray(rgb_uint8,'RGB').resize(
        (content_width,content_height),resample=resampling.BILINEAR),dtype=np.float32)/255.
    resized_seed=_resize_plane(seed,geometry['content_hw'],resampling.BILINEAR)
    resized_seed=_normalize_nonempty_line_map(resized_seed)
    canvas_rgb=_place_rgb(resized_rgb,geometry)
    canvas_seed=_place_plane(resized_seed,geometry)
    tensor=np.concatenate([canvas_rgb.transpose(2,0,1),canvas_seed[None]],axis=0)
    return torch.from_numpy(tensor).float(),geometry


def letterbox_manifest_input(record,target_hw=DEFAULT_TARGET_HW):
    rgb,seed=manifest_native_input(record)
    tensor,geometry=letterbox_native_input(rgb,seed,target_hw)
    return tensor,geometry,rgb,seed


def _native_supervision(record):
    x0,y0,x1,y1=record['roi_xyxy'];width=x1-x0;height=y1-y0
    # Cut target geometry before drawing/blurring it; the exclusion mask alone
    # cannot remove unknown-tail influence from trusted-side target pixels.
    # Input construction remains RGB + the original free_seed only.
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
            first=_resample_polyline(local['edge_flange'])
            second=_resample_polyline(local['edge_surface'])
            polygon=[tuple(point) for point in np.vstack([first,second[::-1]])]
            body_image=Image.new('L',(width,height),0)
            ImageDraw.Draw(body_image).polygon(polygon,fill=255)
            body=np.asarray(body_image,dtype=np.float32)/255.
            dilation=max(3,int(round(short*.04))|1)
            body_valid=np.asarray(body_image.filter(ImageFilter.MaxFilter(dilation)),
                                  dtype=np.float32)/255.
    exclusion_image=Image.new('L',(width,height),0);drawer=ImageDraw.Draw(exclusion_image)
    for polygon in (*record['parsed_ignore_polygons'],
                    *record.get('parsed_quality_exclude_polygons',[])):
        drawer.polygon(_local_points(polygon,record['roi_xyxy']),fill=255)
    exclusion=np.asarray(exclusion_image,dtype=np.float32)/255.
    return body,center,body_valid,center_valid,exclusion


def rasterize_letterbox_record(record,target_hw=DEFAULT_TARGET_HW):
    """Rasterize native annotations directly into one aspect-preserving canvas."""
    input_tensor,geometry,_,_=letterbox_manifest_input(record,target_hw)
    body,center,body_valid,center_valid,exclusion=_native_supervision(record)
    resampling=Image.Resampling;content_hw=geometry['content_hw']
    body=_resize_plane(body,content_hw,resampling.BILINEAR)
    center=_normalize_nonempty_line_map(_resize_plane(center,content_hw,resampling.BILINEAR))
    body_valid=_resize_plane(body_valid,content_hw,resampling.NEAREST)
    center_valid=_resize_plane(center_valid,content_hw,resampling.NEAREST)
    keep=1.-_resize_exclusion_conservative(exclusion,content_hw)
    body_valid*=keep;center_valid*=keep
    body=_place_plane(body,geometry);center=_place_plane(center,geometry)
    body_valid=_place_plane(body_valid,geometry);center_valid=_place_plane(center_valid,geometry)
    return {'input':input_tensor,
            'body_target':torch.from_numpy(body[None]).float(),
            'center_target':torch.from_numpy(center[None]).float(),
            'body_valid_mask':torch.from_numpy(body_valid[None]).float(),
            'center_valid_mask':torch.from_numpy(center_valid[None]).float(),
            'sample_id':record['sample_id'],'group_id':record['group_id']}


def crop_letterbox_probabilities(array,geometry):
    array=np.asarray(array)
    if array.ndim!=2 or list(array.shape)!=geometry['model_input_hw']:
        raise ValueError('probability map must match model_input_hw')
    left,top,right,bottom=geometry['content_xyxy']
    return np.ascontiguousarray(array[top:bottom,left:right])


class ManifestLetterboxDataset(Dataset):
    def __init__(self,records,target_hw=DEFAULT_TARGET_HW,cache_samples=False):
        self.records=list(records);self.target_hw=_positive_hw(target_hw,'target_hw')
        self.cache_samples=bool(cache_samples)
        self._cache=([rasterize_letterbox_record(record,self.target_hw)
                      for record in self.records] if self.cache_samples else None)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        if self._cache is not None:return self._cache[index]
        return rasterize_letterbox_record(self.records[index],self.target_hw)

    def cache_bytes(self):
        if self._cache is None:return 0
        keys=('input','body_target','center_target','body_valid_mask','center_valid_mask')
        return int(sum(sample[key].numel()*sample[key].element_size()
                       for sample in self._cache for key in keys))


class _ConvBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(input_channels,output_channels,3,padding=1,bias=False),
            nn.GroupNorm(1,output_channels),nn.SiLU(inplace=True),
            nn.Conv2d(output_channels,output_channels,3,padding=1,bias=False),
            nn.GroupNorm(1,output_channels),nn.SiLU(inplace=True))

    def forward(self,value):return self.layers(value)


class ContextCenterlineUNet(nn.Module):
    """Six-pool capped-channel U-Net with a third bridge convolution."""
    depth=6

    def __init__(self,base_channels=8,channel_cap=64):
        super().__init__()
        if base_channels<2 or channel_cap<base_channels:
            raise ValueError('base_channels >=2 and channel_cap >= base_channels are required')
        self.base_channels=int(base_channels);self.channel_cap=int(channel_cap)
        channels=[min(self.channel_cap,self.base_channels*(2**index))
                  for index in range(self.depth+1)]
        self.channels=channels
        encoders=[_ConvBlock(4,channels[0])]
        encoders.extend(_ConvBlock(channels[index-1],channels[index])
                        for index in range(1,self.depth))
        self.encoders=nn.ModuleList(encoders)
        self.bridge=nn.Sequential(
            _ConvBlock(channels[self.depth-1],channels[self.depth]),
            nn.Conv2d(channels[self.depth],channels[self.depth],3,padding=1,bias=False),
            nn.GroupNorm(1,channels[self.depth]),nn.SiLU(inplace=True))
        current=channels[self.depth];decoders=[]
        for skip_channels in reversed(channels[:self.depth]):
            decoders.append(_ConvBlock(current+skip_channels,skip_channels))
            current=skip_channels
        self.decoders=nn.ModuleList(decoders)
        self.body_head=nn.Conv2d(channels[0],1,1)
        self.center_head=nn.Conv2d(channels[0],1,1)

    def forward(self,value):
        if value.ndim!=4 or value.shape[1]!=4:
            raise ValueError('Expected N x 4 x H x W input (RGB + seed heatmap)')
        features=[];current=value
        for index,encoder in enumerate(self.encoders):
            if index:current=F.max_pool2d(current,2,ceil_mode=True)
            current=encoder(current);features.append(current)
        current=self.bridge(F.max_pool2d(current,2,ceil_mode=True))
        for decoder,skip in zip(self.decoders,reversed(features)):
            current=F.interpolate(current,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            current=decoder(torch.cat([current,skip],dim=1))
        return {'body_logits':self.body_head(current),
                'center_logits':self.center_head(current)}


def architecture_metadata(architecture,base_channels=8,channel_cap=64):
    if architecture=='shallow':
        return {'name':'shallow','pool_depth':2,'base_channels':int(base_channels),
                'channel_cap':None,'theoretical_structured_receptive_field_px':50,
                'pretrained_weights':None,
                'receptive_field_claim':'calculated dependency extent; not evidence of learned global reasoning'}
    if architecture=='context':
        return {'name':'context','pool_depth':6,'base_channels':int(base_channels),
                'channel_cap':int(channel_cap),'theoretical_structured_receptive_field_px':1078,
                'pretrained_weights':None,
                'receptive_field_claim':'calculated dependency extent; not evidence of learned global reasoning'}
    raise ValueError(f'architecture must be one of {ARCHITECTURES}')


def build_model(architecture,base_channels=8,channel_cap=64):
    architecture_metadata(architecture,base_channels,channel_cap)
    if architecture=='shallow':return TinyCenterlineUNet(int(base_channels))
    return ContextCenterlineUNet(int(base_channels),int(channel_cap))


def _batch_losses(model,batch,center_weight,forward_timer=None):
    started=time.perf_counter();outputs=model(batch['input'])
    if forward_timer is not None:forward_timer['seconds']+=time.perf_counter()-started
    losses=masked_two_head_loss(outputs,batch['body_target'],batch['center_target'],None,
                                center_weight,batch['body_valid_mask'],batch['center_valid_mask'])
    return outputs,losses


def _pixel_metrics(model,loader,center_weight,interpretation):
    totals=dict(body_loss_sum=0.,center_loss_sum=0.,body_intersection=0.,body_sum=0.,
                center_intersection=0.,center_sum=0.,body_valid_pixels=0.,center_valid_pixels=0.)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs,losses=_batch_losses(model,batch,center_weight)
            body_valid=batch['body_valid_mask']>0;center_valid=batch['center_valid_mask']>0
            body_prediction=(torch.sigmoid(outputs['body_logits'])>=.5)&body_valid
            center_prediction=(torch.sigmoid(outputs['center_logits'])>=.5)&center_valid
            body_truth=(batch['body_target']>=.5)&body_valid
            center_truth=(batch['center_target']>=.5)&center_valid
            body_weight=float(losses['body_valid_weight'])
            center_weight_pixels=float(losses['center_valid_weight'])
            totals['body_loss_sum']+=float(losses['body_loss'])*body_weight
            totals['center_loss_sum']+=float(losses['center_loss'])*center_weight_pixels
            totals['body_intersection']+=float((body_prediction&body_truth).sum())
            totals['body_sum']+=float(body_prediction.sum()+body_truth.sum())
            totals['center_intersection']+=float((center_prediction&center_truth).sum())
            totals['center_sum']+=float(center_prediction.sum()+center_truth.sum())
            totals['body_valid_pixels']+=body_weight
            totals['center_valid_pixels']+=center_weight_pixels
    def ratio(numerator,denominator):return None if denominator<=0 else numerator/denominator
    body_loss=ratio(totals['body_loss_sum'],totals['body_valid_pixels'])
    center_loss=ratio(totals['center_loss_sum'],totals['center_valid_pixels'])
    combined=(None if body_loss is None and center_loss is None else
              (0. if body_loss is None else body_loss)+
              float(center_weight)*(0. if center_loss is None else center_loss))
    return {'masked_two_head_loss':combined,'body_masked_loss':body_loss,
            'center_masked_loss':center_loss,
            'body_dice_at_0_5':ratio(2*totals['body_intersection'],totals['body_sum']),
            'center_dice_at_0_5':ratio(2*totals['center_intersection'],totals['center_sum']),
            'body_valid_pixels':int(totals['body_valid_pixels']),
            'center_valid_pixels':int(totals['center_valid_pixels']),
            'interpretation':interpretation}


def _supervision_summary(dataset):
    summary={'body_valid_sample_count':0,'center_valid_sample_count':0,
             'body_positive_sample_count':0,'center_positive_sample_count':0}
    for index in range(len(dataset)):
        sample=dataset[index]
        body_valid=sample['body_valid_mask'].numpy()>0
        center_valid=sample['center_valid_mask'].numpy()>0
        body_positive=bool(((sample['body_target'].numpy()>0)*body_valid).any())
        center_positive=bool(((sample['center_target'].numpy()>0)*center_valid).any())
        summary['body_valid_sample_count']+=int(body_valid.any())
        summary['center_valid_sample_count']+=int(center_valid.any())
        summary['body_positive_sample_count']+=int(body_positive)
        summary['center_positive_sample_count']+=int(center_positive)
    summary['body_head_supervised']=summary['body_positive_sample_count']>0
    summary['center_head_supervised']=summary['center_positive_sample_count']>0
    summary['supervision_complete']=summary['body_head_supervised'] and summary['center_head_supervised']
    return summary


def train_manifest(args):
    command_started=time.perf_counter();torch.set_num_threads(int(args.threads))
    torch.manual_seed(int(args.seed));target_hw=(int(args.target_height),int(args.target_width))
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    inspected=inspect_manifest(args.manifest)
    train_records=inspected['usable']['train'];holdout_records=inspected['usable']['holdout']
    if not train_records:
        report={'schema_version':1,'status':'blocked_no_reviewed_labels',
                'manifest':str(Path(args.manifest).resolve()),
                'manifest_sha256':inspected['manifest_sha256'],
                'skipped_samples':inspected['skipped'],'artifacts':{'checkpoint':None},
                'claim_limits':{'field_accuracy_established':False,'production_ready':False}}
        _write_json(output/'report.json',report);return report
    cache_started=time.perf_counter()
    training=ManifestLetterboxDataset(train_records,target_hw,args.cache_samples)
    holdout=ManifestLetterboxDataset(holdout_records,target_hw,args.cache_samples)
    cache_preparation_seconds=time.perf_counter()-cache_started
    supervision=_supervision_summary(training)
    if not supervision['body_head_supervised'] and not supervision['center_head_supervised']:
        report={'schema_version':1,'status':'blocked_no_positive_target_supervision',
                'manifest':str(Path(args.manifest).resolve()),'supervision':supervision,
                'artifacts':{'checkpoint':None},
                'claim_limits':{'field_accuracy_established':False,'production_ready':False}}
        _write_json(output/'report.json',report);return report
    loader=DataLoader(training,batch_size=args.batch_size,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(int(args.seed)+31))
    train_metric_loader=DataLoader(training,batch_size=args.batch_size,shuffle=False,num_workers=0)
    holdout_loader=(DataLoader(holdout,batch_size=args.batch_size,shuffle=False,num_workers=0)
                    if len(holdout) else None)
    model=build_model(args.architecture,args.base_channels,args.channel_cap).cpu()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    history=[];iterator=iter(loader);model.train();loop_started=time.perf_counter()
    forward_timer={'seconds':0.};batch_fetch_seconds=0.
    for step in range(int(args.steps)):
        fetch_started=time.perf_counter()
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
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
    loop_seconds=time.perf_counter()-loop_started
    train_metrics=_pixel_metrics(model,train_metric_loader,args.center_weight,
        'post-training train split diagnostic; not an independent accuracy estimate')
    holdout_metrics=(_pixel_metrics(model,holdout_loader,args.center_weight,
        'group-held-out diagnostic; not used for gradient updates or early stopping')
        if holdout_loader is not None else None)
    architecture=architecture_metadata(args.architecture,args.base_channels,args.channel_cap)
    checkpoint_path=output/'checkpoint.pt';temporary=output/'checkpoint.pt.tmp'
    checkpoint={'format':CHECKPOINT_FORMAT,'schema_version':1,
                'preprocess_version':PREPROCESS_VERSION,
                'raster_encoding_version':RASTER_ENCODING_VERSION,
                'line_map_encoding':'nonempty Gaussian line normalized to peak 1 after letterbox content resize; empty stays zero',
                'model_config':{'architecture':args.architecture,'input_channels':4,
                                'base_channels':int(args.base_channels),
                                'channel_cap':int(args.channel_cap),'heads':list(OUTPUT_HEADS),
                                'input_size_hw':list(target_hw)},
                'architecture_metadata':architecture,'state_dict':model.state_dict(),
                'input_semantics':list(INPUT_SEMANTICS),
                'output_semantics':{'body':'target rope-body probability',
                                    'center':'target rope-centre heatmap'},
                'preprocess':{'name':'native_roi_fit_pad','padding_supervision':'invalid',
                              'probability_export':'crop model padding; content spans native ROI'},
                'training_source':'reviewed_manifest_annotations',
                'training_manifest_sha256':inspected['manifest_sha256'],
                'training_annotation_sha256_by_sample':{
                    record['sample_id']:record['annotation_sha256'] for record in train_records},
                'train_quality_sha256_by_sample':{
                    record['sample_id']:record['quality_sha256'] for record in train_records},
                'train_group_ids':sorted({record['group_id'] for record in train_records}),
                'body_head_supervised':supervision['body_head_supervised'],
                'center_head_supervised':supervision['center_head_supervised'],
                'body_valid_sample_count':supervision['body_valid_sample_count'],
                'center_valid_sample_count':supervision['center_valid_sample_count'],
                'body_positive_sample_count':supervision['body_positive_sample_count'],
                'center_positive_sample_count':supervision['center_positive_sample_count'],
                'field_validation_performed':False}
    torch.save(checkpoint,temporary);os.replace(temporary,checkpoint_path)
    report={'schema_version':1,'status':('manifest_training_complete_unvalidated'
                if supervision['supervision_complete'] else 'partial_supervision'),
            'manifest':str(Path(args.manifest).resolve()),
            'manifest_sha256':inspected['manifest_sha256'],
            'preprocess_version':PREPROCESS_VERSION,
            'raster_encoding_version':RASTER_ENCODING_VERSION,
            'data':{'split_source':'manifest group split; no random neighbouring-frame split',
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
                    'skipped_samples':inspected['skipped'],'augmentation':None,
                    'holdout_used_for_training':False,'holdout_used_for_early_stopping':False},
            'architecture':architecture,'supervision':supervision,
            'training':{'device':'cpu','steps_completed':int(args.steps),
                        'loss':'masked per-head binary cross entropy; unchanged from V5',
                        'loop_seconds':loop_seconds,'model_forward_seconds':forward_timer['seconds'],
                        'batch_fetch_seconds':batch_fetch_seconds,
                        'cache_samples':bool(args.cache_samples),
                        'cache_preparation_seconds':cache_preparation_seconds,
                        'cache_bytes':training.cache_bytes()+holdout.cache_bytes(),
                        'model_input_hw':list(target_hw),'base_channels':int(args.base_channels),
                        'channel_cap':int(args.channel_cap),'batch_size':int(args.batch_size),
                        'learning_rate':float(args.learning_rate),'center_weight':float(args.center_weight),
                        'seed':int(args.seed),'threads':int(args.threads),'loss_history':history},
            'train_pixel_diagnostic':train_metrics,'holdout_pixel_diagnostic':holdout_metrics,
            'artifacts':{'checkpoint':str(checkpoint_path),
                         'checkpoint_sha256':_file_sha256(checkpoint_path)},
            'command_elapsed_before_report_seconds':time.perf_counter()-command_started,
            'claim_limits':{'field_accuracy_established':False,'production_ready':False,
                            'fully_automatic':False,'seed_assisted':True,
                            'receptive_field_is_capability_proof':False}}
    _write_json(output/'report.json',report)
    print(json.dumps({'report':str(output/'report.json'),'checkpoint':str(checkpoint_path),
                      'status':report['status'],'loop_seconds':loop_seconds}))
    return report


def load_model(checkpoint_path):
    checkpoint=torch.load(Path(checkpoint_path),map_location='cpu',weights_only=True)
    if checkpoint.get('format')!=CHECKPOINT_FORMAT:
        raise ValueError('Incompatible checkpoint: V6 requires a native fit+pad checkpoint; V5 square-resize checkpoints are rejected')
    if checkpoint.get('preprocess_version')!=PREPROCESS_VERSION:
        raise ValueError('Incompatible V6 preprocess_version')
    if checkpoint.get('raster_encoding_version')!=RASTER_ENCODING_VERSION:
        raise ValueError('Incompatible line-map raster encoding')
    config=checkpoint.get('model_config',{})
    if config.get('input_channels')!=4 or config.get('heads')!=list(OUTPUT_HEADS):
        raise ValueError('Checkpoint input/head semantics mismatch')
    target_hw=_positive_hw(config.get('input_size_hw'),'checkpoint model_config.input_size_hw')
    model=build_model(config.get('architecture'),int(config.get('base_channels',0)),
                      int(config.get('channel_cap',0))).cpu()
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    return model,checkpoint,target_hw


def _predict(model,input_tensor):
    started=time.perf_counter()
    with torch.inference_mode():
        outputs=model(input_tensor.unsqueeze(0))
        body=torch.sigmoid(outputs['body_logits'])[0,0].numpy().astype(np.float32,copy=False)
        center=torch.sigmoid(outputs['center_logits'])[0,0].numpy().astype(np.float32,copy=False)
    return body,center,(time.perf_counter()-started)*1000.


def _write_prediction(path,body,center):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,body_probability=body.astype(np.float32,copy=False),
                        center_probability=center.astype(np.float32,copy=False))
    return path


def infer_manifest(args):
    torch.set_num_threads(int(args.threads));model,checkpoint,target_hw=load_model(args.checkpoint)
    inspected=inspect_manifest(args.manifest);output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    records=[record for record in inspected['inference_records']
             if args.split=='all' or record['split']==args.split]
    checkpoint_hash=_file_sha256(args.checkpoint);exported=[]
    training_source=checkpoint.get('training_source','unknown')
    for record in records:
        started=time.perf_counter();input_tensor,geometry,rgb,seed=letterbox_manifest_input(record,target_hw)
        input_path=output/'inputs'/f"{record['sample_id']}.npz";input_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(input_path,rgb=rgb.astype(np.float32,copy=False),
                            seed_heatmap=seed.astype(np.float32,copy=False))
        body_padded,center_padded,inference_ms=_predict(model,input_tensor)
        body=crop_letterbox_probabilities(body_padded,geometry)
        center=crop_letterbox_probabilities(center_padded,geometry)
        prediction_path=_write_prediction(output/'predictions'/f"{record['sample_id']}.npz",body,center)
        metadata={'schema_version':1,'raster_encoding_version':RASTER_ENCODING_VERSION,
                  'preprocess_version':PREPROCESS_VERSION,'status':'unvalidated_model_inference',
                  'sample_id':record['sample_id'],'group_id':record['group_id'],
                  'manifest_split':record['split'],'image_sha256':record['image_sha256'],
                  'annotation_sha256':record['annotation_sha256'],
                  'free_seed_sha256':record['free_seed_sha256'],
                  'quality_sha256':record['quality_sha256'],'roi_xyxy':record['roi_xyxy'],
                  'native_roi_hw':[int(rgb.shape[0]),int(rgb.shape[1])],
                  'model_input_hw':list(target_hw),'probability_map_hw':list(body.shape),
                  'scale_xy':geometry['scale_xy'],'letterbox':geometry,
                  'probability_arrays':{'body_probability':'unpadded aspect-preserving model content spanning native ROI',
                                        'center_probability':'unpadded aspect-preserving model content spanning native ROI'},
                  'input_npz':str(input_path),'input_sha256':_file_sha256(input_path),
                  'output_npz':str(prediction_path),'output_sha256':_file_sha256(prediction_path),
                  'checkpoint':str(Path(args.checkpoint)),'checkpoint_sha256':checkpoint_hash,
                  'inference_ms':inference_ms,'total_elapsed_seconds':time.perf_counter()-started,
                  'architecture':checkpoint['model_config']['architecture'],
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
                  'field_accuracy_established':False,'hardware_access':False,'force_command':None}
        metadata_path=prediction_path.with_suffix('.json');_write_json(metadata_path,metadata)
        exported.append({'sample_id':record['sample_id'],'split':record['split'],
                         'prediction':str(prediction_path),'metadata':str(metadata_path)})
    selected=[sample for sample in inspected['manifest']['samples']
              if args.split=='all' or sample['split']==args.split]
    exported_ids={row['sample_id'] for row in exported}
    reason_by_id={row['sample_id']:row['reason'] for row in inspected['skipped']}
    skipped=[{'sample_id':sample['sample_id'],'split':sample['split'],
              'reason':reason_by_id.get(sample['sample_id'],'no_reviewed_free_seed')}
             for sample in selected if sample['sample_id'] not in exported_ids]
    summary={'schema_version':1,'status':('complete' if exported else 'blocked_no_reviewed_free_seed'),
             'raster_encoding_version':RASTER_ENCODING_VERSION,
             'preprocess_version':PREPROCESS_VERSION,
             'manifest':str(Path(args.manifest).resolve()),
             'manifest_sha256':inspected['manifest_sha256'],'requested_split':args.split,
             'checkpoint':str(Path(args.checkpoint)),'checkpoint_sha256':checkpoint_hash,
             'checkpoint_training_source':training_source,
             'architecture':checkpoint['model_config']['architecture'],
             'model_input_hw':list(target_hw),'exported':exported,'skipped':skipped,
             'ground_truth_center_or_edges_used_as_input':False,
             'hardware_access':False,'force_command':None}
    _write_json(output/'summary.json',summary)
    print(json.dumps({'summary':str(output/'summary.json'),'exported':len(exported),
                      'skipped':len(skipped),'status':summary['status']}))
    return summary


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    train=commands.add_parser('train-manifest',help='Train one reviewed aspect-preserving offline model')
    train.add_argument('--manifest',required=True);train.add_argument('--output-dir',default=str(DEFAULT_OUTPUT))
    train.add_argument('--architecture',choices=ARCHITECTURES,default='context')
    train.add_argument('--steps',type=int,default=60)
    train.add_argument('--target-height',type=int,default=DEFAULT_TARGET_HW[0])
    train.add_argument('--target-width',type=int,default=DEFAULT_TARGET_HW[1])
    train.add_argument('--base-channels',type=int,default=8);train.add_argument('--channel-cap',type=int,default=64)
    train.add_argument('--batch-size',type=int,default=2);train.add_argument('--learning-rate',type=float,default=1e-3)
    train.add_argument('--center-weight',type=float,default=1.)
    train.add_argument('--seed',type=int,default=20260926);train.add_argument('--threads',type=int,default=2)
    train.add_argument('--cache-samples',action='store_true',
                       help='Cache exact deterministic letterbox tensors; no augmentation is introduced')
    train.set_defaults(action=train_manifest)
    infer=commands.add_parser('infer-manifest',help='Export reviewed human-seed cases using a V6 checkpoint')
    infer.add_argument('--manifest',required=True);infer.add_argument('--checkpoint',required=True)
    infer.add_argument('--output-dir',required=True)
    infer.add_argument('--split',choices=('all','train','holdout'),default='all')
    infer.add_argument('--threads',type=int,default=2);infer.set_defaults(action=infer_manifest)
    return parser


def main(argv=None):
    args=build_parser().parse_args(argv)
    for name in ('steps','target_height','target_width','base_channels','channel_cap',
                 'batch_size','threads'):
        if hasattr(args,name) and getattr(args,name)<=0:raise ValueError(f'{name} must be positive')
    if hasattr(args,'learning_rate') and (not math.isfinite(args.learning_rate) or args.learning_rate<=0):
        raise ValueError('learning_rate must be finite and positive')
    if hasattr(args,'center_weight') and (not math.isfinite(args.center_weight) or args.center_weight<0):
        raise ValueError('center_weight must be finite and non-negative')
    return args.action(args)


if __name__=='__main__':main()
