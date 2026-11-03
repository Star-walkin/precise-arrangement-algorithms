"""Fixed two-image digital-photometry response diagnostic; no target labels.

Six original + eighteen perturbed forwards. No model selection, no H6 path
input, no response-derived ROI, and no real-illumination/identity assertion.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from common import HERE,load,sha
from network import load_network
from preprocessing import prepare_input

PROTOCOL=HERE/'photometry_diagnostic_protocol.json'
CONDITIONS=('original','global_gain_0_8','gamma_1_25','horizontal_gain_ramp_0_6_to_1_4')
ARMS=('B','C','D')


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def perturb(gray,condition):
    assert gray.dtype==np.float32 and gray.shape==(975,2048)
    assert np.isfinite(gray).all() and 0<=gray.min()<=gray.max()<=1
    if condition=='original':before=gray.copy()
    elif condition=='global_gain_0_8':before=gray*np.float32(.8)
    elif condition=='gamma_1_25':before=np.power(gray,np.float32(1.25))
    elif condition=='horizontal_gain_ramp_0_6_to_1_4':
        before=gray*np.linspace(.6,1.4,gray.shape[1],dtype=np.float32)[None]
    else:raise ValueError(condition)
    result=np.clip(before,0,1).astype(np.float32)
    return result,{'condition':condition,'clip_low_fraction':float((before<0).mean()),
        'clip_high_fraction':float((before>1).mean()),'mean':float(result.mean()),'std':float(result.std()),
        'min':float(result.min()),'max':float(result.max()),'true_zero_pixels':int((result==0).sum())}


def prepare():
    main=load(HERE/'protocol.json');manifest=load(HERE/'inputs/manifest.json')
    ids=main['user_cases'];assert len(ids)==2
    jobs={j['id']:j for j in manifest['batch'] if j['id'] in ids};assert set(jobs)==set(ids)
    assert all(len(j['seeds'])==1 for j in jobs.values())
    sources={str(HERE/name):sha(HERE/name) for name in ('photometry_diagnostic.py','network.py','preprocessing.py','common.py')}
    value={'created_utc':datetime.now(timezone.utc).isoformat(),'user_cases':ids,'arms':list(ARMS),
        'conditions':list(CONDITIONS),'maximum_forward_calls':24,'original_forward_calls':6,'perturbed_forward_calls':18,
        'entry_index':0,'entry_policy':'same existing automatic short entry for original and every perturbation; no additional path hints',
        'original_input':'frozen nonnegative float grayscale image in original search ROI, resized once to975x2048',
        'perturbation_order':'perturb raw gray in[0,1], clip[0,1], recompute arm-specific full-ROI standardization/LCN, neutral pad, forward',
        'formula':{'global_gain_0_8':'clip(.8*I,0,1)','gamma_1_25':'clip(I**1.25,0,1)',
                   'horizontal_gain_ramp_0_6_to_1_4':'clip(I*linspace(.6,1.4,2048)[None,:],0,1); left-to-right original x axis'},
        'domains':{'full_search_rectangle_original_xy':[0,1050,4096,3000],
                   'separate_spool_domain':None,'reason':'No response-derived mask or H6 path; this bounded diagnostic uses the full fixed search domain only.'},
        'mapping':'model_xy=(original_xy-[0,1050])*.5','fixed_gray_scale':[0,1],'fixed_response_scale':[0,1],
        'heat_lut':{'path':str(HERE/'inputs/turbo_rgb.npy'),'sha256':sha(HERE/'inputs/turbo_rgb.npy')},
        'input_manifest_sha256':sha(HERE/'inputs/manifest.json'),'main_protocol_sha256':sha(HERE/'protocol.json'),
        'inputs':[{'id':sid,'input_cache':jobs[sid]['input_cache'],'input_cache_sha256':jobs[sid]['input_cache_sha256'],
                   'image_sha256':jobs[sid]['image_sha256'],'seed_xy':jobs[sid]['seeds'][0]['xy']} for sid in ids],
        'sources':sources,'labels_read':False,'used_for_model_selection':False,'H6_paths_used':False,
        'interpretation':'Digital input sensitivity only. Fixed global normalization can cancel uniform gain. Not real lighting, shadow geometry, SNR, or target identity accuracy.'}
    write_new(PROTOCOL,value)
    print(json.dumps({'protocol':str(PROTOCOL),'sha256':sha(PROTOCOL),'maximum_forward_calls':24}))


def distribution(a):
    q=np.quantile(a,[.01,.1,.5,.9,.95,.99])
    return {'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'max':float(a.max()),
            'quantiles_p01_p10_p50_p90_p95_p99':[float(x) for x in q],
            'fraction_lt_0_2':float((a<.2).mean()),'fraction_gt_0_5':float((a>.5).mean()),
            'fraction_gt_0_8':float((a>.8).mean())}


def comparison_metric(response,original):
    difference=response-original;absolute=np.abs(difference)
    return {'mean_signed_change':float(difference.mean()),'mean_absolute_change':float(absolute.mean()),
            'p95_absolute_change':float(np.quantile(absolute,.95)),'maximum_absolute_change':float(absolute.max()),
            'fraction_absolute_change_gt_0_1':float((absolute>.1).mean()),
            'pearson_response_original':float(np.corrcoef(response.ravel(),original.ravel())[0,1])
                if response.std()>1.e-8 and original.std()>1.e-8 else None}


def font(size):return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)


def render(path,sid,images,responses,statistics,lut):
    pw,ph=640,305;gap=6;top=120;rh=365
    canvas=Image.new('RGB',(4*pw+3*gap,top+4*rh+85),'white');draw=ImageDraw.Draw(canvas)
    draw.text((20,15),'Fixed digital-photometry diagnostic | same entry | no model selection',font=font(27),fill='black')
    draw.text((20,51),sid,font=font(23),fill='black')
    draw.text((20,85),'Full original search ROI: x[0,4096), y[1050,3000); raw gray and response fixed0..1; no target reference',font=font(19),fill='black')
    names=['Original','Gain0.8','Gamma1.25','Horizontal gain0.6 to1.4']
    for column,condition in enumerate(CONDITIONS):
        x=column*(pw+gap)
        gray=np.uint8(np.rint(np.clip(images[condition],0,1)*255))
        image=Image.fromarray(gray).convert('RGB').resize((pw,ph),Image.Resampling.BILINEAR)
        draw.text((x+8,top),names[column]+' | input gray',font=font(21),fill='black')
        canvas.paste(image,(x,top+34))
        for row,arm in enumerate(ARMS,1):
            y=top+row*rh
            response=responses[arm][condition]
            heat=Image.fromarray(lut[np.rint(np.clip(response,0,1)*255).astype(np.uint8)]).resize((pw,ph),Image.Resampling.BILINEAR)
            st=statistics[arm][condition]
            delta=st['change_from_same_arm_original']['mean_absolute_change']
            draw.text((x+8,y),f'{arm} | mean {st["response"]["mean"]:.3f} | mean abs change {delta:.3f}',font=font(20),fill='black')
            canvas.paste(heat,(x,y+34))
    y=top+4*rh+8;bar=Image.fromarray(lut[np.linspace(0,255,900).astype(np.uint8)][None].repeat(20,axis=0))
    canvas.paste(bar,(80,y));draw.text((20,y-2),'0.0',font=font(19),fill='black');draw.text((990,y-2),'1.0 response strength',font=font(19),fill='black')
    draw.text((20,y+36),'Digital perturbation stability is not real-lighting robustness or correct rope identity; no per-panel min-max stretching.',font=font(20),fill='black')
    canvas.save(path)
    with Image.open(path) as check:assert check.size==canvas.size;check.verify()


@torch.inference_mode()
def run(threads):
    protocol=load(PROTOCOL);choice=load(HERE/'selection.json')
    assert protocol['maximum_forward_calls']==24 and not protocol['used_for_model_selection']
    assert sha(HERE/'inputs/manifest.json')==protocol['input_manifest_sha256']
    assert sha(HERE/'protocol.json')==protocol['main_protocol_sha256']
    assert all(sha(p)==h for p,h in (protocol['sources']|choice['source_hashes']).items())
    assert sha(protocol['heat_lut']['path'])==protocol['heat_lut']['sha256']
    lut=np.load(protocol['heat_lut']['path'],allow_pickle=False);assert lut.shape==(256,3) and lut.dtype==np.uint8
    output=HERE/'photometry_diagnostic';output.mkdir(exist_ok=False);(output/'comparisons').mkdir()
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    write_new(output/'run_start.json',{'protocol_sha256':sha(PROTOCOL),'selection_sha256':sha(HERE/'selection.json'),
        'models':choice['models'],'sources':protocol['sources'],'threads':threads,'device':'cpu','command':sys.argv})
    inputs={};seeds={};input_stats={};files=[]
    for item in protocol['inputs']:
        assert sha(item['input_cache'])==item['input_cache_sha256']
        with np.load(item['input_cache'],allow_pickle=False) as z:gray=z['image'];seed=z['seeds'][0].copy()
        sid=item['id'];inputs[sid]={};input_stats[sid]={};seeds[sid]=seed
        for condition in CONDITIONS:inputs[sid][condition],input_stats[sid][condition]=perturb(gray,condition)
        path=output/'inputs'/(sid+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,seed=seed,**inputs[sid]);files.append({'path':str(path),'sha256':sha(path)})
    all_responses={sid:{a:{} for a in ARMS} for sid in inputs};all_stats={sid:{a:{} for a in ARMS} for sid in inputs}
    count=0;tick=time.time()
    for arm in ARMS:
        model=choice['models'][arm];assert sha(model['path'])==model['sha256'];network=load_network(model['path'])
        cfg={**network.config,'variant':'D' if arm=='D' else 'C'}
        for item in protocol['inputs']:
            sid=item['id'];original=None
            for condition in CONDITIONS:
                x,prep=prepare_input(inputs[sid][condition],seeds[sid],cfg)
                response=network(x)[0,0,:975,:2048].numpy().astype(np.float32);count+=1
                assert np.isfinite(response).all() and 0<=response.min()<=response.max()<=1
                if condition=='original':original=response.copy()
                all_responses[sid][arm][condition]=response
                stat={'response':distribution(response),'change_from_same_arm_original':comparison_metric(response,original),
                      'raw_input':input_stats[sid][condition],'model_input':prep}
                all_stats[sid][arm][condition]=stat
                folder=output/'responses'/arm/sid/condition;folder.mkdir(parents=True)
                path=folder/'response.npz';np.savez_compressed(path,response=response)
                meta={'id':sid,'arm':arm,'condition':condition,'model':model,'image_sha256':item['image_sha256'],
                      'seed_xy':item['seed_xy'],'input_cache_sha256':item['input_cache_sha256'],
                      'response_path':str(path),'response_sha256':sha(path),'mapping':protocol['mapping'],
                      'shape':[975,2048],'statistics':stat,'only_digital_perturbation':True,'labels_read':False}
                write_new(folder/'inference.json',meta)
                files.extend({'path':str(p),'sha256':sha(p)} for p in (path,folder/'inference.json'))
        print(json.dumps({'model':arm,'completed_forwards':count,'maximum':24,'seconds':round(time.time()-tick,2)}),flush=True)
    assert count==24
    for sid in inputs:
        path=output/'comparisons'/(sid+'.png')
        render(path,sid,inputs[sid],all_responses[sid],all_stats[sid],lut)
        files.append({'path':str(path),'sha256':sha(path)})
    summary={'exit_code':0,'created_utc':datetime.now(timezone.utc).isoformat(),'forward_calls':count,
        'original_forwards':6,'perturbed_forwards':18,'seconds':time.time()-tick,
        'protocol_sha256':sha(PROTOCOL),'selection_sha256':sha(HERE/'selection.json'),
        'source_unchanged':all(sha(p)==h for p,h in protocol['sources'].items()),
        'model_selection_altered':False,'domains':protocol['domains'],'rows':all_stats,
        'limits':protocol['interpretation'],'files':files}
    write_new(output/'summary.json',summary)
    print(json.dumps({'exit_code':0,'forward_calls':count,'summary':str(output/'summary.json')}))


def checks():
    a=np.broadcast_to(np.linspace(0,1,2048,dtype=np.float32),(975,2048)).copy()
    b,st=perturb(a,'global_gain_0_8');assert np.allclose(b,.8*a) and st['clip_high_fraction']==0
    b,st=perturb(a,'gamma_1_25');assert np.isfinite(b).all() and np.all(b[:,0]==0)
    b,st=perturb(a,'horizontal_gain_ramp_0_6_to_1_4');assert st['clip_high_fraction']>0 and b.max()==1
    assert np.array_equal(perturb(a,'original')[0],a)
    assert comparison_metric(a,a)['mean_absolute_change']==0
    print(json.dumps({'passed':True,'checks':5,'no_images_or_models_read':True}))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','check']);p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='check':checks()
    else:run(a.threads)
