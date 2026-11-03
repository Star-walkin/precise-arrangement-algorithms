"""H4 fixed numeric checks and the frozen development13, without inference."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from common import HERE,H3,inputs,surface,load,sha
from heat_free import describe_free,_evaluate_axis,PARAMETERS


def numeric_checks():
    h,w=250,400;yy,xx=np.mgrid[:h,:w]
    a=np.deg2rad(1.5);center=103.+np.tan(a)*(xx-20)
    gray=np.full((h,w),.5,np.float32);gray[abs(yy-center)<4]=0.
    heat=(.8*np.exp(-.5*((yy-center)/3.)**2)).astype(np.float32);heat[:,300:]=0
    image=np.full((775,w,3),127,np.uint8)
    image[525:]=np.repeat(np.rint(gray*255).astype(np.uint8)[...,None],3,axis=2)
    a7=dict(candidate_id='synthetic_initial_axis',accepted=True,plausible=True,
        origin_xy=[300.,1250.],direction_xy=[1.,0.],width_px=20.,anchor_span_px=160.)
    priors=dict(original_half=image,a7_candidates=[a7],a6_entries=[],pitch_px=40.,original_shape=[1550,800])
    seed=dict(xy=[[40.,1250.],[160.,1250.]],source='synthetic',entry_id='synthetic')
    old_seed=copy.deepcopy(seed);old_heat=heat.copy()
    result=describe_free(priors,seed,heat)
    assert result['available'] and len(result['points_xy'])>70
    assert result['diagnostics']['selected_axis']['span_original_px']>400
    assert seed==old_seed and np.array_equal(heat,old_heat)
    assert abs(result['diagnostics']['selected_axis']['angle_degrees']-1.5)<=.4
    assert result['diagnostics']['axis_candidates']<=180
    n=len(result['points_xy']);assert len(result['segment_status'])==n-1
    assert len(result['supported'])==len(result['protect_support'])==n
    assert all(len(v)==n for v in result['evidence'].values())
    assert np.all(np.asarray(result['supported'])<=np.asarray(result['protect_support']))
    assert not describe_free(priors,seed,np.full_like(heat,.8))['available']
    gap_heat=heat.copy();gap_heat[:,120:150]=np.nan
    gap=describe_free(priors,seed,gap_heat)
    xy=np.asarray(gap['points_xy']);inside=(xy[:,0]>=120)&(xy[:,0]<150)
    assert not np.asarray(gap['protect_support'])[inside].any()
    assert not np.asarray(gap['supported'])[inside].any()
    # A shared denominator must not reward a short strong prefix over an
    # otherwise comparable much longer continuously supported ridge.
    g=np.full((h,w),.5,np.float32);g[abs(yy-100)<4]=0
    long=(.6*np.exp(-.5*((yy-100)/3.)**2)).astype(np.float32);long[:,300:]=0
    short=(.85*np.exp(-.5*((yy-100)/3.)**2)).astype(np.float32);short[:,80:]=0
    valid=np.ones((h,w),np.float32);origin=np.array([20.,100.]);d=np.array([1.,0.])
    sr=_evaluate_axis(g,short,valid,origin,d,10.,60.,0.,0.,3.)
    lr=_evaluate_axis(g,long,valid,origin,d,10.,60.,0.,0.,3.)
    assert lr['score']>sr['score']
    assert lr['common_view_original_px']==sr['common_view_original_px']==2000.
    assert result['diagnostics']['old_endpoint_used'] is False
    return dict(passed=True,checks=['unchanged_seed_and_response','finite_angle_offset_reestimation',
        'beyond_initial_short_hint_extent','180_axis_budget','aligned_per_point_arrays',
        'raw_support_subset_of_heat_protection','constant_heat_rejected','NaN_gap_not_observed',
        'common_2000px_integral_does_not_favor_short_prefix'],
        short_score=sr['score'],long_score=lr['score'])


def draw_line(image,points,states):
    p=np.asarray(points,float)
    if len(p)<2:return image
    out=image.copy();arc=0.
    for i in range(1,len(p)):
        status=states[i] if i<len(states) else 'unknown'
        color=(40,230,40) if status in ('image_supported','joint_heat_image_supported','a7_image_supported') else (0,225,255) if status=='heatmap_supported' else (255,255,255) if status=='entry_condition' else (0,140,255)
        distance=float(np.linalg.norm(p[i]-p[i-1]))
        if status not in ('ambiguous_gap','unknown') or int(arc/7)%2==0:
            cv2.line(out,tuple(np.rint(p[i-1]).astype(int)),tuple(np.rint(p[i]).astype(int)),color,2,cv2.LINE_AA)
        arc+=distance
    return out


def preview(path,priors,heat,old,new,identifier,index):
    base=priors['original_half'][525:].copy()
    h,w=base.shape[:2]
    all_points=np.concatenate([np.asarray(p,float).reshape(-1,2) for p in (old.get('points_xy',[]),new.get('points_xy',[]))])
    if not len(all_points):return
    lo=np.maximum(np.floor(all_points.min(0)-[60,90]).astype(int),[0,0]);hi=np.minimum(np.ceil(all_points.max(0)+[60,90]).astype(int),[w,h])
    x0,y0=lo;x1,y1=hi
    old_img=draw_line(base,old.get('points_xy',[]),old.get('point_status',[]))
    new_img=draw_line(base,new.get('points_xy',[]),new.get('point_status',[]))
    heat_img=cv2.applyColorMap(np.rint(np.clip(heat,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
    panels=[]
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
    for title,im in [('Original',base),('Frozen B response (0..1)',heat_img),('H3 free result',old_img),('H4 free | green=joint; yellow=heat only',new_img)]:
        tile=im[y0:y1,x0:x1];scale=800/tile.shape[1];height=max(120,int(tile.shape[0]*scale))
        tile=cv2.resize(tile,(800,height),interpolation=cv2.INTER_AREA)
        board=Image.new('RGB',(800,height+90),(245,247,250));board.paste(Image.fromarray(tile[:,:,::-1]),(0,90));dr=ImageDraw.Draw(board)
        dr.text((12,6),title,font=font,fill=(20,30,40));dr.text((12,34),f'{identifier} E{index+1}',font=font,fill=(20,30,40))
        dr.text((12,62),'Orange gaps unknown; local response is not identity probability.',font=font,fill=(50,60,70))
        panels.append(board)
    canvas=Image.new('RGB',(1600,panels[0].height*2),(245,247,250))
    for k,p in enumerate(panels):canvas.paste(p,((k%2)*800,(k//2)*p.height))
    canvas.save(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='free_checks_002');args=ap.parse_args()
    out=HERE/args.run;out.mkdir(exist_ok=False);tick=time.perf_counter()
    report=dict(source_sha256=sha(HERE/'heat_free.py'),numeric=numeric_checks(),cases=[],
        labels_read=False,new_inference=False,devices_connected=False)
    cfg=inputs();allowed=set(cfg['pilot_ids'])
    examples={'R_static_20260922_01__f000095__right','dual_20260923_113305__pair0555__right',
        'dual_20260923_101643__pair0375__right','dual_20260923_101517__pair0075__right'}
    for job in cfg['jobs']:
        if job['id'] not in allowed:continue
        priors=surface.load_surface_priors(dict(job,seeds=job['original_seeds']))
        oldrecord=json.loads(gzip.decompress((H3/'all_001/记录'/f'{job["id"]}.json.gz').read_bytes()))
        for index,(seed,rp) in enumerate(zip(job['seeds'],job['response_files'])):
            with np.load(rp,allow_pickle=False) as z:heat=z['response'].copy()
            result=describe_free(priors,seed,heat);n=len(result['points_xy'])
            assert n==len(result['supported'])==len(result['protect_support'])
            assert len(result['segment_status'])==max(0,n-1)
            assert np.all(np.asarray(result['supported'])<=np.asarray(result['protect_support']))
            old=oldrecord['entries'][index]['free']
            name=out/f'{job["id"]}__entry{index+1:02d}.json.gz'
            name.write_bytes(gzip.compress(json.dumps(result,ensure_ascii=False,allow_nan=False).encode('utf-8')))
            d=result['diagnostics'];c=d.get('selected_axis',{})
            summary=dict(id=job['id'],entry_index=index,available=result['available'],
                old_points=len(old['points_xy']),new_points=n,raw_support=sum(result['supported']),
                heat_protected=sum(result['protect_support']),selected_axis=c,
                angle_degrees=c.get('angle_degrees'),normal_shift_original_px=d.get('normal_shift_original_px'),
                elapsed_seconds=d['elapsed_seconds'],reason=result['reason'],axis_candidates=d['axis_candidates'])
            report['cases'].append(summary)
            if job['id'] in examples:
                preview(name.with_suffix('').with_suffix('.png'),priors,heat,old,result,job['id'],index)
    report['elapsed_seconds']=time.perf_counter()-tick
    (out/'check_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
