"""H5 numeric support/domain checks and frozen development14; no inference."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import heat_free as core

from common import HERE,H4,inputs,surface,load,sha
from heat_free import describe_free,_evaluate_axis,_rank,_segments,_surface_domain,_terminal_domain,_a7_context,_measure_raw_a7,PARAMETERS


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
    measurement_count=[0];original_measure=core._measure
    def counted_measure(*args,**kwargs):
        measurement_count[0]+=1
        return original_measure(*args,**kwargs)
    core._measure=counted_measure
    try:result=describe_free(priors,seed,heat)
    finally:core._measure=original_measure
    assert measurement_count[0]==result['diagnostics']['axis_candidates']<=180
    assert result['diagnostics']['shadow_profile_evaluations']==0
    assert result['diagnostics']['selected_axis']['length_bin']==result['diagnostics']['maximum_generated_union_length_bin']
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
    assert _rank(lr)>_rank(sr)
    assert lr['common_view_original_px']==sr['common_view_original_px']==4400.
    assert result['diagnostics']['old_endpoint_used'] is False
    sam=np.zeros((775,w),np.uint8);sam[525+60:525+180,80:240]=255;sam[525+90:525+110,120:130]=0
    domain=_surface_domain(dict(priors,sam2_surface_half=sam))
    assert domain['inside'][100,125]==1
    pp=np.c_[np.arange(20.,390.,3.),np.full(len(np.arange(20.,390.,3.)),100.)]
    terminal=_terminal_domain(domain,pp,3.,10.)
    assert terminal['applied'] and 250<=pp[terminal['index'],0]<=265
    no_crossing=_terminal_domain(domain,pp+[0,120],3.,10.)
    assert not no_crossing['applied']
    protected=np.zeros(105,bool);protected[:21]=True;protected[60:101]=True
    ev=dict(heat_profile_supported=protected,image_heat_supported=protected.copy(),center=np.full(105,.6),side_drop=np.full(105,.55))
    runs,gaps,_,_=_segments(ev,3.,5.,60.)
    assert len(runs)==2 and len(gaps)==1 and gaps[0]['allowed_gap_original_px']==400.
    assert not gaps[0]['supported'] and not protected[21:60].any()
    for axis in [result]+result.get('alternative_free_axes',[]):
        p=np.asarray(axis['points_xy']);s=np.asarray(axis['protect_support']);entry=np.array(axis['point_status'])=='entry_condition'
        observed=s[:-1]&s[1:]&~entry[:-1]&~entry[1:]
        length=float((np.linalg.norm(np.diff(p,axis=0),axis=1)*2)[observed].sum())
        assert abs(length-axis['visible_straight_length_original_px'])<1.e-6
    # Strict A7-only evidence is separate from B and never extrapolates a gap.
    ap=np.c_[np.arange(40.,440.,4.),np.full(100,1250.)]
    aa=dict(a7,center_xy=ap.tolist(),visible=[True]*100,raw_supported=[True]*100,
        local_repeat_supported=[True]*100,signature={'status':'periodic_strand_evidence'})
    ctx=_a7_context(aa);pq=(ap-np.array([0.,1050.]))*.5
    empty=np.zeros(100,bool);e=dict(valid=np.ones(100,bool),raw_score=np.full(100,.8),
        raw_contrast=np.full(100,.10),heat_profile_supported=empty.copy())
    me=_measure_raw_a7(copy.deepcopy(e),pq,2.,10.,ctx)
    assert me['raw_a7_supported'].all() and me['B_conflict'].all()
    assert not me['heat_profile_supported'].any()
    assert _a7_context(dict(aa,accepted=False)) is None
    assert _a7_context(dict(aa,signature={'status':'unverified_bright_tool'})) is None
    assert not _measure_raw_a7(copy.deepcopy(e),pq+[0,3.],2.,10.,ctx)['raw_a7_supported'].any()
    for field in ('visible','raw_supported','local_repeat_supported'):
        bad=dict(aa);bad[field]=[False]*100
        assert not _measure_raw_a7(copy.deepcopy(e),pq,2.,10.,_a7_context(bad))['raw_a7_supported'].any()
    bad_e=copy.deepcopy(e);bad_e['raw_score'][:]=.34
    assert not _measure_raw_a7(bad_e,pq,2.,10.,ctx)['raw_a7_supported'].any()
    bad_e=copy.deepcopy(e);bad_e['raw_contrast'][:]=.019
    assert not _measure_raw_a7(bad_e,pq,2.,10.,ctx)['raw_a7_supported'].any()
    bad_e=copy.deepcopy(e);bad_e['valid'][:]=False
    assert not _measure_raw_a7(bad_e,pq,2.,10.,ctx)['raw_a7_supported'].any()
    broken=dict(aa);broken['visible']=[True]*35+[False]*30+[True]*35
    be=_measure_raw_a7(copy.deepcopy(e),pq,2.,10.,_a7_context(broken))
    assert not be['raw_a7_supported'][35:65].any()
    assert be['raw_a7_supported'][:35].all() and be['raw_a7_supported'][65:].all()
    tiny=dict(aa);tiny['visible']=[True]*10+[False]*90
    assert not _measure_raw_a7(copy.deepcopy(e),pq,2.,10.,_a7_context(tiny))['raw_a7_supported'].any()
    assert not _measure_raw_a7(copy.deepcopy(e),pq+[220,0],2.,10.,ctx)['raw_a7_supported'].any()
    ca=_evaluate_axis(g,long,valid,origin,d,10.,60.,0.,0.,3.,a7_context=ctx)
    cb=_evaluate_axis(g,long,valid,origin,d,10.,60.,0.,0.,3.,a7_context=None)
    assert core._summary(ca['b_only_shadow'])==core._summary(cb['b_only_shadow'])
    return dict(passed=True,checks=['unchanged_seed_and_response','finite_angle_offset_reestimation',
        'beyond_initial_short_hint_extent','180_axis_budget','aligned_per_point_arrays',
        'raw_support_subset_of_heat_protection','constant_heat_rejected','NaN_gap_not_observed',
        'full_bounded_length_priority','SAM2_internal_holes_not_terminal','stable_far_side_exit_bound',
        'outside_entry_not_truncated_without_surface_crossing','joint_flank_gap_within_existing400_budget',
        'entry_and_unknown_excluded_from_all_axis_lengths','raw_A7_does_not_change_B_flags',
        'raw_A7_requires_accepted_periodic_current_match','raw_A7_axis_distance_bound',
        'raw_A7_requires_all_three_saved_flags','raw_A7_current_image_and_valid_gates',
        'raw_A7_gaps_remain_false','raw_A7_piece_minimum48','raw_A7_no_extrapolation',
        'B_shadow_does_not_resample_profiles','B_shadow_unchanged_by_raw_A7_switch',
        'selected_union_tier_is_generated_maximum'],
        short_length=sr['visible_straight_length_original_px'],long_length=lr['visible_straight_length_original_px'])


def draw_line(image,points,states):
    p=np.asarray(points,float)
    if len(p)<2:return image
    out=image.copy();arc=0.
    for i in range(1,len(p)):
        status=states[i] if i<len(states) else 'unknown'
        color=(40,230,40) if status in ('image_supported','joint_heat_image_supported','a7_image_supported') else (255,80,220) if status=='raw_a7_supported' else (0,225,255) if status=='heatmap_supported' else (255,255,255) if status=='entry_condition' else (0,140,255)
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
    for title,im in [('Original',base),('Frozen B response (0..1)',heat_img),('H4 free result',old_img),('H5: green=joint; yellow=B; pink=raw A7',new_img)]:
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
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='free_checks_001');args=ap.parse_args()
    out=HERE/args.run;out.mkdir(exist_ok=False);tick=time.perf_counter()
    report=dict(source_sha256=sha(HERE/'heat_free.py'),numeric=numeric_checks(),cases=[],
        labels_read=False,new_inference=False,devices_connected=False)
    cfg=inputs();allowed=set(cfg['pilot_ids'])
    examples={'dual_20260923_101517__pair0195__right','L_run_20260923_01__f000015__left',
        'L_run_20260923_01__f000142__left','dual_20260924_111500__pair0015__left'}
    for job in cfg['jobs']:
        if job['id'] not in allowed:continue
        priors=surface.load_surface_priors(dict(job,seeds=job['original_seeds']))
        oldrecord=json.loads(gzip.decompress((H4/'all_001/记录'/f'{job["id"]}.json.gz').read_bytes()))
        for index,(seed,rp) in enumerate(zip(job['seeds'],job['response_files'])):
            with np.load(rp,allow_pickle=False) as z:heat=z['response'].copy()
            result=describe_free(priors,seed,heat);n=len(result['points_xy'])
            assert n==len(result['supported'])==len(result['protect_support'])
            assert len(result['segment_status'])==max(0,n-1)
            assert np.all(np.asarray(result['supported'])<=np.asarray(result['protect_support']))
            r2file=HERE/'free_checks_002'/f'{job["id"]}__entry{index+1:02d}.json.gz'
            r2=json.loads(gzip.decompress(r2file.read_bytes()))
            parents=lambda r:[(a['angle_degrees'],a['offset_widths']) for a in r['diagnostics'].get('refinement_parent_axes',[])]
            assert parents(result)==parents(r2),(job['id'],parents(result),parents(r2))
            if result['available']:
                assert result['diagnostics']['selected_axis']['length_bin']==result['diagnostics']['maximum_generated_union_length_bin']
            for axis in [result]+result.get('alternative_free_axes',[]):
                assert len(axis['raw_a7_supported'])==len(axis['observed_straight_support'])==len(axis['points_xy'])
                obs=np.asarray(axis['observed_straight_support'],bool);hp=np.asarray(axis['protect_support'],bool);ar=np.asarray(axis['raw_a7_supported'],bool)
                assert np.array_equal(obs,hp|ar)
                if not axis['available']:continue
                p=np.asarray(axis['points_xy']);ent=np.asarray(axis['point_status'])=='entry_condition'
                arc=np.linalg.norm(np.diff(p,axis=0),axis=1)*2.;notentry=~ent[:-1]&~ent[1:]
                assert axis['selected_sample_step_original_px'] in (6.,12.)
                assert np.allclose(arc,axis['selected_sample_step_original_px'],atol=1.e-6)
                interval=obs[:-1]&obs[1:]&notentry
                assert abs(float(arc[interval].sum())-axis['observed_straight_length_original_px'])<1.e-6
                rawonly=ar&~hp
                assert abs(float(arc[rawonly[:-1]&rawonly[1:]&notentry].sum())-axis['raw_a7_only_length_original_px'])<1.e-6
                binterval=hp[:-1]&hp[1:]&notentry
                assert abs(float(arc[interval&~binterval].sum())-axis['observed_added_length_original_px'])<1.e-6
                assert np.array_equal(np.asarray(axis['B_conflict']),ar&~hp)
            old=oldrecord['entries'][index]['free']
            name=out/f'{job["id"]}__entry{index+1:02d}.json.gz'
            name.write_bytes(gzip.compress(json.dumps(result,ensure_ascii=False,allow_nan=False).encode('utf-8')))
            d=result['diagnostics'];c=d.get('selected_axis',{})
            summary=dict(id=job['id'],entry_index=index,available=result['available'],
                old_points=len(old['points_xy']),new_points=n,raw_support=sum(result['supported']),
                heat_protected=sum(result['protect_support']),selected_axis=c,
                raw_a7_support=sum(result['raw_a7_supported']),B_conflict=sum(result['B_conflict']),
                observed_length=result.get('observed_straight_length_original_px'),raw_a7_only_length=result.get('raw_a7_only_length_original_px'),
                observed_added_length=result.get('observed_added_length_original_px'),
                visible_length=result.get('visible_straight_length_original_px'),
                joint_length=result.get('joint_straight_length_original_px'),
                terminal=result.get('terminal_domain'),alternatives=len(result.get('alternative_free_axes',[])),
                selected_sample_step_original_px=result.get('selected_sample_step_original_px'),R2_parent_axes_preserved=True,
                angle_degrees=c.get('angle_degrees'),normal_shift_original_px=d.get('normal_shift_original_px'),
                elapsed_seconds=d['elapsed_seconds'],reason=result['reason'],axis_candidates=d['axis_candidates'])
            report['cases'].append(summary)
            if job['id'] in examples:
                preview(name.with_suffix('').with_suffix('.png'),priors,heat,old,result,job['id'],index)
    report['elapsed_seconds']=time.perf_counter()-tick
    (out/'check_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
