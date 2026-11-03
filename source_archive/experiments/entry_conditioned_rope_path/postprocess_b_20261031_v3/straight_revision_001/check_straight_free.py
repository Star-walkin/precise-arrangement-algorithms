"""Numeric checks plus the existing fixed H2 development12; no new inference."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
H1=HERE.parent/'postprocess_b_20261030_v1'
H2=HERE.parent/'postprocess_b_20261030_v2'
sys.path.insert(0,str(H1))
from priors import load_priors
from straight_free import describe_free,correct_free


def synthetic():
    h,w=250,400; yy,xx=np.mgrid[:h,:w]
    gray=np.full((h,w),.5,np.float32);gray[abs(yy-100)<5]=0
    heat=(.8*np.exp(-.5*((yy-100)/3.)**2)).astype(np.float32)
    image=np.full((775,w,3),127,np.uint8)
    image[525:]=np.repeat(np.rint(gray*255).astype(np.uint8)[...,None],3,axis=2)
    x=np.arange(20.,304.,3.);p=np.c_[x,100.+2*np.sin(x/8)]
    support=np.ones(len(p),bool);support[30:40]=False
    old=dict(points_xy=p.tolist(),supported=support.tolist(),width_half_px=10.,
        point_status=['entry_condition' if q<=80 else 'joint_heat_image_supported' for q in x],
        visible_runs=[dict(start_index=0,end_index=len(p)-1)],gap_hypotheses=[dict(start_index=30,end_index=39)],
        entry_conflict=False,available=True,diagnostics={})
    cp=np.c_[np.arange(0,w*2,4.),np.full(w//2,1250.)]
    a7=dict(candidate_id='synthetic',accepted=True,plausible=True,origin_xy=[400.,1250.],
        direction_xy=[-1.,0.],width_px=20.,visible=(cp[:,0]<=500).tolist(),
        raw_supported=(cp[:,0]<=500).tolist(),center_xy=cp[::-1].tolist())
    # The real A7 array follows its saved axis order, including reverse axes.
    # H2 saved axes have increasing x, so use that storage convention here;
    # orient the entry backwards instead to verify sign disambiguation below.
    a7['direction_xy']=[1.,0.];a7['center_xy']=cp.tolist()
    priors=dict(original_half=image,a7_candidates=[a7],original_shape=[1550,800],a6_entries=[],pitch_px=30.)
    seed=dict(xy=[[40.,1250.],[160.,1250.]],source='synthetic')
    unchanged=copy.deepcopy(seed)
    result=correct_free(priors,seed,heat,old)
    assert seed==unchanged and result['straightening']['seed_changed'] is False
    assert result['straightening']['axis_source']=='accepted_A7_measured_axis'
    assert result['straightening']['after']['perpendicular_rms_original_px']<1.e-8
    assert result['straightening']['after']['turn_max_deg']<1.e-4
    assert not any(result['supported'][30:40])
    assert len(result['raw_points_xy'])==len(p) and result['unmodified_response_tail']['points_xy']
    assert len(result['points_xy'])<len(p)
    assert len(result['segment_status'])==len(result['points_xy'])-1
    for v in result['evidence'].values(): assert len(v)==len(result['points_xy'])
    nan_heat=heat.copy();nan_heat[:,100:120]=np.nan
    nr=correct_free(priors,seed,nan_heat,old);coords=np.asarray(nr['points_xy'])
    assert not np.asarray(nr['supported'])[(coords[:,0]>=100)&(coords[:,0]<120)].any()
    fallback=correct_free(dict(priors,a7_candidates=[]),seed,heat,old)
    assert 'fallback_not_A7' in fallback['straightening']['axis_source']
    assert fallback['straightening']['after']['perpendicular_rms_original_px']<1.e-8
    reverse_seed=dict(xy=seed['xy'][::-1],source='synthetic')
    revold=copy.deepcopy(old)
    revold['points_xy']=list(reversed(old['points_xy']));revold['supported']=list(reversed(old['supported']))
    revold['point_status']=list(reversed(old['point_status']))
    reversed_result=correct_free(priors,reverse_seed,heat,revold)
    assert reversed_result['axis_direction_xy'][0]<0
    return dict(passed=True,checks=['unchanged_seed_and_B','A7_axis_projection',
        'straightness_residual_and_turn','no_unknown_promotion','A7_end_tail_preserved',
        'array_alignment','NaN_invalid_domain','explicit_robust_fallback','entry_orients_A7_axis'])


def save_preview(path,original,result,identifier,index):
    panels=[]
    for mode in ('before','after'):
        canvas=original.copy()
        points=np.asarray(result['raw_points_xy'] if mode=='before' else result['points_xy'])
        support=result['raw_supported'] if mode=='before' else result['supported']
        if len(points):
            q=points+[0,525]
            for k in range(1,len(q)):
                c=(40,220,40) if support[k-1] and support[k] else (0,150,255)
                if support[k-1] and support[k] or k%5<3:
                    cv2.line(canvas,tuple(np.rint(q[k-1]).astype(int)),tuple(np.rint(q[k]).astype(int)),c,3,cv2.LINE_AA)
        if mode=='after':
            tail=np.asarray(result['unmodified_response_tail']['points_xy'])
            if len(tail)>1:cv2.polylines(canvas,[np.rint(tail+[0,525]).astype(np.int32)],False,(150,150,150),1,cv2.LINE_AA)
        canvas=cv2.resize(canvas,(1024,750),interpolation=cv2.INTER_AREA)
        board=Image.new('RGB',(1024,840),(245,247,250));board.paste(Image.fromarray(canvas[:,:,::-1]),(0,90))
        draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',21)
        label='H2 raw free candidate' if mode=='before' else 'H3 straight free | gray = unmodified tail'
        draw.text((12,8),label,font=font,fill=(20,30,40))
        draw.text((12,37),f'{identifier} / entry {index+1}',font=font,fill=(20,30,40))
        draw.text((12,64),'Green: local support; orange: unobserved gap. No identity guarantee.',font=font,fill=(50,60,70))
        panels.append(board)
    image=Image.new('RGB',(2048,840));image.paste(panels[0],(0,0));image.paste(panels[1],(1024,0));image.save(path)


def main():
    tick=time.perf_counter();out=HERE/'straight_checks_001'
    out.mkdir(exist_ok=False)
    report=dict(synthetic=synthetic(),real_examples=[],labels_read=False,new_inference=False,
        source_sha256=hashlib.sha256((HERE/'straight_free.py').read_bytes()).hexdigest())
    manifest=json.loads((H2/'inputs.json').read_text(encoding='utf-8-sig'))
    ids=set(manifest['pilot_ids'])
    for job in manifest['jobs']:
        if job['id'] not in ids: continue
        priors=load_priors(dict(job,seeds=job['original_seeds']))
        for k,(seed,response_path) in enumerate(zip(job['seeds'],job['response_files'])):
            with np.load(response_path,allow_pickle=False) as data: heat=data['response'].copy()
            result=describe_free(priors,seed,heat)
            record=out/f'{job["id"]}__entry{k+1:02d}.json'
            record.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            summary=dict(id=job['id'],entry_index=k+1,available=result.get('available'),
                points=len(result['points_xy']),supported_points=sum(result['supported']),
                raw_points=len(result['raw_points_xy']),entry_conflict=result.get('entry_conflict'),
                straightening=result['straightening'])
            report['real_examples'].append(summary)
            if job['id'] in ('dual_20260923_101517__pair0075__right','R_run_20260923_01__f000055__right','L_run_20260923_01__f000015__left'):
                save_preview(record.with_suffix('.png'),priors['original_half'],result,job['id'],k)
    report['elapsed_seconds']=time.perf_counter()-tick
    (out/'check_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    brief=[dict(id=r['id'],entry=r['entry_index'],points=r['points'],supported=r['supported_points'],
        applied=r['straightening']['applied'],source=r['straightening'].get('axis_source'),
        before_rms=r['straightening'].get('before',{}).get('perpendicular_rms_original_px'),
        after_rms=r['straightening'].get('after',{}).get('perpendicular_rms_original_px'),
        downgraded=r['straightening'].get('downgraded_support_points')) for r in report['real_examples']]
    print(json.dumps(dict(synthetic=report['synthetic'],source_sha256=report['source_sha256'],
        elapsed_seconds=report['elapsed_seconds'],real=brief),ensure_ascii=False))


if __name__=='__main__':main()
