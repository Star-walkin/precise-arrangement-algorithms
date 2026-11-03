"""Frozen H4/H5 comparison of saved results; no labels, network calls or tuning.

Reuses the exact H4 comparison's free-profile gates and length calculations.
Direction is measured independently on a shared finite continuation horizon;
shorter or absent continuations are not counted as lower-turn improvements.
"""
from pathlib import Path
from collections import Counter
import argparse, gzip, hashlib, importlib.util, json, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
H4=HERE.parent/'postprocess_b_20261031_v4'
HELPER=H4/'compare_h3_h4.py'
HELPER_SHA='f35cf70abe2bdf012eed44a7d7c91a43a2a20412cafe608f5525460cfc29da5a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HELPER)==HELPER_SHA
spec=importlib.util.spec_from_file_location('h5_frozen_h4_comparison',HELPER)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
load=base.load;save=base.save
SAMPLE_FILE=HERE/'diagnostics_source/fixed_samples_metadata.json'
SETTINGS=dict(base.SETTINGS,continuation_horizon_original_px=240.,direction_sample_original_px=8.,direction_chord_original_px=48.,
    exact_surface_coordinate_tolerance_half_px=1.e-7,
    comparison_revision='003_separate_raw_A7_support',
    raw_A7_only_rule='Both interval endpoints must be raw_a7_supported AND fail the frozen B heat gate, and neither may be entry_condition. Mixed B/A7 transitions are excluded from this strict A7-only length.',
    historical_unknown_rule='unknown_length_original_px retains the historical not-B-supported meaning. It can include new A7-only segments; unresolved_after_B_or_A7_original_px measures the remaining union-unsupported part.',
    raw_A7_color_RGB=[50,220,255],
    note='H4 heat/raw joint gates are unchanged. A7-only is separate image-based evidence, not model support or identity truth. Total/free/union-unknown coverage accompanies direction.')

def rename_versions(obj):
    if isinstance(obj,list):return [rename_versions(x) for x in obj]
    if not isinstance(obj,dict):return obj
    names={'H3':'H4','H4':'H5','H4_free_longer_count':'H5_free_longer_count','H4_free_shorter_count':'H5_free_shorter_count'}
    return {names.get(k,k):rename_versions(v) for k,v in obj.items()}

def common_profiles(old,new):return rename_versions(base.common_profiles(old,new))

def turn_geometry(entry,candidate):
    if not candidate:return dict(available=False,reason='no_candidate')
    points=np.asarray(candidate['points_xy'],float);stages=np.asarray(candidate['stage'])
    free=np.flatnonzero(stages=='free');post=np.flatnonzero(stages!='free')
    if not len(free) or not len(post):return dict(available=False,reason='only_free_or_missing_stage')
    start=int(free[-1]);path=points[start:]
    dist=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    unique=np.r_[True,np.diff(dist)>1.e-9];path=path[unique];dist=dist[unique]
    if len(path)<2:return dict(available=False,reason='zero_continuation')
    length=float(dist[-1]);horizon=min(120.,length)
    grid=np.linspace(0.,horizon,max(2,int(np.ceil(horizon/4.))+1))
    def at(s):return np.stack([np.interp(s,dist,path[:,k]) for k in (0,1)],axis=-1)
    # Measurement uses 48-original-px chords; it does not smooth output points.
    ahead=at(np.minimum(grid+12.,horizon));behind=at(np.maximum(grid-12.,0.))
    vectors=ahead-behind;theta=np.unwrap(np.arctan2(vectors[:,1],vectors[:,0]))
    direction=np.asarray(entry['free']['axis_direction_xy'],float)
    reference=float(np.arctan2(direction[1],direction[0]))
    theta=theta-theta[0]+((theta[0]-reference+np.pi)%(2*np.pi)-np.pi)
    return dict(available=True,continuation_length_original_px=2*length,measured_horizon_original_px=2*horizon,
        full_common_horizon=bool(length>=120.-1.e-7),maximum_turn_from_free_degrees=float(np.rad2deg(np.max(np.abs(theta)))),
        direction_total_variation_degrees=float(np.rad2deg(np.abs(np.diff(theta)).sum())),
        initial_chord_jump_degrees=float(abs(np.rad2deg(theta[0]))),identity_error_metric=False)

def surface_constraint(entry,candidate):
    if not candidate:return dict(applicable=False,reason='no_candidate')
    stages=np.asarray(candidate['stage']);use=stages=='surface'
    if not use.any():return dict(applicable=False,reason='no_surface_stage')
    geometry=entry.get('candidate_track_geometry',{});track_id=candidate.get('track_id')
    if track_id not in geometry:return dict(applicable=True,exact=False,reason='missing_saved_track_geometry')
    indices=np.asarray(candidate.get('surface_source_indices',[]),int);track=np.asarray(geometry[track_id],float)
    if len(indices)!=int(use.sum()) or np.any(indices<0) or np.any(indices>=len(track)):
        return dict(applicable=True,exact=False,reason='invalid_surface_source_indices')
    error=np.linalg.norm(np.asarray(candidate['points_xy'],float)[use]-track[indices],axis=1)
    return dict(applicable=True,exact=bool(error.max()<=1.e-7),surface_points=len(indices),
        maximum_error_original_px=float(2*error.max()),track_id=track_id,identity_verified=False)

def measure(entry,candidate):
    result=base.measure(entry,candidate)
    free=entry['free'];points=np.asarray(free['points_xy'],float).reshape(-1,2)
    result.update(raw_A7_supported_length_original_px=0.,raw_A7_only_length_original_px=0.,
        observed_B_or_A7_length_original_px=0.,unresolved_after_B_or_A7_original_px=0.,
        raw_A7_only_abandoned_length_original_px=0.)
    if len(points)>=2:
        heat,_=base.flags(free['evidence']);raw=np.asarray(free.get('raw_a7_supported',[False]*len(points)),bool)
        observed=np.asarray(free.get('observed_straight_support',heat|raw),bool)
        assert raw.shape==heat.shape==observed.shape
        post=np.asarray(free['point_status'])!='entry_condition';length=2*np.linalg.norm(np.diff(points,axis=0),axis=1)
        def intervals(mask):return mask[:-1]&mask[1:]&post[:-1]&post[1:]
        ao=intervals(raw&~heat);union=intervals(heat|raw)
        result.update(raw_A7_supported_length_original_px=float(length[intervals(raw)].sum()),
            raw_A7_only_length_original_px=float(length[ao].sum()),
            observed_B_or_A7_length_original_px=float(length[union].sum()),
            unresolved_after_B_or_A7_original_px=float(length[post[:-1]&post[1:]&~union].sum()),
            stored_observed_straight_support_disagrees_with_fixed_union_points=int(np.count_nonzero(observed!=(heat|raw))))
        if candidate:
            distance=base.poly_dist(points,np.asarray(candidate['points_xy'],float));lost=distance>.4*float(free['width_half_px'])
            result['raw_A7_only_abandoned_length_original_px']=float(length[ao&(lost[:-1]|lost[1:])].sum())
    result['turn_geometry']=turn_geometry(entry,candidate)
    result['surface_constraint']=surface_constraint(entry,candidate)
    diagnostics=entry.get('free',{}).get('diagnostics',{})
    alternatives=diagnostics.get('best_refined_candidates',[])
    result['longest_saved_refined_axis_original_px']=max([x.get('span_original_px',0.) for x in alternatives],default=None)
    return result

def aggregate(rows):
    if not rows:return dict(pairs=0)
    oldrows=[]
    for r in rows:
        cp=r['common_profiles'].copy()
        if cp.get('eligible'):cp.update(H3=cp.pop('H4'),H4=cp.pop('H5'))
        oldrows.append(dict(H3=r['H4'],H4=r['H5'],common_profiles=cp))
    result=rename_versions(base.aggregate(oldrows))
    for v in ('H4','H5'):
        extra_keys=['raw_A7_supported_length_original_px','raw_A7_only_length_original_px',
            'observed_B_or_A7_length_original_px','unresolved_after_B_or_A7_original_px','raw_A7_only_abandoned_length_original_px']
        result[v]['sums'].update({k:float(sum(r[v][k] for r in rows)) for k in extra_keys})
        result[v]['medians'].update({k:float(np.median([r[v][k] for r in rows])) for k in extra_keys})
        tracks=[r[v]['surface_constraint'] for r in rows if r[v]['surface_constraint']['applicable']]
        result[v]['surface_constraint']=dict(applicable=len(tracks),exact=sum(x['exact'] for x in tracks),
            failures=[r['id'] for r in rows if r[v]['surface_constraint']['applicable'] and not r[v]['surface_constraint']['exact']])
    comparable=[r for r in rows if r['common_profiles']['eligible'] and all(r[v]['turn_geometry'].get('full_common_horizon',False) for v in ('H4','H5'))]
    result['direction_on_equal_continuation_horizon']=dict(eligible=len(comparable),
        ineligible=len(rows)-len(comparable),horizon_original_px=240.,
        both_amplitude_and_variation_lower_count=sum(r['H5']['turn_geometry']['maximum_turn_from_free_degrees']<r['H4']['turn_geometry']['maximum_turn_from_free_degrees'] and r['H5']['turn_geometry']['direction_total_variation_degrees']<r['H4']['turn_geometry']['direction_total_variation_degrees'] for r in comparable),
        median_amplitude_change_degrees=float(np.median([r['H5']['turn_geometry']['maximum_turn_from_free_degrees']-r['H4']['turn_geometry']['maximum_turn_from_free_degrees'] for r in comparable])) if comparable else None,
        median_variation_change_degrees=float(np.median([r['H5']['turn_geometry']['direction_total_variation_degrees']-r['H4']['turn_geometry']['direction_total_variation_degrees'] for r in comparable])) if comparable else None,
        no_connection_is_not_lower_turn_success=True)
    return result

def visual(oldrun,newrun,identity,dest):
    paths=[run/'四栏检测结果'/f'{identity}.jpg' for run in (oldrun,newrun)]
    with Image.open(paths[0]) as old,Image.open(paths[1]) as new:
        assert old.size==new.size==(8192,1655)
        panels=[new.crop((0,680,2048,1655)).convert('RGB'),old.crop((6144,680,8192,1655)).convert('RGB'),new.crop((6144,680,8192,1655)).convert('RGB')]
    board=Image.new('RGB',(6144,1065),(246,248,251));draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    for k,(title,im) in enumerate(zip(['原图（固定搜索ROI）','H4：固定2000px积分评分','H5：有支持直段长度优先'],panels)):
        x=k*2048;draw.text((x+20,8),title,font=font,fill=(20,30,40));draw.text((x+20,50),identity,font=small,fill=(50,65,80));board.paste(im,(x,90))
    # Show only A7-only intervals retained by the selected path; this overlay is
    # diagnostic and never modifies the saved candidate, input or response.
    record=load(newrun/'记录'/f'{identity}.json.gz');entry=record['entries'][record['selected_entry']]
    selected=record.get('selected_candidate');candidate=entry['connection']['candidates'][selected] if selected is not None else None
    free=entry['free'];points=np.asarray(free['points_xy'],float).reshape(-1,2);count=0
    if len(points)>=2 and candidate:
        heat,_=base.flags(free['evidence']);raw=np.asarray(free.get('raw_a7_supported',[False]*len(points)),bool)
        post=np.asarray(free['point_status'])!='entry_condition';mask=raw&~heat&post
        retained=base.poly_dist(points,np.asarray(candidate['points_xy'],float))<=.4*float(free['width_half_px'])
        mask&=retained
        for i in np.flatnonzero(mask[:-1]&mask[1:]):
            a=points[i]+[4096,90];b=points[i+1]+[4096,90]
            draw.line([tuple(a),tuple(b)],fill=tuple(SETTINGS['raw_A7_color_RGB']),width=3);count+=1
    legend=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    draw.text((4116,71),'青蓝：A7原图补段，B未支持；不计入旧热图联合指标',font=legend,fill=(15,85,110))
    board.save(dest,quality=93,subsampling=0)
    return dict(id=identity,source_paths=list(map(str,paths)),source_sha256=[sha(p) for p in paths],output=str(dest),output_sha256=sha(dest),raw_A7_only_displayed_intervals=count)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--h4-run',default='all_001');ap.add_argument('--h5-run',default='pilot_001');ap.add_argument('--out',required=True);args=ap.parse_args()
    oldrun=H4/args.h4_run;newrun=HERE/args.h5_run;output=HERE/args.out;output.mkdir(exist_ok=False)
    config=load(HERE/'inputs.json');sample=load(SAMPLE_FILE);summary=load(newrun/'summary.json')
    ids=[r['id'] for r in summary['rows']];allowed={j['id']:j for j in config['jobs']}
    assert len(ids)==len(set(ids)) and set(ids)<=set(allowed)
    rows=[];images=[];sources=[];start=time.monotonic()
    with (output/'paired_entries.jsonl').open('x',encoding='utf-8') as log:
        for identity in ids:
            paths=[run/'记录'/f'{identity}.json.gz' for run in (oldrun,newrun)];old,new=[load(p) for p in paths]
            assert old['source_image_sha256']==new['source_image_sha256']==allowed[identity]['image_sha256']
            assert len(old['entries'])==len(new['entries'])
            for k,(e4,e5) in enumerate(zip(old['entries'],new['entries'])):
                assert e4['seed']==e5['seed'] and e4['response_sha256']==e5['response_sha256']
                row=dict(id=identity,entry_index=k,H4=measure(e4,base.main_candidate(e4)),H5=measure(e5,base.main_candidate(e5)),common_profiles=common_profiles(e4,e5))
                rows.append(row);log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            i4,i5=old['selected_entry'],new['selected_entry'];e4,e5=old['entries'][i4],new['entries'][i5]
            c4=e4['connection']['candidates'][old['selected_candidate']] if old['selected_candidate'] is not None else None
            c5=e5['connection']['candidates'][new['selected_candidate']] if new['selected_candidate'] is not None else None
            common=common_profiles(e4,e5) if e4['seed']==e5['seed'] else dict(eligible=False,reason='selected_entry_changed')
            images.append(dict(id=identity,selected_entry_H4=i4,selected_entry_H5=i5,H4=measure(e4,c4),H5=measure(e5,c5),common_profiles=common))
            sources.append(dict(id=identity,H4_sha256=sha(paths[0]),H5_sha256=sha(paths[1])))
    development=set(sample['development_ids']);extra={j['id'] for j in sample['extra_open_check']};visuals=[]
    for label,subset in [('固定14开发图_原图_H4_H5',development),('冻结后8图_原图_H4_H5',extra)]:
        folder=output/label;folder.mkdir()
        for identity in sorted(set(ids)&subset):visuals.append(visual(oldrun,newrun,identity,folder/f'{identity}.jpg'))
    result=dict(command=sys.argv,comparison_source_sha256=sha(__file__),frozen_H4_comparison_source=str(HELPER),frozen_H4_comparison_sha256=HELPER_SHA,
        fixed_sample_manifest_sha256=sha(SAMPLE_FILE),settings=SETTINGS,H4_run=str(oldrun),H5_run=str(newrun),
        H4_summary_sha256=sha(oldrun/'summary.json'),H5_summary_sha256=sha(newrun/'summary.json'),
        by_image=aggregate(images),by_entry=aggregate(rows),development14=aggregate([r for r in images if r['id'] in development]),
        extra8_postfreeze=aggregate([r for r in images if r['id'] in extra]),
        selected_entry_changed=sum(r['selected_entry_H4']!=r['selected_entry_H5'] for r in images),
        specified_counterexample=next((r for r in images if r['id']=='dual_20260923_101517__pair0195__right'),None),
        fixed_visual_count=len(visuals),seconds=time.monotonic()-start,labels_read=False,new_inference=False,
        claims='Supported straight length, smaller turn and exact F3 center coordinates are diagnostic properties, not correct target identity or independent recognition accuracy. All failures and shorter paths remain in the denominator.')
    save(output/'summary.json',result);save(output/'paired_images.json',images);save(output/'sources.json',sources);save(output/'visuals.json',visuals)
    print(json.dumps(result,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
