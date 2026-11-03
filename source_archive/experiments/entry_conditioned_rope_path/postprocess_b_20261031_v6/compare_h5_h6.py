"""Frozen-protocol H5/H6 saved-result comparison, without labels or inference.

F3 source geometry is an input constraint, never geometric ground truth. Report
coverage removed by topology separately from observed support lost. Rejecting a
connection or outputting a shorter prefix is not automatically a correct result.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
H5=HERE.parent/'postprocess_b_20261031_v5'
HELPER=H5/'compare_h4_h5.py'
HELPER_SHA='e3e1ef3a380a1830ee09f3d219b8df00b0e17fdf49b8899695c96bd5e6c3bd3b'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(HELPER)==HELPER_SHA
spec=importlib.util.spec_from_file_location('h6_frozen_h5_measurement',HELPER)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
load=old.load;save=old.save
SETTINGS=dict(profile_gates=old.SETTINGS,seed_intervals_excluded=True,
    removed_tail_definition='Intervals after the last saved task-prefix vertex; never all called unknown. Original-image support = fixed B+raw gate OR recorded strict raw_A7_supported.',
    topology_definition='Saved H6 finite-source constraint receipts; old H5 curves are not used as truth. No connection is reported separately, never counted as a no-cross success.',
    retention_distance_widths=.4,source_coordinates='half crop xy: original = 2*xy+[0,1050]',
    visual_crop_original_xyxy=[0,1050,4096,3000])

def supports(f):
    p=np.asarray(f.get('points_xy',[]),float).reshape(-1,2);n=len(p)
    if n<2:return p,np.empty(0),{}
    heat,joint=old.base.flags(f['evidence']);raw=np.asarray(f.get('raw_a7_supported',[False]*n),bool)
    post=np.asarray(f['point_status'])!='entry_condition';seg=2*np.linalg.norm(np.diff(p,axis=0),axis=1)
    both=lambda m:m[:-1]&m[1:]&post[:-1]&post[1:]
    masks=dict(post_entry=both(np.ones(n,bool)),B_profile=both(heat),B_and_raw=both(joint),
        raw_A7_only=both(raw&~heat),original_image_supported=both(joint|raw),B_or_A7=both(heat|raw))
    masks['B_without_original_support']=masks['B_profile']&~masks['original_image_supported']
    masks['neither_B_nor_A7']=masks['post_entry']&~masks['B_or_A7']
    return p,seg,masks

def interval_totals(seg,masks,select=None):
    if select is None:select=np.ones(len(seg),bool)
    return {k:float(seg[v&select].sum()) for k,v in masks.items()}

def event_counts(events):return dict(Counter(e.get('kind','missing') for e in events))

def measure(e,c,version):
    m=old.measure(e,c)
    full=e.get('full_free_observation',e['free']);p,seg,masks=supports(full)
    task=np.asarray(e['free'].get('points_xy',[]),float).reshape(-1,2)
    projection=e.get('free_projection',{})
    last=int(projection.get('task_prefix_last_index',len(task)-1))
    prefix_equal=len(task)<=len(p) and np.allclose(task,p[:len(task)],atol=1e-7,rtol=0)
    removed=np.arange(len(seg))>=max(0,last)
    kept=~removed
    m.update(full_free_span_original_px=float(seg.sum()),task_prefix_matches_full=bool(prefix_equal),
        task_prefix_last_index=last,full_free_support_lengths=interval_totals(seg,masks),
        retained_prefix_support_lengths=interval_totals(seg,masks,kept),
        removed_tail_support_lengths=interval_totals(seg,masks,removed),
        removed_tail_span_original_px=float(seg[removed].sum()),
        recorded_unresolved_tail_original_px=float(projection.get('unresolved_tail_original_px',0.)),
        task_trimmed=bool(projection.get('trimmed',False)),
        free_projection_event_counts=event_counts(projection.get('events',[])),
        free_projection_permitted_events=sum(bool(v.get('permitted_projected_free_crossing')) for v in projection.get('events',[])),
        free_projection_unpermitted_events=sum(not bool(v.get('permitted_projected_free_crossing')) for v in projection.get('events',[])),
        retained_projection_unpermitted_events=sum(not bool(v.get('permitted_projected_free_crossing')) for v in projection.get('retained_prefix_events',[])),
        connection_rejected_counts=e['connection'].get('rejected',{}),
        rejected_topology_examples_saved=len(e['connection'].get('rejected_topology_examples',[])),
        selected_stage_lengths_original_px={},selected_topology=None,
        full_observation_retained_by_selected_support_lengths={k:0. for k in masks})
    if c:
        q=np.asarray(c['points_xy'],float);ds=2*np.linalg.norm(np.diff(q,axis=0),axis=1);stages=np.asarray(c['stage'])
        m['selected_stage_lengths_original_px']={s:float(ds[stages[1:]==s].sum()) for s in sorted(set(stages))}
        if len(p)>=2:
            retained=old.base.poly_dist(p,q)<=.4*float(full['width_half_px'])
            m['full_observation_retained_by_selected_support_lengths']=interval_totals(seg,masks,retained[:-1]&retained[1:])
        topo=c.get('topology')
        attached=bool(np.any(stages=='surface'))
        m['selected_topology']=dict(applicable=attached,recorded=topo is not None,
            allowed=topo.get('allowed') if topo else None,
            other_track_event_counts=event_counts(topo.get('other_track_events',[])) if topo else None,
            own_track_capture_event_counts=event_counts(topo.get('own_track_capture_events',[])) if topo else None,
            premature_target_event_counts=event_counts(topo.get('premature_target_events',[])) if topo else None,
            single_source_track_id=c.get('track_id'),
            attached_and_recorded_no_other_or_premature=bool(attached and topo and topo.get('allowed') and not topo.get('other_track_events') and not topo.get('premature_target_events')),
            identity_verified=False)
    return m

def same_full_observation(e5,e6):
    a=e5['free'];b=e6.get('full_free_observation',e6['free'])
    pa=np.asarray(a.get('points_xy',[]),float).reshape(-1,2);pb=np.asarray(b.get('points_xy',[]),float).reshape(-1,2)
    same=pa.shape==pb.shape and np.allclose(pa,pb,atol=1e-7,rtol=0)
    result=dict(coordinates_equal=bool(same),H5_points=len(pa),H6_full_points=len(pb))
    if same:
        result['support_array_disagreements']={k:int(np.count_nonzero(np.asarray(a.get(k,[False]*len(pa)))!=np.asarray(b.get(k,[False]*len(pb))))) for k in ['supported','protect_support','raw_a7_supported']}
    return result

def aggregate(rows):
    out=dict(pairs=len(rows))
    if not rows:return out
    keys=['free_length_original_px','heat_supported_length_original_px','joint_supported_length_original_px','raw_A7_only_length_original_px',
        'selected_path_length_original_px','selected_joint_length_original_px','full_free_span_original_px','removed_tail_span_original_px','recorded_unresolved_tail_original_px']
    nested=['full_free_support_lengths','retained_prefix_support_lengths','removed_tail_support_lengths','full_observation_retained_by_selected_support_lengths']
    for v in ['H5','H6']:
        ms=[r[v] for r in rows];connected=[m for m in ms if m['selected_topology'] and m['selected_topology']['applicable']]
        out[v]=dict(kind_counts=dict(Counter(m['kind'] for m in ms)),only_free_count=sum(m['only_free'] for m in ms),
            connected_count=len(connected),no_candidate_count=sum(m['kind']=='no_candidate' for m in ms),
            sums={k:float(sum(m[k] for m in ms)) for k in keys},
            medians={k:float(np.median([m[k] for m in ms])) for k in keys},
            task_trimmed_count=sum(m['task_trimmed'] for m in ms),prefix_mismatch_count=sum(not m['task_prefix_matches_full'] for m in ms),
            attached_recorded_no_other_or_premature_count=sum(m['selected_topology']['attached_and_recorded_no_other_or_premature'] for m in connected),
            topology_records_missing_for_connected=sum(not m['selected_topology']['recorded'] for m in connected),
            retained_projection_unpermitted_events=sum(m['retained_projection_unpermitted_events'] for m in ms),
            free_projection_permitted_events=sum(m['free_projection_permitted_events'] for m in ms),
            free_projection_unpermitted_events=sum(m['free_projection_unpermitted_events'] for m in ms),
            connection_rejected_counts=dict(sum((Counter(m['connection_rejected_counts']) for m in ms),Counter())),
            exact_saved_source_coordinates_count=sum(m['surface_constraint'].get('exact',False) for m in ms),
            selected_stage_lengths_original_px=dict(sum((Counter(m['selected_stage_lengths_original_px']) for m in ms),Counter())))
        for k in nested:
            subkeys=sorted({s for m in ms for s in m[k]});out[v][k]={s:float(sum(m[k].get(s,0.) for m in ms)) for s in subkeys}
    kind=lambda m:'none' if m['kind']=='no_candidate' else 'free' if m['only_free'] else 'connected'
    out['selection_transitions']=dict(Counter(kind(r['H5'])+' -> '+kind(r['H6']) for r in rows))
    out['H6_free_shorter_count']=sum(r['H6']['free_length_original_px']<r['H5']['free_length_original_px']-1 for r in rows)
    out['H6_free_longer_count']=sum(r['H6']['free_length_original_px']>r['H5']['free_length_original_px']+1 for r in rows)
    out['H6_path_shorter_count']=sum(r['H6']['selected_path_length_original_px']<r['H5']['selected_path_length_original_px']-1 for r in rows)
    out['full_observation_coordinates_changed_count']=sum(not r['full_observation_comparison']['coordinates_equal'] for r in rows)
    out['retained_original_image_support_decreased_count']=sum(r['H6']['full_observation_retained_by_selected_support_lengths'].get('original_image_supported',0)<r['H5']['full_observation_retained_by_selected_support_lengths'].get('original_image_supported',0)-1 for r in rows)
    return out

def visual(oldrun,newrun,ident,dest):
    paths=[run/'四栏检测结果'/(ident+'.jpg') for run in [oldrun,newrun]]
    with Image.open(paths[0]) as a,Image.open(paths[1]) as b:
        assert a.size==b.size==(8192,1655)
        panels=[b.crop((0,680,2048,1655)).convert('RGB'),a.crop((6144,680,8192,1655)).convert('RGB'),b.crop((6144,680,8192,1655)).convert('RGB')]
    im=Image.new('RGB',(6144,1065),(246,248,251));d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    for k,(title,panel) in enumerate(zip(['原图（相同固定ROI）','H5：长度优先，碎片中心轨','H6：F3来源轨迹与有限拓扑约束'],panels)):
        d.text((k*2048+20,8),title,font=font,fill=(20,30,40));d.text((k*2048+20,50),ident,font=small,fill=(50,65,80));im.paste(panel,(k*2048,90))
    im.save(dest,quality=93,subsampling=0)
    return dict(id=ident,source_paths=list(map(str,paths)),source_sha256=list(map(sha,paths)),output=str(dest),output_sha256=sha(dest))

def pair(identity,e5,e6,c5,c6):
    assert e5['seed']==e6['seed'] and e5['response_sha256']==e6['response_sha256']
    return dict(id=identity,H5=measure(e5,c5,'H5'),H6=measure(e6,c6,'H6'),full_observation_comparison=same_full_observation(e5,e6))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--h5-run',default='all_001');ap.add_argument('--h6-run',default='pilot_001');ap.add_argument('--out',required=True);a=ap.parse_args()
    oldrun=H5/a.h5_run;newrun=HERE/a.h6_run;out=HERE/a.out;out.mkdir(exist_ok=False)
    cfg=load(HERE/'inputs.json');sm=load(newrun/'summary.json');ids=[r['id'] for r in sm['rows']];allowed={j['id']:j for j in cfg['jobs']}
    assert len(ids)==len(set(ids)) and set(ids)<=set(allowed)
    start=time.monotonic();entries=[];images=[];sources=[]
    with (out/'paired_entries.jsonl').open('x',encoding='utf-8') as log:
        for ident in ids:
            paths=[r/'记录'/(ident+'.json.gz') for r in [oldrun,newrun]];r5,r6=[load(p) for p in paths]
            assert r5['source_image_sha256']==r6['source_image_sha256']==allowed[ident]['image_sha256'] and len(r5['entries'])==len(r6['entries'])
            for i,(e5,e6) in enumerate(zip(r5['entries'],r6['entries'])):
                row=pair(ident,e5,e6,old.base.main_candidate(e5),old.base.main_candidate(e6));row['entry_index']=i;entries.append(row);log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            i5,i6=r5['selected_entry'],r6['selected_entry'];e5,e6=r5['entries'][i5],r6['entries'][i6]
            c5=e5['connection']['candidates'][r5['selected_candidate']] if r5['selected_candidate'] is not None else None
            c6=e6['connection']['candidates'][r6['selected_candidate']] if r6['selected_candidate'] is not None else None
            # Image selection changes are real outcomes; comparisons of original
            # measurement use the H5 entry matching the selected H6 condition.
            m5=r5['entries'][i6]
            row=dict(id=ident,H5=measure(e5,c5,'H5'),H6=measure(e6,c6,'H6'),selected_entry_H5=i5,selected_entry_H6=i6,
                full_observation_comparison=same_full_observation(m5,e6),selected_entry_changed=i5!=i6)
            images.append(row);sources.append(dict(id=ident,H5_record_sha256=sha(paths[0]),H6_record_sha256=sha(paths[1])))
    visuals=[]
    for label,subset in [('固定16开发图_原图_H5_H6',set(cfg['pilot_ids'])),('冻结后8图_原图_H5_H6',set(cfg['extra_ids']))]:
        folder=out/label;folder.mkdir()
        for ident in sorted(set(ids)&subset):visuals.append(visual(oldrun,newrun,ident,folder/(ident+'.jpg')))
    report=dict(command=sys.argv,comparison_source_sha256=sha(__file__),helper_source_sha256=HELPER_SHA,input_protocol_sha256=sha(HERE/'inputs.json'),settings=SETTINGS,
        H5_run=str(oldrun),H6_run=str(newrun),H5_summary_sha256=sha(oldrun/'summary.json'),H6_summary_sha256=sha(newrun/'summary.json'),
        by_image=aggregate(images),by_entry=aggregate(entries),development16=aggregate([r for r in images if r['id'] in cfg['pilot_ids']]),extra8_postfreeze=aggregate([r for r in images if r['id'] in cfg['extra_ids']]),
        selected_entry_changed=sum(r['selected_entry_changed'] for r in images),fixed_visual_count=len(visuals),seconds=time.monotonic()-start,labels_read=False,new_inference=False,
        claims='Topology-constrained and shorter do not imply correct target identity. Removed tails include measured original-image-supported pixels and must not all be called unknown. Source F3 is a constraint, not truth; missing/absent connections remain failures to resolve attachment.')
    save(out/'summary.json',report);save(out/'paired_images.json',images);save(out/'sources.json',sources);save(out/'visuals.json',visuals)
    print(json.dumps(dict(output=str(out),pairs=report['by_image']['pairs'],entries=report['by_entry']['pairs'],visuals=len(visuals),seconds=report['seconds'],comparison_source_sha256=report['comparison_source_sha256']),ensure_ascii=True))

if __name__=='__main__':main()
