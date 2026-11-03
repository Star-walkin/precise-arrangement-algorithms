"""Post-hoc V11 prefix coverage audit. Saved inference/denominators unchanged.

The seed endpoint plane is a label-coordinate origin, NOT a physical contact.
GT is used only to partition already scored length, never to crop inference.
"""
from __future__ import annotations
import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .v11_path_experiment import BASE, verify, load_cache, read, sha
from .evaluate_centerline_multiscale_v9 import _dataset, scorable_reference, score_decoded
from .centerline_path_metrics_v6 import _sampling, _nearest, _line, _seed_geometry
from .v11_path_contract import image_to_strip
from .v18_front_diagnostics import font

OUT=Path(__file__).resolve().parent/'outputs/v23_path_prefix_audit'
BINS=((0.,4.),(4.,8.),(8.,16.),(16.,32.),(32.,None))


def project_arc(points, reference):
    """Nearest projection on the existing GT polyline; no gap bridge added."""
    line=_line(reference)
    cumulative=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]
    best=np.full(len(points),np.inf);positions=np.full(len(points),np.nan)
    for k,(a,b) in enumerate(zip(line[:-1],line[1:])):
        delta=b-a;length=np.linalg.norm(delta)
        t=np.clip((points-a)@delta/(length*length),0,1)
        distance=np.linalg.norm(points-(a+t[:,None]*delta),axis=1)
        keep=distance<best
        best[keep]=distance[keep];positions[keep]=cumulative[k]+t[keep]*length
    if len(points) and np.max(best)>1e-5:raise ValueError('scored support left reference geometry')
    return positions


def seed_plane_origin(reference, seed):
    line=_line(reference);origin,direction=_seed_geometry(seed)
    if len(line)<2:return None,'no_trusted_reference'
    u=(line-origin)@direction
    if np.any(np.diff(u)<-1e-6):return None,'nonmonotone_GT_order_unknown'
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]
    for i in range(len(line)-1):
        if u[i]<=0<=u[i+1] and u[i+1]-u[i]>1e-9:
            return float(lengths[i]+(-u[i])/(u[i+1]-u[i])*(lengths[i+1]-lengths[i])),'seed_endpoint_plane_intersection'
    return None,'trusted_GT_does_not_cross_seed_plane_origin_unknown'


def cell_overlap(edges, low, high):
    edges=np.asarray(edges,float)
    return np.maximum(0.,np.minimum(edges[1:],high)-np.maximum(edges[:-1],low))


def audit_coverage(reference, seed, width, reference_segments, prediction_segments,
                   geometry=None, valid_mask=None):
    pieces=_sampling(reference_segments)
    reference_length=float(sum(p['length'] for p in pieces))
    zero=[dict(range_widths=[lo,hi],reference_length_px=0.,covered_reference_length_px=0.,
               input_supported_reference_length_px=0.) for lo,hi in BINS]
    origin,origin_reason=seed_plane_origin(reference,seed)
    usable_width=width is not None and np.isfinite(width) and width>0
    result=dict(origin_arc_px=origin,origin_reason=origin_reason,width_px=width,
        full_downstream_reference_length_px=reference_length,bins=zero,
        unknown_binned_reference_length_px=reference_length,
        evaluated_precision=False,physical_contact_identified=False)
    if origin is None or not usable_width:
        for row in zero:row.update(status='unknown_origin_or_width',coverage_5px=None,input_support_fraction=None)
        return result
    retained=[]
    for piece in pieces:
        start=float(project_arc(piece['vertices'][:1],reference)[0]-origin)
        end=float(project_arc(piece['vertices'][-1:],reference)[0]-origin)
        if abs((end-start)-piece['length'])>1e-5:raise ValueError('reference gap compressed or ordering changed')
        cells=piece['cell_edges']+start
        distances,_=_nearest(piece['points'],prediction_segments)
        matched=distances<=5.
        support=np.zeros(len(distances),bool)
        if geometry is not None and valid_mask is not None:
            uv=image_to_strip(piece['points'],geometry)
            inside=(uv[:,0]>=0)&(uv[:,0]<valid_mask.shape[1]-1)&(uv[:,1]>=0)&(uv[:,1]<valid_mask.shape[0]-1)
            values=cv2.remap(valid_mask.astype(np.float32),uv[:,0].astype(np.float32)[None],uv[:,1].astype(np.float32)[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[0]
            support=inside&(values>=1-1e-6)
        retained.append(dict(start_arc_px=start,end_arc_px=end,vertices=piece['vertices'].tolist()))
        for row,(low,high) in zip(zero,BINS):
            weight=cell_overlap(cells,low*width,np.inf if high is None else high*width)
            row['reference_length_px']+=float(weight.sum())
            row['covered_reference_length_px']+=float(weight[matched].sum())
            row['input_supported_reference_length_px']+=float(weight[support].sum())
    for row in zero:
        n=row['reference_length_px']
        row.update(status='scored_GT_support' if n else 'unknown_no_trusted_GT_in_bin',
            coverage_5px=row['covered_reference_length_px']/n if n else None,
            input_support_fraction=row['input_supported_reference_length_px']/n if n and valid_mask is not None else None)
    unknown=reference_length-sum(row['reference_length_px'] for row in zero)
    if abs(unknown)>1e-5:raise ValueError('prefix bins changed original reference denominator')
    result.update(unknown_binned_reference_length_px=0.,trusted_arc_intervals=retained)
    return result


def departure_diagnostic(reference,seed,width,reference_segments,prediction_segments,origin):
    if origin is None or width is None:return dict(status='unknown_origin_or_width')
    seed_origin,d=_seed_geometry(seed);normal=np.array([-d[1],d[0]])
    candidates=[]
    for segment in reference_segments:
        line=np.asarray(segment,float)
        for a,b in zip(line[:-1],line[1:]):
            first,last=(np.array([a,b])-seed_origin)@normal
            fraction=None
            if abs(first)>=width*.5:fraction=0.
            elif abs(last)>=width*.5 and abs(last-first)>1e-10:
                fraction=((width*.5 if last>0 else -width*.5)-first)/(last-first)
            if fraction is not None:
                point=a+fraction*(b-a)
                arc=float(project_arc(point[None],reference)[0]-origin)
                candidates.append((arc,point))
    if not candidates:return dict(status='no_half_width_departure_in_trusted_GT',is_contact=False)
    arc,point=min(candidates,key=lambda value:value[0]);distance=float(_nearest(point[None],prediction_segments)[0][0])
    length=covered=0.
    for piece in _sampling(reference_segments):
        start=float(project_arc(piece['vertices'][:1],reference)[0]-origin)
        weight=cell_overlap(piece['cell_edges']+start,max(0.,arc-2*width),arc+2*width)
        matched=_nearest(piece['points'],prediction_segments)[0]<=5
        length+=float(weight.sum());covered+=float(weight[matched].sum())
    return dict(status='GT_interpolated_half_width_departure_only',point_xy=point.tolist(),
        arc_from_seed_plane_px=arc,arc_from_seed_plane_widths=arc/width,
        prediction_distance_px=distance if np.isfinite(distance) else None,prediction_hits_5px=bool(distance<=5),
        neighborhood_half_widths=2.,neighborhood_reference_length_px=length,
        neighborhood_coverage_5px=covered/length if length else None,
        is_contact=False,localization_is_GT_posthoc_only=True,used_to_crop_inference=False)


def render(rows,split):
    chosen=[r for r in rows if r['split']==split]
    image=Image.new('RGB',(1900,210+len(chosen)*72),'#151c25');draw=ImageDraw.Draw(image)
    label='训练10张（拟合诊断）' if split=='train' else '开发留出8张（已用于迭代，不是盲测）'
    draw.text((20,10),f'V23 已有V11轨迹前缀审计 | {label}',font=font(28),fill='white')
    draw.text((20,54),'原始5px覆盖按可信GT弧长分桶；每格=覆盖率 / GT像素长度。保留旧总分母，无标注=未知。',font=font(23),fill='#ffd388')
    draw.text((20,91),'起点=free_seed末点法向平面，不是物理接触；蓝色仅几何覆盖，不证明活动绳身份或换层。',font=font(23),fill='#d0dbe5')
    for j,text in enumerate(('0–4宽','4–8宽','8–16宽','16–32宽','32+宽')):
        draw.text((770+j*176,153),text,font=font(23),fill='white')
    draw.text((1670,153),'GT偏离处命中',font=font(20),fill='white')
    for i,row in enumerate(chosen):
        y=202+i*72
        draw.text((15,y+10),row['sample_id'],font=font(17),fill='white')
        for j,cell in enumerate(row['prefix']['bins']):
            x=760+j*176;value=cell['coverage_5px']
            color=(77,80,86) if value is None else (24,int(45+112*value),int(60+132*value))
            draw.rectangle((x,y,x+165,y+58),fill=color)
            text='未知 / 0px' if value is None else f"{100*value:.1f}% / {cell['reference_length_px']:.0f}px"
            draw.text((x+6,y+17),text,font=font(17),fill='white')
        turn=row['GT_departure'];hit=turn.get('prediction_hits_5px')
        text='未知/未偏离' if hit is None else ('是' if hit else '否')+f" ({turn['arc_from_seed_plane_widths']:.1f}宽)"
        draw.text((1670,y+16),text,font=font(19),fill='#ffd388')
    return image


def run():
    verify(BASE)
    index,cached=load_cache(BASE,development=True)
    _,records,_=_dataset(index['manifest'])
    if len(records)!=18 or sum(r['sample']['split']=='train' for r in records)!=10:raise ValueError('fixed 10+8 development rows required')
    run=BASE/'development';evaluation=read(run/'evaluation.json');exports=read(run/'exports.json')
    hashes={str(p.resolve()):sha(p) for p in (BASE/'protocol.json',run/'evaluation.json',run/'exports.json',run/'checkpoint.pt',run/'report.json',BASE/'development_cache/index.json',Path(index['manifest']))}
    if evaluation['exports_sha256']!=sha(run/'exports.json') or evaluation['checkpoint_sha256']!=sha(run/'checkpoint.pt'):raise ValueError('evaluation source mismatch')
    if exports['protocol_sha256']!=sha(BASE/'protocol.json') or exports['cache_index_sha256']!=sha(BASE/'development_cache/index.json'):raise ValueError('export source mismatch')
    if (evaluation['protocol_sha256']!=sha(BASE/'protocol.json') or exports['checkpoint_sha256']!=sha(run/'checkpoint.pt')
            or exports['training_report_sha256']!=sha(run/'report.json')):raise ValueError('frozen model/report binding mismatch')
    entries={e['sample_id']:(e,a) for e,a in cached};predictions={p['sample_id']:p for p in exports['records']};saved={r['sample_id']:r for r in evaluation['rows']}
    names={r['sample']['sample_id'] for r in records}
    if names!=set(entries) or names!=set(predictions) or names!=set(saved):raise ValueError('fixed sample inventory changed')
    rows=[]
    for record in records:
        sample,label,quality=record['sample'],record['label'],record['quality'];name=sample['sample_id']
        entry,arrays=entries[name];pred=predictions[name];old=saved[name]
        if (entry['annotation_sha256']!=record['annotation_sha256'] or entry['quality_sha256']!=quality['quality_sha256']
                or old['independent_width']!=entry['width']):raise ValueError('label quality or independent width changed')
        if pred['available']:
            if sha(pred['path'])!=pred['sha256'] or pred['input_sha256']!=entry['inference_input_sha256']:raise ValueError('prediction/input mismatch')
            hashes[pred['path']]=pred['sha256'];hashes[entry['cache']]=entry['cache_sha256']
        for key in ('image_path','annotation_path'):hashes[sample[key]]=sha(sample[key])
        quality_path=Path(sample['annotation_path']).with_suffix('.quality.json')
        if quality_path.exists():hashes[str(quality_path)]=sha(quality_path)
        reference,source=scorable_reference(label,quality)
        rescored=score_decoded(label,quality,old['decoded'],None,sample['image_shape_hw'])['metrics']
        oldmetrics=old['metrics']['downstream'];newmetrics=rescored['downstream']
        for key in ('reference_length_px','covered_reference_length_px'):
            if not np.isclose(oldmetrics[key],newmetrics[key],atol=1e-6):raise ValueError('saved original score changed')
        support=rescored['score_support'];width=entry['width']['width_px'];seed=label['polylines']['free_seed']
        prefix=audit_coverage(reference,seed,width,support['downstream_reference_segments'],support['downstream_prediction_segments'],entry.get('geometry'),arrays['valid_mask'] if arrays is not None else None)
        if prefix['origin_arc_px'] is not None and width is not None:
            if not np.isclose(sum(b['covered_reference_length_px'] for b in prefix['bins']),oldmetrics['covered_reference_length_px'],atol=1e-6):raise ValueError('bin coverage sum differs from frozen V6 score')
        turn=departure_diagnostic(reference,seed,width,support['downstream_reference_segments'],support['downstream_prediction_segments'],prefix['origin_arc_px'])
        rows.append(dict(sample_id=name,split=sample['split'],group_id=sample['group_id'],cohort=record['cohort'],
            reference_source=source,quality_sha256=quality['quality_sha256'],independent_width=entry['width'],
            prediction_status=old['prediction_status'],decoder_reason=old['decoded']['reason'],
            old_full_downstream_metrics=oldmetrics,prefix=prefix,GT_departure=turn))
    summaries=[]
    for split in ('train','holdout'):
        selected=[r for r in rows if r['split']==split];bins=[]
        for j,bounds in enumerate(BINS):
            cells=[r['prefix']['bins'][j] for r in selected];length=sum(c['reference_length_px'] for c in cells);covered=sum(c['covered_reference_length_px'] for c in cells)
            bins.append(dict(range_widths=bounds,total_samples=len(selected),samples_with_GT=sum(c['reference_length_px']>0 for c in cells),
                reference_length_px=length,covered_reference_length_px=covered,coverage_5px=covered/length if length else None))
        turns=[r['GT_departure'] for r in selected if 'prediction_hits_5px' in r['GT_departure']]
        summaries.append(dict(split=split,sample_count=len(selected),bins=bins,
            GT_departure_supported_samples=len(turns),GT_departure_prediction_hits=sum(t['prediction_hits_5px'] for t in turns)))
    verify(BASE)
    if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('audit input changed')
    output=OUT/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');output.mkdir(parents=True,exist_ok=False)
    figures={}
    for split in ('train','holdout'):
        p=output/(split+'_prefix_coverage.png');render(rows,split).save(p);figures[split]=dict(path=str(p),sha256=sha(p))
    report=dict(schema='v23_saved_v11_arc_prefix_diagnostics_1',script_sha256=sha(__file__),input_sha256=hashes,rows=rows,summaries=summaries,figures=figures,
        physical_contact_identified=False,identity_confirmed=False,events_evaluated=False,hardware_access=False,
        inference_repeated=False,model_changed=False,denominator_changed=False,blind_20260925_read=False,
        width_bins=[list(b) for b in BINS],quality_gap_policy='retain original arc offsets, no joining across excluded support; C07/C08 trusted C7 cutoff reused',
        precision_not_reported='Only GT-length coverage is partitioned; no prefix precision is inferred from later predicted segments.',
        origin_definition='Intersection of trusted GT with plane through human free_seed last point normal to first-last direction. Not physical contact.',
        metric_definition='Original V6 <=4px Voronoi arc cells and 5 native pixel nearest-path coverage. Cells split at fixed width-bin boundaries, identical global denominator.',
        limitations='GT interpolated centerline is not per-pixel or physical material proof; near-seed coverage can be easy straight free rope. Development holdout already consulted historically.')
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(output=str(output),summaries=summaries),ensure_ascii=False,indent=2));return output


if __name__=='__main__':run()
