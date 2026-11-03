"""POSTHOC ONLY: original V24 probability-node support, not better predictions.

GT never creates nodes, changes probabilities, or writes into actual exports.
The relaxed graph is an optimistic bound for an always-GT-compatible PREFIX,
not an upper bound on later coverage after a wrong-path detour/recovery.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from .v11_path_contract import HEIGHT, WIDTH, STEP_WIDTHS, image_to_strip, strip_to_image, _checked_geometry
from .v11_path_decode import _column_nodes, DECODE_PARAMETERS

LAB=Path(__file__).resolve().parent
OLD=LAB/'outputs/v11_path_identity'
ARMS=('baseline','control','anchor_mix')
VARIANTS=('manual','auto')
TOLERANCE_PX=5.


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def save(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def relaxed_prefix(nodes, target_rows, trusted, native_step, anchor):
    """Exact finite reachability in a deliberately RELAXED original-node graph.

    Curvature penalties, rewards, pruning, predicted missing-point trajectory
    and learned slope are ignored. At most two missing columns and <=1.2 rows
    per column remain necessary constraints. Unknown GT columns allow all nodes
    but are not scored. Nodes at trusted columns must be within5 native px.
    """
    reachable=[np.zeros(len(v),bool) for v in nodes]
    for col in range(anchor,len(nodes)):
        for k,node in enumerate(nodes[col]):
            row=float(node['row'])
            if trusted[col] and abs(row-target_rows[col])*native_step>TOLERANCE_PX:
                continue
            if col==anchor:
                reachable[col][k]=abs(row-127.5)<=.75/STEP_WIDTHS
                continue
            for delta in (1,2,3):
                prev=col-delta
                if prev<anchor:continue
                if any(reachable[prev][j] and abs(row-old['row'])<=1.2*delta+1e-10
                       for j,old in enumerate(nodes[prev])):
                    reachable[col][k]=True;break
    return np.array([r.any() for r in reachable]),reachable


def decoded_column_rows(segments,geometry):
    """Same-column intersections of saved observed segments; never bridge gaps."""
    output=[[] for _ in range(WIDTH)]
    for segment in segments:
        points=image_to_strip(np.asarray(segment,float),geometry)
        for a,b in zip(points[:-1],points[1:]):
            dx=b[0]-a[0]
            if abs(dx)<1e-8:continue
            lo=max(0,int(math.ceil(min(a[0],b[0])-1e-6)))
            hi=min(WIDTH-1,int(math.floor(max(a[0],b[0])+1e-6)))
            for col in range(lo,hi+1):
                fraction=(col-a[0])/dx
                output[col].append(float(a[1]+fraction*(b[1]-a[1])))
    return output


def audit(probability,valid_mask,geometry,target_rows,trusted_columns,saved_segments,scopes=None):
    _checked_geometry(geometry)
    p=np.asarray(probability,float);valid=np.asarray(valid_mask,bool)
    targets=np.asarray(target_rows,float);trusted=np.asarray(trusted_columns,bool)
    if p.shape!=(HEIGHT,WIDTH) or valid.shape!=p.shape or targets.shape!=(WIDTH,) or trusted.shape!=(WIDTH,):
        raise ValueError('Invalid audit shapes')
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or np.any(p.sum(axis=0)>1+1e-4):
        raise ValueError('Invalid original conditional probabilities')
    if not np.isfinite(targets[trusted]).all():raise ValueError('Nonfinite trusted GT')
    native=float(geometry['native_px_per_row'])
    radius=max(1,int(math.ceil(2*max(1.,2/native))))
    anchor=int(round((-2.-(-8.))/STEP_WIDTHS-.5))
    nodes=[_column_nodes(p,valid,c,radius) for c in range(WIDTH)]
    node_good=np.array([any(abs(n['row']-targets[c])*native<=TOLERANCE_PX for n in values)
                        if trusted[c] else False for c,values in enumerate(nodes)])
    reachable,_=relaxed_prefix(nodes,targets,trusted,native,anchor)
    raw_good=np.abs(p.argmax(axis=0)-targets)*native<=TOLERANCE_PX
    mass=np.sum(np.where(np.abs(np.arange(HEIGHT)[:,None]-targets[None])*native<=TOLERANCE_PX,p,0),axis=0)
    actual=decoded_column_rows(saved_segments,geometry)
    saved_good=np.array([any(abs(r-targets[c])*native<=TOLERANCE_PX for r in values)
                         if trusted[c] else False for c,values in enumerate(actual)])
    saved_present=np.array([bool(v) for v in actual])
    all_scopes={'all_trusted':np.ones(WIDTH,bool)}
    if scopes:all_scopes.update(scopes)
    summary={}
    for name,scope in all_scopes.items():
        m=trusted&np.asarray(scope,bool);total=int(m.sum())
        values=dict(trusted_columns=total,raw_argmax_hit=int((m&raw_good).sum()),
            original_node_hit=int((m&node_good).sum()),
            gt_compatible_relaxed_prefix_reachable=int((m&reachable&node_good).sum()),
            saved_decoder_hit=int((m&saved_good).sum()),saved_decoder_present=int((m&saved_present).sum()),
            no_correct_original_node=int((m&~node_good).sum()),
            raw_hit_but_no_correct_node=int((m&raw_good&~node_good).sum()),
            correct_node_but_saved_miss=int((m&node_good&~saved_good).sum()),
            reachable_correct_node_but_saved_miss=int((m&node_good&reachable&~saved_good).sum()))
        values['fractions']={key:(value/total if total else None) for key,value in values.items() if key!='trusted_columns'}
        summary[name]=values
    rows=[dict(column=c,target_row=float(targets[c]),raw_argmax_row=int(p[:,c].argmax()),
        target5px_mass=float(mass[c]),node_count=len(nodes[c]),correct_node_present=bool(node_good[c]),
        gt_compatible_relaxed_prefix_reachable=bool(reachable[c]&node_good[c]),
        saved_hit=bool(saved_good[c]),raw_hit=bool(raw_good[c]),
        original_nodes=[dict(row=float(n['row']),mass=float(n['mass'])) for n in nodes[c]]) for c in np.flatnonzero(trusted)]
    for row in rows:row['column']=int(row['column'])
    return dict(summary=summary,per_column=rows,anchor_column=anchor,radius_rows=radius,
        nodes_injected=False,probabilities_modified=False,actual_predictions_modified=False,
        is_prediction=False,GT_used_posthoc_only=True,identity_confirmed=False,
        warning='Reachability ignores curvature/score/pruning and permits unscored GT-unknown intervals; '
        'it is only an optimistic bound for a GT-compatible prefix. Higher reachability does NOT prove a beam bug.')


def freeze(base,out):
    if out.exists():raise FileExistsError(out)
    paths=[Path(__file__),LAB/'tests/test_v24_candidate_support_audit.py',
        LAB/'v11_path_decode.py',LAB/'v11_path_contract.py',LAB/'v23_path_prefix_audit.py',
        LAB/'centerline_path_metrics_v6.py',LAB/'evaluate_centerline_multiscale_v9.py',
        LAB/'centerline_quality_v5.py',LAB/'centerline_quality_cutoff_v8.py',
        LAB/'centerline_review_v5.py',LAB/'prepare_centerline_v5.py',
        base/'protocol.json',base/'cache/input_index.json',OLD/'development_manifest.json',OLD/'development_cache/index.json']
    write(out/'protocol.json',dict(schema='v24_posthoc_candidate_support_1',base=str(base),
        arms=list(ARMS),variants=list(VARIANTS),split='holdout',tolerance_native_px=TOLERANCE_PX,
        decoder_parameters=DECODE_PARAMETERS,
        layers=['raw response','unchanged _column_nodes','GT-compatible relaxed prefix','saved actual segments'],
        departure='original human free-seed half-width departure, original optical width, trusted GT arc +-2width; not physical contact',
        restrictions='No new predictions, training, thresholds, GT node injection or real-time use',
        hashes={str(p.resolve()):sha(p) for p in paths},exports_read_at_freeze=False))


def verify(out):
    p=read(out/'protocol.json')
    for path,h in p['hashes'].items():
        if sha(path)!=h:raise ValueError('Audit snapshot changed: '+path)
    return p


def _arc(points,reference):
    line=np.asarray(reference,float);cum=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]
    best=np.full(len(points),np.inf);arc=np.full(len(points),np.nan)
    for k,(a,b) in enumerate(zip(line[:-1],line[1:])):
        delta=b-a;norm=float(delta@delta)
        if norm<1e-12:continue
        t=np.clip((points-a)@delta/norm,0,1);dist=np.linalg.norm(points-a-t[:,None]*delta,axis=1)
        keep=dist<best;arc[keep]=cum[k]+t[keep]*math.sqrt(norm);best[keep]=dist[keep]
    return arc


def prepare(out):
    p=verify(out);base=Path(p['base'])
    if (out/'targets/index.json').exists():raise FileExistsError('No overwrite')
    from .evaluate_centerline_multiscale_v9 import _dataset,scorable_reference
    from .v11_path_contract import build_targets
    from .centerline_path_metrics_v6 import clip_scoring_segments,_seed_geometry
    from .v23_path_prefix_audit import seed_plane_origin,departure_diagnostic
    _,records,_=_dataset(OLD/'development_manifest.json');by_id={r['sample']['sample_id']:r for r in records}
    manual={r['sample_id']:r for r in read(OLD/'development_cache/index.json')['entries']}
    inputs=read(base/'cache/input_index.json');rows=[]
    for key,e in inputs['entries'].items():
        if e['split']!='holdout' or e['variant'] not in VARIANTS:continue
        record=by_id[e['sample_id']];sample=record['sample'];label=record['label'];q=record['quality']
        target_meta=dict(key=key,sample_id=e['sample_id'],variant=e['variant'],available=e['available'],
            annotation_sha256=sha(sample['annotation_path']),quality_sha256=q['quality_sha256'],
            reason=e.get('reason'),input_sha256=e.get('input_sha256'))
        if not e['available']:
            rows.append(target_meta);continue
        if sha(e['input_path'])!=e['input_sha256']:raise ValueError('Input changed')
        a=load(e['input_path']);reference,_=scorable_reference(label,q)
        ignored=label['ignore_polygons']+q['precision_exclude_polygons']
        t=build_targets(e['geometry'],reference,ignored,a['valid_mask'])
        target=(t['target_distribution']*np.arange(HEIGHT)[:,None]).sum(axis=0)
        xy=strip_to_image(np.stack((np.arange(WIDTH),target),axis=1),e['geometry'])
        seed=label['polylines']['free_seed'];origin,direction=_seed_geometry(seed)
        downstream=(xy-origin)@direction>=0
        ref_segments=clip_scoring_segments([reference],free_seed=seed,downstream=True,ignore_polygons=ignored)
        arc_origin,_=seed_plane_origin(reference,seed);width=manual[e['sample_id']]['width']['width_px']
        departure=departure_diagnostic(reference,seed,width,ref_segments,[],arc_origin)
        near=np.zeros(WIDTH,bool)
        if 'point_xy' in departure:
            absolute_departure=departure['arc_from_seed_plane_px']+arc_origin
            near=np.abs(_arc(xy,reference)-absolute_departure)<=2*width
        path=out/'targets'/f'{len(rows):02d}.npz'
        save(path,target_rows=target,trusted_columns=t['column_valid'],downstream=downstream,departure_neighborhood=near)
        target_meta.update(path=str(path),sha256=sha(path),geometry=e['geometry'],GT_departure=departure,
            trustworthy_columns=int(t['column_valid'].sum()))
        rows.append(target_meta)
    if len(rows)!=16:raise ValueError('Expected8development images x manual/auto')
    write(out/'targets/index.json',dict(rows=rows,audit_protocol_sha256=sha(out/'protocol.json'),
        input_index_sha256=sha(base/'cache/input_index.json'),GT_never_used_in_inference=True))
    verify(out)


def run(out):
    p=verify(out);base=Path(p['base']);targets=read(out/'targets/index.json')
    if targets['audit_protocol_sha256']!=sha(out/'protocol.json'):raise ValueError('Target protocol mismatch')
    if (out/'report.json').exists():raise FileExistsError('No overwrite')
    inputs=read(base/'cache/input_index.json');all_rows=[];source_exports={}
    for arm in ARMS:
        ipath=base/'exports'/arm/'index.json';index=read(ipath)
        if index['protocol_sha256']!=sha(base/'protocol.json') or index['input_index_sha256']!=sha(base/'cache/input_index.json'):
            raise ValueError('Export provenance mismatch')
        source_exports[arm]=dict(index_sha256=sha(ipath),model_sha256=index['model_sha256'])
        by_key={r['key']:r for r in index['rows']}
        for target in targets['rows']:
            key=target['key'];actual=by_key[key];entry=inputs['entries'][key]
            if actual.get('input_sha256')!=entry.get('input_sha256'):raise ValueError('Mismatched inference input')
            row=dict(arm=arm,key=key,sample_id=target['sample_id'],variant=target['variant'],available=target['available'])
            if not target['available']:
                row.update(reason=target.get('reason'),summary=None);all_rows.append(row);continue
            for path,h in ((entry['input_path'],entry['input_sha256']),(actual['probability_path'],actual['probability_sha256']),(target['path'],target['sha256'])):
                if sha(path)!=h:raise ValueError('Audit input changed')
            a=load(entry['input_path']);pred=load(actual['probability_path']);t=load(target['path'])
            decoded=actual['decoded']
            if isinstance(decoded,str):decoded=read(decoded)
            result=audit(pred['probability'],a['valid_mask'],entry['geometry'],t['target_rows'],t['trusted_columns'],
                decoded['segments'],{'manual_seed_downstream':t['downstream'],'GT_departure_neighborhood':t['departure_neighborhood']})
            detail=out/'details'/arm/f'{targets["rows"].index(target):02d}.json';write(detail,result)
            row.update(summary=result['summary'],detail_path=str(detail),detail_sha256=sha(detail),
                prediction_sha256=actual['probability_sha256'],GT_departure=target['GT_departure'])
            all_rows.append(row)
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for variant in VARIANTS:
            chosen=[r for r in all_rows if r['arm']==arm and r['variant']==variant]
            scopes={}
            for scope in ('all_trusted','manual_seed_downstream','GT_departure_neighborhood'):
                data=[r['summary'][scope] for r in chosen if r['available']]
                keys=[k for k in data[0] if k!='fractions'] if data else []
                summed={k:sum(d[k] for d in data) for k in keys};n=summed.get('trusted_columns',0)
                summed['fractions']={k:(v/n if n else None) for k,v in summed.items() if k!='trusted_columns'}
                scopes[scope]=summed
            summary[arm][variant]=dict(samples=len(chosen),available=sum(r['available'] for r in chosen),scopes=scopes,
                unavailable_not_success=True,denominator_warning='column diagnostics conditional on available input, not original arc-length total')
    write(out/'report.json',dict(schema='v24_candidate_support_audit_result_1',summary=summary,rows=all_rows,
        protocol_sha256=sha(out/'protocol.json'),target_index_sha256=sha(out/'targets/index.json'),exports=source_exports,
        is_prediction=False,GT_used_posthoc_only=True,original_exports_modified=False,identity_confirmed=False,
        important='GT-compatible relaxed prefix is not a total-coverage upper bound and not a real decoder. '
        'A correct node without actual hit suggests an audit opportunity, not proof the beam is faulty.'))
    verify(out);print(json.dumps(dict(report=str(out/'report.json'),rows=len(all_rows))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','prepare','run','verify'))
    parser.add_argument('--base',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='freeze':
        if args.base is None:parser.error('--base required for freeze')
        freeze(args.base.resolve(),args.out.resolve())
    else:{'prepare':prepare,'run':run,'verify':verify}[args.command](args.out.resolve())
