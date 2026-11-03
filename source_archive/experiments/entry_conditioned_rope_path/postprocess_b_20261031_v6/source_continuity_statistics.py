"""Read-only statistics of already saved H6 source rows and center tracks.

This does not load images, labels, model weights, masks or algorithm modules.
It must never connect separate pieces or classify a track as the target rope.
Full mode requires a completed run and processes its frozen ID denominator.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE=Path(__file__).resolve().parent
KNOWN_STATES=('observed','inferred','inferred_gap')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read_record(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def measure_piece(points,statuses):
    p=np.asarray(points,float).reshape(-1,2)
    if len(statuses)!=max(0,len(p)-1):raise ValueError('Authoritative segment status is not aligned')
    if not np.isfinite(p).all():raise ValueError('Nonfinite saved geometry')
    arc=np.linalg.norm(np.diff(p,axis=0),axis=1)*2.
    counts=Counter(statuses);lengths=defaultdict(float);runs=Counter();max_run=defaultdict(float)
    last=None;running=0.
    for state,d in zip(statuses,arc):
        lengths[state]+=float(d)
        if state!=last:
            if last is not None:max_run[last]=max(max_run[last],running)
            runs[state]+=1;running=0.;last=state
        running+=float(d)
    if last is not None:max_run[last]=max(max_run[last],running)
    return dict(point_count=len(p),edge_count=len(arc),length_original_px=float(arc.sum()),
        edge_counts=dict(counts),state_lengths_original_px=dict(lengths),state_run_counts=dict(runs),
        maximum_state_run_original_px=dict(max_run))


def endpoint_source_observation(track,index,boundaries):
    """Describe source-end geometry; it does not infer the unpublished stop gate."""
    observations=[]
    for side,key in enumerate(('upper_boundary_xy','lower_boundary_xy')):
        bid=track['boundary_ids'][side];point=np.asarray(track[key][index],float)
        boundary=boundaries.get(bid)
        if boundary is None:raise ValueError('Track references a missing boundary')
        candidate=[]
        for piece in boundary.get('pieces',[]):
            q=np.asarray(piece['points_xy'],float)
            if len(q)<2:continue
            lo,hi=float(q[:,0].min()),float(q[:,0].max())
            if lo-1.e-5<=point[0]<=hi+1.e-5:
                residual=abs(float(np.interp(point[0],q[:,0],q[:,1]))-point[1])*2
                candidate.append((residual,lo,hi))
        if not candidate:
            observations.append('no_single_source_piece_spans_endpoint_x');continue
        residual,lo,hi=min(candidate)
        if residual>3.:observations.append('saved_piece_interpolation_differs_over3_original_px');continue
        # Output decimation may place the last center sample a few pixels
        # before its source endpoint. This is a geometric proximity tolerance.
        near=(point[0]-lo if index==0 else hi-point[0])*2<=6.0001
        observations.append('source_piece_endpoint_within6_original_px' if near else 'source_piece_continues_beyond_center_endpoint')
    endings=observations.count('source_piece_endpoint_within6_original_px')
    if endings==2:summary='both_source_pieces_end_here'
    elif endings==1:summary='one_source_piece_ends_here'
    elif all(s=='source_piece_continues_beyond_center_endpoint' for s in observations):summary='source_pieces_continue_but_center_pair_stops'
    else:summary='endpoint_source_geometry_not_resolved_by_saved_fields'
    return summary,observations


def image_statistics(record):
    result=record['surface_tracks'];tracks=result['tracks'];boundaries=result['boundaries'];diag=result['diagnostics']
    by_boundary={b['boundary_id']:b for b in boundaries}
    if len(by_boundary)!=len(boundaries):raise ValueError('Duplicate boundary lineage ID')
    track_ids=[t['track_id'] for t in tracks]
    if len(set(track_ids))!=len(track_ids):raise ValueError('Duplicate center track ID')
    state_counts=Counter();state_runs=Counter();state_lengths=defaultdict(float);max_runs=defaultdict(float)
    source_state_counts=Counter();source_state_lengths=defaultdict(float);source_max_runs=defaultdict(float)
    endpoints=Counter();stop_reasons=Counter();source_gaps=[];track_rows=[];sources=set()
    for b in boundaries:
        sources.update(b.get('source_ids',[]));pieces=b.get('pieces',[])
        for piece in pieces:
            m=measure_piece(piece['points_xy'],piece['segment_status'])
            source_state_counts.update(m['edge_counts'])
            for k,v in m['state_lengths_original_px'].items():source_state_lengths[k]+=v
            for k,v in m['maximum_state_run_original_px'].items():source_max_runs[k]=max(source_max_runs[k],v)
        for before,after in zip(pieces[:-1],pieces[1:]):
            a=np.asarray(before['points_xy'][-1]);bpoint=np.asarray(after['points_xy'][0])
            source_gaps.append(dict(boundary_id=b['boundary_id'],source_lineage_id=b.get('source_lineage_id'),
                start_original_xy=(a*2+[0,1050]).tolist(),end_original_xy=(bpoint*2+[0,1050]).tolist(),
                chord_original_px=float(np.linalg.norm(bpoint-a)*2),
                status='unconnected_saved_source_pieces',filled=False,
                reason='saved_geometry_has_no_accepted_continuous_piece; no_extrapolation',
                physical_gap_length_not_measured=True))
    midpoint_max=0.
    for t in tracks:
        m=measure_piece(t['points_xy'],t['segment_status']);state_counts.update(m['edge_counts']);state_runs.update(m['state_run_counts'])
        for k,v in m['state_lengths_original_px'].items():state_lengths[k]+=v
        for k,v in m['maximum_state_run_original_px'].items():max_runs[k]=max(max_runs[k],v)
        mid=(np.asarray(t['upper_boundary_xy'])+np.asarray(t['lower_boundary_xy']))*.5
        midpoint_max=max(midpoint_max,float(np.max(np.abs(np.asarray(t['points_xy'])-mid)))*2)
        start,start_detail=endpoint_source_observation(t,0,by_boundary);end,end_detail=endpoint_source_observation(t,-1,by_boundary)
        endpoints[start]+=1;endpoints[end]+=1;stop_reasons[t['stop_reason']]+=1
        track_rows.append(dict(track_id=t['track_id'],boundary_ids=t['boundary_ids'],measurement=m,
            start_source_observation=start,end_source_observation=end,start_detail=start_detail,end_detail=end_detail,
            saved_stop_reason=t['stop_reason']))
    unknown_states=(set(state_counts)|set(source_state_counts))-set(KNOWN_STATES)
    if unknown_states:raise ValueError('Unrecognized saved source states: '+str(unknown_states))
    if midpoint_max>1.e-6:raise ValueError('Center is not the exact saved boundary midpoint')
    receipts=diag.get('saved_source_gap_receipts',[])
    correspondence_reasons=Counter(x['basis'] for x in diag.get('source_correspondences',[]))
    return dict(id=record['id'],source_piece_count=diag['source_piece_count'],
        boundary_lineage_count=len(boundaries),unique_source_piece_ids=len(sources),center_track_count=len(tracks),
        boundary_piece_count=sum(len(b['pieces']) for b in boundaries),
        center_edge_counts=dict(state_counts),center_state_run_counts=dict(state_runs),
        center_state_lengths_original_px=dict(state_lengths),center_maximum_state_run_original_px=dict(max_runs),
        source_edge_counts=dict(source_state_counts),source_state_lengths_original_px=dict(source_state_lengths),source_maximum_state_run_original_px=dict(source_max_runs),
        maximum_center_track_length_original_px=max((t['measurement']['length_original_px'] for t in track_rows),default=0.),
        source_gap_receipt_count=len(receipts),maximum_source_gap_receipt_original_px=max((g['length_original_px'] for g in receipts),default=0.),
        unconnected_source_piece_gap_count=len(source_gaps),unconnected_source_gaps=source_gaps,
        center_endpoint_source_observations=dict(endpoints),saved_stop_reason_counts=dict(stop_reasons),
        source_correspondence_basis_counts=dict(correspondence_reasons),
        midpoint_maximum_error_original_px=midpoint_max,
        extrapolated_points=diag.get('extrapolated_points'),fixed_spacing_lines_generated=diag.get('fixed_spacing_lines_generated'),
        candidate_full_D4_curves_adopted=diag.get('candidate_full_D4_curves_adopted'),
        sparse_raster_CC_used_as_identity=diag.get('sparse_raster_CC_used_as_identity'),
        rejected_geometry_counts=diag.get('rejected',{}),tracks=track_rows)


def combine(cases):
    total=Counter();counts=Counter();runs=Counter();lengths=defaultdict(float);maximum=defaultdict(float);endpoints=Counter();reasons=Counter();basis=Counter()
    for c in cases:
        for k in ('source_piece_count','boundary_lineage_count','unique_source_piece_ids','center_track_count','boundary_piece_count','source_gap_receipt_count','unconnected_source_piece_gap_count'):total[k]+=c[k]
        counts.update(c['center_edge_counts']);runs.update(c['center_state_run_counts']);endpoints.update(c['center_endpoint_source_observations']);reasons.update(c['saved_stop_reason_counts']);basis.update(c['source_correspondence_basis_counts'])
        for k,v in c['center_state_lengths_original_px'].items():lengths[k]+=v
        for k,v in c['center_maximum_state_run_original_px'].items():maximum[k]=max(maximum[k],v)
    return dict(images=len(cases),summed_per_image_counts=dict(total),center_edge_counts=dict(counts),
        center_state_run_counts=dict(runs),center_state_lengths_original_px=dict(lengths),
        maximum_contiguous_center_state_run_original_px=dict(maximum),
        maximum_source_gap_receipt_original_px=max((c['maximum_source_gap_receipt_original_px'] for c in cases),default=0.),
        center_endpoint_source_observations=dict(endpoints),saved_stop_reason_counts=dict(reasons),source_correspondence_basis_counts=dict(basis),
        cases_with_inferred_gap=sum(c['center_state_lengths_original_px'].get('inferred_gap',0)>0 for c in cases),
        cases_with_unconnected_source_pieces=sum(c['unconnected_source_piece_gap_count']>0 for c in cases),
        no_fixed_spacing_or_extrapolation_all=all(c['extrapolated_points']==0 and c['fixed_spacing_lines_generated']==0 for c in cases),
        no_unaccepted_full_D4_pool_adopted_all=all(c['candidate_full_D4_curves_adopted'] is False for c in cases),
        maximum_midpoint_error_original_px=max((c['midpoint_maximum_error_original_px'] for c in cases),default=0.))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='all_001');ap.add_argument('--output',default='source_continuity_all_001');ap.add_argument('--schema-only',action='store_true');args=ap.parse_args()
    run=(HERE/args.run).resolve();out=(HERE/args.output).resolve()
    if run.parent!=HERE or out.parent!=HERE:raise ValueError('Use direct H6 run/output children')
    if out.exists():raise FileExistsError(out)
    start=load(run/'run_start.json');source_receipts=start['sources']|start['reused_sources']
    if not all(sha(p)==h for p,h in source_receipts.items()):raise ValueError('Frozen source changed')
    summary_path=run/'summary.json'
    ids=start['input_ids']
    if args.schema_only:
        progress=[json.loads(line) for line in (run/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        ids=[progress[0]['id']]
    else:
        if not summary_path.exists():raise RuntimeError('Full run is not complete; do not report a partial denominator')
        summary=load(summary_path)
        if summary['exit_code']!=0 or summary['completed']!=summary['planned'] or summary['completed']!=len(ids) or summary['failures']:raise ValueError('Full run did not complete without failures')
    tick=time.perf_counter();out.mkdir();rows=[];records=[]
    for ident in ids:
        p=run/'记录'/(ident+'.json.gz');record=read_record(p)
        if record['id']!=ident:raise ValueError('Record ID mismatch')
        case=image_statistics(record);rows.append(case);records.append(dict(id=ident,path=str(p),sha256=sha(p)))
    assert all(sha(p)==h for p,h in source_receipts.items())
    report=dict(status='schema_check_only' if args.schema_only else 'completed_full_geometry_statistics',
        run=str(run),source_script_sha256=sha(__file__),frozen_source_receipts=source_receipts,records=records,
        run_start_sha256=sha(run/'run_start.json'),run_summary_sha256=sha(summary_path) if summary_path.exists() else None,
        denominator=len(ids),aggregate=combine(rows),cases=rows,elapsed_seconds=time.perf_counter()-tick,exit_code=0,
        scope='Only saved source/center geometry; no image, mask, annotation, model or algorithm module was loaded.',
        interpretation=['Observed means the frozen F3 vector is backed by its saved final boundary raster; it is not independent annotation or material identity.',
            'Inferred and inferred_gap retain separate segment counts and arc lengths. Different source pieces are never joined for these statistics.',
            'Source lineage may include multiple source IDs only through frozen F2/F3 visual correspondences; not through new nearest-row assignment.',
            'The saved stop_reason is composite. Endpoint observations only describe which source pieces end; the exact unpublished rejection gate cannot be reconstructed.',
            'Gap chord is an endpoint separation, not measured hidden rope length. Many images share acquisition sources, so summed tracks/pixels are not independent samples.',
            'These are geometric inventory statistics, not target identity accuracy or proof of reaching the correct spool edge.'],
        labels_read=False,images_read=False,model_calls=0,core_changed=False,devices=False)
    dump(out/'statistics.json',report)
    dump(out/'summary.json',{k:v for k,v in report.items() if k not in ('cases','records','frozen_source_receipts')})
    print(json.dumps(report['aggregate'],ensure_ascii=False))


if __name__=='__main__':main()
