"""Separate edge-pair identity continuity from contrast qualification.

Post-discovery statistical definition correction on the immutable V22 pair
table, NOT an independent blind experiment. No image processing, threshold
change, interpolation, truth labels or physical/material identity inference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import datetime
from pathlib import Path

import numpy as np

from vision_geometry_lab import v22_local_edges as original
from vision_geometry_lab import v19_front_trial as serialization


BASE = Path(__file__).resolve().parent/'outputs/v22_pair_continuity_audit'
RULES = dict(identity=['original_row_index','left_edge_track_id','right_edge_track_id','polarity'],
    continuation='same_identity_present_on_strictly_adjacent_original_source_frames',
    missing='no_interpolation_no_bridging',
    qualification='preserve_original_band_candidate_boolean; false_breaks_qualified_segment_only',
    minimum_observations=5,minimum_duration_s=.4,
    persistence='same_V22_observation_count_and_duration_definition',
    source_table_only=True,reextract_peaks=False,threshold_changes=False,
    study='post_discovery_separation_of_coupled_statistics_not_blind_validation',
    new_material_identity_confirmed=False,new_layer_confirmed=False,mechanical_phase_confirmed=False,
    ground_truth_used=False,device_access=False,deployment_allowed=False)
COLUMNS = ['time_index','original_row_index','edge_pair_chain_id','left_edge_track_id','right_edge_track_id',
    'polarity','first_observed_s','observations','direct_duration_s','persistent','first_available_s',
    'qualification','qualified_segment_id','qualified_first_observed_s','qualified_observations',
    'qualified_duration_s','qualified_persistent','qualified_first_available_s','original_pair_track_id',
    'width_px']


def audit_pairs(table,times,frame_indices):
    times=np.asarray(times,float);frames=np.asarray(frame_indices);table=np.asarray(table,float)
    if (times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(times<0.)
            or np.any(np.diff(times)<=0.)):
        raise ValueError('finite strictly increasing source times required')
    if (frames.shape!=times.shape or frames.dtype.kind not in 'iu' or np.any(frames<0)
            or np.any(np.diff(frames)<=0)):
        raise ValueError('strictly increasing nonnegative source frame indices required')
    if table.ndim!=2 or table.shape[1]!=len(original.PAIR_COLUMNS):
        raise ValueError('original V22 pair table schema required')
    col={name:i for i,name in enumerate(original.PAIR_COLUMNS)}
    integer_fields=['time_index','original_row_index','row_pair_track_id','left_edge_track_id','right_edge_track_id']
    for name in integer_fields:
        a=table[:,col[name]]
        if not np.isfinite(a).all() or np.any(a<0) or np.any(a!=np.floor(a)):
            raise ValueError('invalid integer source field '+name)
    if np.any(table[:,col['time_index']]>=len(times)):
        raise ValueError('table time index outside source')
    if not np.isin(table[:,col['band_candidate']],[0.,1.]).all() or not np.isin(table[:,col['bright_1_dark_minus1']],[-1.,1.]).all():
        raise ValueError('invalid original qualification or polarity')
    if not np.isfinite(table[:,col['width_px']]).all():
        raise ValueError('nonfinite original pair width')
    rows_by_time=[[] for _ in times]
    for row in table:rows_by_time[int(row[col['time_index']])].append(row)
    previous={};chains=[];segments=[];records=[];new_table=[];original_fragments={};transition_records=[]
    for i,now in enumerate(times):
        current={};new_pair_confirms=[];new_qualified_confirms=[]
        for source in rows_by_time[i]:
            key=tuple(int(source[col[n]]) for n in ('original_row_index','left_edge_track_id','right_edge_track_id','bright_1_dark_minus1'))
            if key in current:raise ValueError('duplicate pair identity in one source frame')
            qualified=bool(source[col['band_candidate']]);old_id=int(source[col['row_pair_track_id']])
            prior=previous.get(key)
            continues=bool(prior is not None and i>0 and int(frames[i])-int(frames[i-1])==1)
            if continues:
                chain=prior['chain'];chain['observations']+=1
                if old_id!=prior['original_id']:
                    transition_records.append(dict(time_s=float(now),source_frame_index=int(frames[i]),
                        row=key[0],edge_pair_chain_id=chain['chain_id'],previous_original_pair_id=prior['original_id'],
                        current_original_pair_id=old_id,qualification_changed=qualified!=prior['qualified']))
                if qualified!=prior['qualified']:
                    chain['qualification_flips']+=1
            else:
                chain=dict(chain_id=len(chains)+1,original_row_index=key[0],left_edge_track_id=key[1],right_edge_track_id=key[2],
                    polarity=key[3],first_observed_s=float(now),first_source_frame=int(frames[i]),observations=1,
                    first_available_s=None,qualification_flips=0,minimum_width_px=float(source[col['width_px']]),
                    maximum_width_px=float(source[col['width_px']]),original_pair_ids=[])
                chains.append(chain)
            chain.update(last_observed_s=float(now),last_source_frame=int(frames[i]),
                duration_s=float(now-chain['first_observed_s']))
            chain['minimum_width_px']=min(chain['minimum_width_px'],float(source[col['width_px']]))
            chain['maximum_width_px']=max(chain['maximum_width_px'],float(source[col['width_px']]))
            if old_id not in chain['original_pair_ids']:chain['original_pair_ids'].append(old_id)
            persistent=chain['observations']>=RULES['minimum_observations'] and chain['duration_s']>=RULES['minimum_duration_s']-1e-9
            if persistent and chain['first_available_s'] is None:
                chain['first_available_s']=float(now)
                new_pair_confirms.append(dict(chain_id=chain['chain_id'],original_row_index=key[0],
                    first_observed_s=chain['first_observed_s'],first_available_s=float(now),physical_birth=False))
            segment=None
            if qualified:
                if continues and prior['qualified']:
                    segment=prior['segment'];segment['observations']+=1
                else:
                    segment=dict(segment_id=len(segments)+1,chain_id=chain['chain_id'],original_row_index=key[0],
                        first_observed_s=float(now),first_source_frame=int(frames[i]),observations=1,first_available_s=None)
                    segments.append(segment)
                segment.update(last_observed_s=float(now),last_source_frame=int(frames[i]),
                    duration_s=float(now-segment['first_observed_s']))
                spersistent=segment['observations']>=RULES['minimum_observations'] and segment['duration_s']>=RULES['minimum_duration_s']-1e-9
                if spersistent and segment['first_available_s'] is None:
                    segment['first_available_s']=float(now)
                    new_qualified_confirms.append(dict(segment_id=segment['segment_id'],chain_id=chain['chain_id'],
                        original_row_index=key[0],first_observed_s=segment['first_observed_s'],first_available_s=float(now),physical_birth=False))
            # An archive may omit a source frame without changing its old ID;
            # never let that old label bridge a gap in this stricter audit.
            fragment_key=(key[0],old_id,chain['chain_id'])
            fragment=original_fragments.setdefault(fragment_key,dict(original_row_index=key[0],original_pair_id=old_id,
                corrected_chain_id=chain['chain_id'],first_observed_s=float(now),observations=0,qualification=qualified))
            fragment['observations']+=1;fragment['last_observed_s']=float(now);fragment['duration_s']=float(now-fragment['first_observed_s'])
            if fragment['qualification']!=qualified:raise ValueError('original pair track qualification unexpectedly changes')
            current[key]=dict(chain=chain,segment=segment,qualified=qualified,original_id=old_id)
            new_table.append([i,key[0],chain['chain_id'],key[1],key[2],key[3],chain['first_observed_s'],chain['observations'],
                chain['duration_s'],persistent,chain['first_available_s'] if chain['first_available_s'] is not None else np.nan,
                qualified,segment['segment_id'] if segment else 0,segment['first_observed_s'] if segment else np.nan,
                segment['observations'] if segment else 0,segment['duration_s'] if segment else np.nan,
                bool(segment and segment['first_available_s'] is not None),
                segment['first_available_s'] if segment and segment['first_available_s'] is not None else np.nan,
                old_id,float(source[col['width_px']])])
        records.append(dict(index=i,time_s=float(now),source_frame_index=int(frames[i]),present_edge_pairs=len(current),
            persistent_edge_pairs=sum(x['chain']['first_available_s'] is not None for x in current.values()),
            currently_qualified_pairs=sum(x['qualified'] for x in current.values()),
            persistent_qualified_segments=sum(x['segment'] is not None and x['segment']['first_available_s'] is not None for x in current.values()),
            new_edge_pair_persistence_confirmations=new_pair_confirms,new_qualified_persistence_confirmations=new_qualified_confirms))
        previous=current
    persistent_chains=[c for c in chains if c['first_available_s'] is not None]
    persistent_segments=[s for s in segments if s['first_available_s'] is not None]
    fragments=list(original_fragments.values())
    short_fragments=[f for f in fragments if f['observations']<5 or f['duration_s']<.4-1e-9]
    coupling_transitions=[t for t in transition_records if t['qualification_changed']]
    coupling_chains={t['edge_pair_chain_id'] for t in coupling_transitions}
    persistent_ids={c['chain_id'] for c in persistent_chains}
    def stats(items):
        selected=[x for x in items if x['first_available_s'] is not None]
        return dict(total=len(items),persistent=len(selected),
            earliest_first_observed_s=min((x['first_observed_s'] for x in selected),default=None),
            earliest_first_available_s=min((x['first_available_s'] for x in selected),default=None),
            longest=deepcopy(max(items,key=lambda x:x['duration_s'])) if items else None)
    summary=dict(edge_pair_chains=stats(chains),qualified_segments=stats(segments),
        original_pair_fragments=len(fragments),original_short_fragments=len(short_fragments),
        original_pair_ID_restart_transitions=len(transition_records),
        restarts_at_qualification_flip=len(coupling_transitions),
        restarts_without_qualification_flip=len(transition_records)-len(coupling_transitions),
        corrected_chains_containing_qualification_flip=len(coupling_chains),
        original_short_fragments_in_flip_affected_chains=sum(f['corrected_chain_id'] in coupling_chains for f in short_fragments),
        original_short_fragments_in_flip_affected_now_persistent_chains=sum(f['corrected_chain_id'] in coupling_chains&persistent_ids for f in short_fragments),
        corrected_persistent_chains_involving_flips=len(coupling_chains&persistent_ids),
        caution='Continuity of gradient IDs is not material identity; no new-layer timing is inferred.')
    return dict(records=records,chains=chains,qualified_segments=segments,original_fragments=fragments,
        restart_transitions=transition_records,summary=summary,table=np.asarray(new_table,float).reshape((-1,len(COLUMNS))))


def _dependencies(original_base):
    b=Path(original_base)
    return [b/'protocol.json',b/'left/report.json',b/'left/observations.npz']


def freeze(base=BASE,original_base=original.BASE):
    base=Path(base)
    if base.exists():raise FileExistsError('fresh continuity-audit output required')
    original.verify(original_base)
    report=serialization._read(Path(original_base)/'left/report.json')
    if (report['status']!='complete' or report['protocol_sha256']!=serialization.sha(Path(original_base)/'protocol.json')
            or report['observations_sha256']!=serialization.sha(Path(original_base)/'left/observations.npz')
            or report['pair_columns']!=original.PAIR_COLUMNS):raise ValueError('original V22 output binding failed')
    p=dict(schema='v22_pair_continuity_separation_protocol_1',rules=deepcopy(RULES),
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_base=str(original_base),
        sources={str(Path(__file__)):serialization.sha(Path(__file__))},
        dependencies={str(p):serialization.sha(p) for p in _dependencies(original_base)},
        no_truth_files_read=True,no_peaks_recomputed=True,
        correction_reason='Original pair key included contrast qualification; separate identity continuity from contrast runs.')
    serialization._write(base/'protocol.json',p);return p


def verify(base=BASE):
    p=serialization._read(Path(base)/'protocol.json');original.verify(p['original_base'])
    if p.get('schema')!='v22_pair_continuity_separation_protocol_1' or p['rules']!=RULES:raise ValueError('rules changed')
    if set(p['sources'])!={str(Path(__file__))} or set(p['dependencies'])!={str(x) for x in _dependencies(p['original_base'])}:
        raise ValueError('inventory changed')
    for path,digest in {**p['sources'],**p['dependencies']}.items():
        if serialization.sha(path)!=digest:raise ValueError('frozen dependency changed '+path)
    return p


def run(base=BASE):
    base=Path(base);p=verify(base);dest=base/'left'
    if dest.exists():raise FileExistsError('do not overwrite continuity audit')
    old=serialization._read(Path(p['original_base'])/'left/report.json')
    with np.load(Path(p['original_base'])/'left/observations.npz',allow_pickle=False) as z:
        table=z['pair_table'].copy();times=z['times_s'].copy()
    frame_indices=np.asarray([r['frame_index'] for r in old['records']],np.int64)
    if len(times)!=len(old['records']) or not np.array_equal(times,np.asarray([r['time_s'] for r in old['records']])):
        raise ValueError('source time binding mismatch')
    dest.mkdir(parents=True);serialization._write(dest/'started.json',dict(status='started_not_complete'))
    result=audit_pairs(table,times,frame_indices);verify(base)
    with (dest/'continuity.npz').open('xb') as f:
        np.savez_compressed(f,times_s=times,frame_indices=frame_indices,table=result.pop('table'))
    # All observation-level keys needed to reproduce the aggregates remain in
    # the NPZ. Avoid duplicating thousands of verbose mutable track dictionaries.
    result['detail_inventory']={name:len(result.pop(name)) for name in
        ('chains','qualified_segments','original_fragments','restart_transitions')}
    result.update(schema='v22_pair_continuity_separation_result_1',status='complete',columns=COLUMNS,
        protocol_sha256=serialization.sha(base/'protocol.json'),table_sha256=serialization.sha(dest/'continuity.npz'),
        interpretation='Post-discovery statistical separation; not new detection, independent validation or event confirmation.',
        material_identity_confirmed=False,new_layer_confirmed=False,actuator_command=None)
    serialization._write(dest/'report.json',result)
    print(result['summary'],flush=True);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','verify','run'));parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args.base)
    elif args.stage=='verify':verify(args.base);print('continuity audit verified')
    else:run(args.base)


if __name__=='__main__':main()
