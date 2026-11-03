"""Past-only local multiple-image-edge observations; NOT new-rope identity.

Use the archived V18b normal profiles on the 24 original rows spatially checked
against the user's box. Preserve multiple peaks and paired bright/dark bands.
This is development-set observability, not layer-event detection or validation.
No truth labels, models, video decoding, hardware or deployment are used here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import datetime
from pathlib import Path

import numpy as np

from vision_geometry_lab import v18_gradient_units_trial as upstream
from vision_geometry_lab import v19_front_trial as serialization


LAB = Path(__file__).resolve().parent
BASE = LAB/'outputs/v22_local_edges'
BOX_AUDIT = LAB/'outputs/v21_bulge_review/box_audit_273/box_audit.json'
ROW_INDICES = np.arange(66, 90, dtype=np.int64)
RULES = dict(
    original_row_indices=ROW_INDICES.tolist(), profile_offsets_px=[-65,65],
    row_scope='whole_original_normal_strips_including_parts_outside_user_rectangle',
    coordinate_system='canonical_reference_image_normal_pixels_not_mm_or_rope_diameter',
    startup_ready_s=5., startup_outputs='warmup_no_causal_candidates',
    profile_required='pose_valid_all_131_samples_finite_in_0_to_255',
    maximum_saturated_fraction=.25,
    smooth_sigma_px=1., smooth_radius_px=3, gradient='central_difference_of_smoothed_profile',
    minimum_abs_gradient=1., gradient_units='gray_levels_per_reference_pixel',
    threshold_scope='fixed_image_observation_heuristic_not_revalidated_after_smoothing',
    same_polarity_nms_distance_px=2., gradient_support_guard_px=5,
    support_guard_derivation='Gaussian_radius3_plus_derivative_radius1_plus_peak_neighbor_radius1',
    pair_selection='adjacent_opposite_polarity_edges_only',
    pair_width_min_px=3., pair_width_max_px=40.,
    pair_width_scope='fixed_analysis_range_only; all_out_of_range_edge_peaks_retained; not_rope_absence',
    distance_parameter_units='original_reference_image_pixels; not_general_rope_diameter_limits',
    band_outer_samples='3_original_samples_offsets_3_to_5_beyond_each_edge',
    band_inner_samples='original_samples_strictly_between_edges',
    band_minimum_two_sided_contrast_gray=4.,
    edge_association='same_polarity_mutual_nearest_position_only_unique_minimum',
    maximum_step_px=3., maximum_time_gap_s=.3,
    missing_or_ambiguous='do_not_bridge; reset affected_edge_or_row',
    persistence_minimum_observations=5, persistence_minimum_duration_s=.4,
    pair_association='same_two_edge_track_ids_and_same_band_kind',
    profile_identity_confirmed=False, new_rope_identity_confirmed=False,
    new_layer_confirmed=False, mechanical_revolution_confirmed=False,
    truth_input=False, hardware_access=False, deployment_allowed=False,
    end_s=60., summary_bucket_seconds=5., summary_last_bucket_includes_end=True)
STOP_RULE = dict(
    no_threshold_search='One frozen parameter set; no adjustment to expose a desired time.',
    unavailable='Keep unknown observations and interrupted tracks; do not interpolate.',
    persistent_not_new='Static stripes, periodic texture and flange wrinkles can also persist.',
    no_event='No t0/t1/t2, force timing or new-layer confirmation from this experiment.',
    reference_time='27.3s is withdrawn as definite new-layer truth; not read or used for scoring.',
    deployment='Do not modify live UI, devices, safety locks or frozen V18-V21.')


def _flag(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _strict_peaks(values, guard):
    """Local maxima including a flat top, excluding monotonic/constant ramps."""
    peaks=[]; k=guard
    while k < len(values)-guard:
        end=k
        while end+1 < len(values)-guard and abs(values[end+1]-values[k]) <= 1e-10:
            end+=1
        if values[k] > values[k-1]+1e-10 and values[end] > values[end+1]+1e-10:
            centre=.5*(k+end)
            if end == k:
                a,b,c=values[k-1:k+2]; denom=a-2*b+c
                if denom < -1e-10:
                    centre += float(np.clip(.5*(a-c)/denom,-.5,.5))
            peaks.append((centre,float(values[k])))
        k=end+1
    return peaks


def observe_profile(profile, pose_valid=True, saturated_fraction=0.):
    """Return all supported signed edges and adjacent opposite-edge pairs."""
    p=np.asarray(profile,float)
    if p.shape != (131,):
        raise ValueError('original 131-sample profile required')
    result=dict(valid=False,reason='pose_unavailable',edges=[],pairs=[],opposite_pairs_outside_analysis_range=0,
                smooth=np.full(131,np.nan),gradient=np.full(131,np.nan))
    if not _flag(pose_valid):
        return result
    if not np.isfinite(p).all() or np.any(p<0.) or np.any(p>255.):
        return {**result,'reason':'profile_missing_or_out_of_range'}
    if (isinstance(saturated_fraction,(bool,np.bool_)) or not np.isscalar(saturated_fraction)
            or not np.isfinite(saturated_fraction) or not 0.<=saturated_fraction<=RULES['maximum_saturated_fraction']):
        return {**result,'reason':'saturation_diagnostic_unacceptable'}
    x=np.arange(-3,4,dtype=float); kernel=np.exp(-.5*x*x); kernel/=kernel.sum()
    smooth=np.convolve(np.pad(p,3,mode='reflect'),kernel,mode='valid')
    gradient=np.gradient(smooth); candidates=[]
    for sign in (1,-1):
        same=[]
        for position,strength in _strict_peaks(sign*gradient,RULES['gradient_support_guard_px']):
            if strength >= RULES['minimum_abs_gradient']:
                same.append(dict(offset_px=position-65.,polarity=sign,abs_gradient=strength))
        selected=[]
        for edge in sorted(same,key=lambda e:(-e['abs_gradient'],e['offset_px'])):
            if all(abs(edge['offset_px']-other['offset_px']) >= RULES['same_polarity_nms_distance_px'] for other in selected):
                selected.append(edge)
        candidates.extend(selected)
    edges=sorted(candidates,key=lambda e:e['offset_px']); pairs=[]; outside_range=0
    for left_index,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
        width=right['offset_px']-left['offset_px']
        if left['polarity']==right['polarity']:
            continue
        if not RULES['pair_width_min_px']<=width<=RULES['pair_width_max_px']:
            outside_range+=1;continue
        a,b=left['offset_px']+65.,right['offset_px']+65.
        inside=np.arange(int(np.floor(a))+1,int(np.ceil(b)))
        outside_left=np.arange(int(np.floor(a))-5,int(np.floor(a))-2)
        outside_right=np.arange(int(np.ceil(b))+3,int(np.ceil(b))+6)
        supported=bool(len(inside) and outside_left[0]>=0 and outside_right[-1]<131)
        kind='bright' if left['polarity']==1 else 'dark'
        contrast=None
        if supported:
            interior=float(np.median(p[inside])); l=float(np.mean(p[outside_left])); r=float(np.mean(p[outside_right]))
            contrast=interior-max(l,r) if kind=='bright' else min(l,r)-interior
        pairs.append(dict(left_edge_index=left_index,right_edge_index=left_index+1,
            left_offset_px=left['offset_px'],right_offset_px=right['offset_px'],width_px=width,
            center_px=.5*(left['offset_px']+right['offset_px']),polarity_kind=kind,
            raw_two_sided_contrast_gray=contrast,band_support_complete=supported,
            band_candidate=bool(supported and contrast>=RULES['band_minimum_two_sided_contrast_gray']),
            kind='opposite_edge_pair_not_material_identity'))
    return dict(valid=True,reason='direct_multi_edge_observation_not_identity',edges=edges,pairs=pairs,
                smooth=smooth,gradient=gradient,opposite_pairs_outside_analysis_range=outside_range)


class RowTracker:
    """Bounded previous-observation image associations; no material identities."""
    def __init__(self):
        self.last_time=None; self.edges=[]; self.pairs={}; self.next_edge=1; self.next_pair=1

    def clear(self):
        self.edges=[]; self.pairs={}

    def update(self,time_s,observation):
        now=float(time_s)
        if not np.isfinite(now) or now<0.:
            raise ValueError('finite nonnegative time required')
        previous_time=self.last_time; self.last_time=now
        if previous_time is not None and now<=previous_time:
            self.clear(); return dict(valid=False,reason='time_not_increasing_reset',edges=[],pairs=[])
        if not observation['valid']:
            self.clear(); return dict(valid=False,reason=observation['reason'],edges=[],pairs=[])
        if previous_time is not None and now-previous_time>RULES['maximum_time_gap_s']+1e-9:
            self.clear()
        current=deepcopy(observation['edges']); old=self.edges; matches={}
        if current and old:
            distances=np.full((len(old),len(current)),np.inf)
            for i,a in enumerate(old):
                for j,b in enumerate(current):
                    if a['polarity']==b['polarity']:
                        distances[i,j]=abs(a['offset_px']-b['offset_px'])
            for j in range(len(current)):
                col=distances[:,j]; i=int(np.argmin(col)); distance=col[i]
                if not np.isfinite(distance) or distance>RULES['maximum_step_px']:
                    continue
                row=distances[i]
                if (int(np.argmin(row))==j and np.sum(abs(col-distance)<1e-9)==1
                        and np.sum(abs(row-distance)<1e-9)==1):
                    matches[j]=i
        tracked=[]
        for j,edge in enumerate(current):
            if j in matches:
                prior=old[matches[j]]; identity=prior['track_id']; first=prior['first_observed_s']; count=prior['observations']+1
            else:
                identity=self.next_edge;self.next_edge+=1;first=now;count=1
            age=now-first
            persistent=bool(count>=RULES['persistence_minimum_observations'] and age>=RULES['persistence_minimum_duration_s']-1e-9)
            first_available=old[matches[j]].get('first_available_s') if j in matches else None
            if persistent and first_available is None:first_available=now
            tracked.append({**edge,'track_id':identity,'first_observed_s':first,'observations':count,
                'direct_duration_s':age,'persistent':persistent,'first_available_s':first_available,
                'association':'same_polarity_position_only_not_material_identity'})
        paired=[]; next_pairs={}
        for pair in observation['pairs']:
            left=tracked[pair['left_edge_index']];right=tracked[pair['right_edge_index']]
            key=(left['track_id'],right['track_id'],pair['polarity_kind'],pair['band_candidate'])
            prior=self.pairs.get(key)
            if prior is None:
                identity=self.next_pair;self.next_pair+=1;first=now;count=1;initial=pair['width_px']
            else:
                identity=prior['track_id'];first=prior['first_observed_s'];count=prior['observations']+1;initial=prior['initial_width_px']
            age=now-first
            persistent=bool(count>=RULES['persistence_minimum_observations'] and age>=RULES['persistence_minimum_duration_s']-1e-9)
            first_available=prior.get('first_available_s') if prior is not None else None
            if persistent and first_available is None:first_available=now
            out={**pair,'track_id':identity,'left_edge_track_id':left['track_id'],'right_edge_track_id':right['track_id'],
                'first_observed_s':first,'observations':count,'direct_duration_s':age,
                'persistent':persistent,'first_available_s':first_available,
                'initial_width_px':initial,'relative_separation_change_px':pair['width_px']-initial,
                'first_observed_is_physical_birth':False}
            paired.append(out);next_pairs[key]=out
        self.edges=tracked;self.pairs=next_pairs
        return dict(valid=True,reason='past_only_image_association',edges=tracked,pairs=paired)


def replay(times,profiles,poses,saturated,rows):
    times=np.asarray(times,float);profiles=np.asarray(profiles,float);poses=np.asarray(poses);saturated=np.asarray(saturated,float)
    rows=np.asarray(rows)
    if (times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(times<0.)
            or np.any(np.diff(times)<=0.) or np.any(times>RULES['end_s'])):
        raise ValueError('strict causal 0..60s source timeline required')
    if (profiles.shape!=(len(times),len(rows),131) or saturated.shape!=profiles.shape[:2]
            or poses.shape!=(len(times),) or poses.dtype.kind!='b'):
        raise ValueError('source array shape/type mismatch')
    trackers=[RowTracker() for _ in rows]; edge_table=[]; pair_table=[]; records=[]
    usable=np.zeros(profiles.shape[:2],bool);edge_counts=np.zeros(profiles.shape[:2],np.int16)
    pair_counts=np.zeros_like(edge_counts);band_counts=np.zeros_like(edge_counts);persistent_counts=np.zeros_like(edge_counts)
    smooth=np.full_like(profiles,np.nan,dtype=np.float32);gradient=np.full_like(smooth,np.nan)
    for i,now in enumerate(times):
        reasons=Counter(); frame_confirmations=[];outside_count=0
        for j,original_row in enumerate(rows):
            if now<=RULES['startup_ready_s']:
                observation=dict(valid=False,reason='startup_reference_not_yet_available',edges=[],pairs=[])
            else:
                observation=observe_profile(profiles[i,j],bool(poses[i]),saturated[i,j])
                smooth[i,j]=observation['smooth'];gradient[i,j]=observation['gradient']
                outside_count+=observation['opposite_pairs_outside_analysis_range']
            tracked=trackers[j].update(now,observation);usable[i,j]=tracked['valid'];reasons[tracked['reason']]+=1
            edge_counts[i,j]=len(tracked['edges']);pair_counts[i,j]=len(tracked['pairs'])
            band_counts[i,j]=sum(p['band_candidate'] for p in tracked['pairs'])
            for edge in tracked['edges']:
                edge_table.append([i,original_row,edge['offset_px'],edge['polarity'],edge['abs_gradient'],
                    edge['track_id'],edge['first_observed_s'],edge['observations'],edge['direct_duration_s'],edge['persistent'],
                    edge['first_available_s'] if edge['first_available_s'] is not None else np.nan])
            for pair in tracked['pairs']:
                pair_table.append([i,original_row,pair['left_offset_px'],pair['right_offset_px'],pair['width_px'],
                    1 if pair['polarity_kind']=='bright' else -1,pair['raw_two_sided_contrast_gray'] if pair['raw_two_sided_contrast_gray'] is not None else np.nan,
                    pair['band_candidate'],pair['track_id'],pair['left_edge_track_id'],pair['right_edge_track_id'],
                    pair['first_observed_s'],pair['observations'],pair['direct_duration_s'],pair['persistent'],
                    pair['initial_width_px'],pair['relative_separation_change_px'],
                    pair['first_available_s'] if pair['first_available_s'] is not None else np.nan])
                if pair['persistent']:
                    persistent_counts[i,j]+=1
                    if abs(pair['first_available_s']-now)<1e-9:
                        frame_confirmations.append(dict(original_row_index=int(original_row),pair_track_id=pair['track_id'],
                            first_observed_s=pair['first_observed_s'],first_available_s=pair['first_available_s'],
                            confirmation_delay_s=pair['first_available_s']-pair['first_observed_s'],
                            width_px=pair['width_px'],relative_separation_change_px=pair['relative_separation_change_px'],
                            polarity_kind=pair['polarity_kind'],band_candidate=pair['band_candidate'],
                            physical_new_rope_birth_confirmed=False))
        records.append(dict(index=i,time_s=float(now),usable_rows=int(usable[i].sum()),
            edge_count=int(edge_counts[i].sum()),opposite_pair_count=int(pair_counts[i].sum()),
            band_candidate_count=int(band_counts[i].sum()),persistent_pair_count=int(persistent_counts[i].sum()),
            opposite_pairs_outside_analysis_range=outside_count,
            row_reasons=dict(reasons),new_persistence_confirmations=frame_confirmations))
    arrays=dict(times_s=times,original_row_indices=rows,usable=usable,edge_counts=edge_counts,pair_counts=pair_counts,
        band_counts=band_counts,persistent_pair_counts=persistent_counts,smooth_profiles=smooth,gradient_profiles=gradient,
        edge_table=np.asarray(edge_table,float).reshape((-1,11)),pair_table=np.asarray(pair_table,float).reshape((-1,18)))
    return dict(records=records,arrays=arrays)


EDGE_COLUMNS=['time_index','original_row_index','normal_offset_px','polarity','abs_gradient','row_track_id',
              'first_observed_s','observations','direct_duration_s','persistent','first_available_s']
PAIR_COLUMNS=['time_index','original_row_index','left_offset_px','right_offset_px','width_px','bright_1_dark_minus1',
              'raw_two_sided_contrast_gray','band_candidate','row_pair_track_id','left_edge_track_id','right_edge_track_id',
              'first_observed_s','observations','direct_duration_s','persistent','initial_width_px','relative_separation_change_px',
              'first_available_s']


def summary(records,arrays):
    def one(indices):
        ids=np.asarray(indices,int); selected=[records[i] for i in ids]
        return dict(frames=len(ids),row_frame_cells=len(ids)*len(arrays['original_row_indices']),
            usable_cells=int(arrays['usable'][ids].sum()),
            cells_with_multiple_edges=int((arrays['edge_counts'][ids]>=2).sum()),
            cells_with_opposite_pair=int((arrays['pair_counts'][ids]>0).sum()),
            cells_with_bright_or_dark_band=int((arrays['band_counts'][ids]>0).sum()),
            cells_with_persistent_image_pair=int((arrays['persistent_pair_counts'][ids]>0).sum()),
            frames_with_persistent_image_pair=sum(r['persistent_pair_count']>0 for r in selected),
            opposite_pairs_outside_analysis_range=sum(r['opposite_pairs_outside_analysis_range'] for r in selected),
            row_reasons=dict(sum((Counter(r['row_reasons']) for r in selected),Counter())))
    buckets=[]
    for start in range(0,60,5):
        ids=[i for i,r in enumerate(records) if start<=r['time_s'] and (r['time_s']<=60 if start==55 else r['time_s']<start+5)]
        buckets.append(dict(start_s=start,end_s=start+5,end_inclusive=start==55,summary=one(ids)))
    table=arrays['pair_table'];persistent=table[table[:,14]==1.]
    def track_stats(selected):
        if not len(selected):return dict(persistent_row_pair_tracks=0,earliest_observed_s=None,earliest_first_available_s=None,longest_direct_chain=None)
        longest=selected[int(np.argmax(selected[:,13]))]
        return dict(persistent_row_pair_tracks=len(set((int(r[1]),int(r[8])) for r in selected)),
            earliest_observed_s=float(selected[:,11].min()),earliest_first_available_s=float(selected[:,17].min()),
            longest_direct_chain=dict(original_row_index=int(longest[1]),pair_track_id=int(longest[8]),
                first_observed_s=float(longest[11]),first_available_s=float(longest[17]),
                last_observed_s=float(arrays['times_s'][int(longest[0])]),observations=int(longest[12]),
                duration_s=float(longest[13]),polarity_kind='bright' if longest[5]==1 else 'dark',
                band_candidate=bool(longest[7]),initial_width_px=float(longest[15]),last_width_px=float(longest[4]),
                relative_separation_change_px=float(longest[16])))
    edges=arrays['edge_table'];pe=edges[edges[:,9]==1.]
    edge_stats=dict(persistent_row_edge_tracks=0,earliest_observed_s=None,earliest_first_available_s=None,longest_direct_chain=None)
    if len(pe):
        longest=pe[int(np.argmax(pe[:,8]))]
        edge_stats=dict(persistent_row_edge_tracks=len(set((int(e[1]),int(e[5])) for e in pe)),
            earliest_observed_s=float(pe[:,6].min()),earliest_first_available_s=float(pe[:,10].min()),
            longest_direct_chain=dict(original_row_index=int(longest[1]),edge_track_id=int(longest[5]),
                first_observed_s=float(longest[6]),first_available_s=float(longest[10]),
                last_observed_s=float(arrays['times_s'][int(longest[0])]),observations=int(longest[7]),
                duration_s=float(longest[8]),polarity=int(longest[3])))
    return dict(complete=one(range(len(records))),fixed_five_second_buckets=buckets,edge_tracks=edge_stats,
        opposite_pair_tracks=track_stats(persistent),bright_or_dark_band_tracks=track_stats(persistent[persistent[:,7]==1.]),
        negative_control_warning='Static bands and periodic texture can pass persistence; persistence is not new material or layer birth.',
        accuracy_or_event_performance_measured=False)


def _dependencies(upstream_base,box_audit):
    b=Path(upstream_base)
    return [b/'protocol.json',b/'left/index.json',b/'left/measurement.npz',Path(box_audit)]


def freeze(base=BASE,upstream_base=upstream.BASE,box_audit=BOX_AUDIT):
    base=Path(base)
    if base.exists():raise FileExistsError('fresh V22 directory required')
    upstream.verify(upstream_base)
    audit=serialization._read(box_audit)
    if [r['index'] for r in audit['touched_rows']]!=ROW_INDICES.tolist():
        raise ValueError('frozen spatial row audit mismatch')
    ip=Path(upstream_base)/'left/index.json';index=serialization._read(ip)
    if index['measurement_sha256']!=serialization.sha(Path(upstream_base)/'left/measurement.npz'):
        raise ValueError('upstream measurement hash mismatch')
    p=dict(schema='v22_local_multiple_image_edges_protocol_1',
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rules=deepcopy(RULES),stop_rule=deepcopy(STOP_RULE),
        source_hashes={str(path):serialization.sha(path) for path in (Path(__file__),Path(serialization.__file__))},
        dependencies={str(p):serialization.sha(p) for p in _dependencies(upstream_base,box_audit)},
        upstream_base=str(upstream_base),box_audit=str(box_audit),video=index['video'],
        study='entire_left_development_video_not_blind_validation',no_truth_files_read=True,
        original_user_box_raw_xyxy_inclusive=audit['red_box_outer_inclusive_raw_xyxy'])
    serialization._write(base/'protocol.json',p);return p


def verify(base=BASE):
    p=serialization._read(Path(base)/'protocol.json')
    if p.get('schema')!='v22_local_multiple_image_edges_protocol_1' or p['rules']!=RULES or p['stop_rule']!=STOP_RULE:
        raise ValueError('frozen V22 rules changed')
    upstream.verify(p['upstream_base'])
    if set(p['source_hashes'])!={str(Path(__file__)),str(Path(serialization.__file__))} or set(p['dependencies'])!={str(x) for x in _dependencies(p['upstream_base'],p['box_audit'])}:
        raise ValueError('frozen inventory changed')
    for path,digest in {**p['source_hashes'],**p['dependencies']}.items():
        if serialization.sha(path)!=digest:raise ValueError('frozen file changed '+path)
    return p


def run(base=BASE):
    base=Path(base);p=verify(base);dest=base/'left'
    if dest.exists():raise FileExistsError('do not overwrite V22 output')
    index=serialization._read(Path(p['upstream_base'])/'left/index.json')
    with np.load(Path(p['upstream_base'])/'left/measurement.npz',allow_pickle=False) as z:
        times=z['times_s'].copy();profiles=z['normal_profiles'][:,ROW_INDICES].copy()
        poses=z['pose_valid'].copy();saturated=z['normal_clipped_fraction'][:,ROW_INDICES].copy()
        row_y=z['rows_y'][ROW_INDICES].copy()
    if len(times)!=601 or not np.allclose(times,np.arange(601)/10.,rtol=0,atol=1e-9):
        raise ValueError('expected complete 601-frame 0..60s development clip')
    if any(r['frame_index']!=i or abs(r['time_s']-times[i])>1e-9 for i,r in enumerate(index['records'])):
        raise ValueError('source timeline mismatch')
    dest.mkdir(parents=True);serialization._write(dest/'started.json',{'status':'started_not_complete'})
    result=replay(times,profiles,poses,saturated,ROW_INDICES)
    for record,source in zip(result['records'],index['records']):
        record.update(frame_index=source['frame_index'],frame_sha256=source['frame_sha256'])
    result['summary']=summary(result['records'],result['arrays'])
    verify(base)
    with (dest/'observations.npz').open('xb') as f:
        np.savez_compressed(f,rows_y=row_y,**result.pop('arrays'))
    result.update(schema='v22_local_multiple_image_edges_observations_1',status='complete',
        protocol_sha256=serialization.sha(base/'protocol.json'),observations_sha256=serialization.sha(dest/'observations.npz'),
        edge_columns=EDGE_COLUMNS,pair_columns=PAIR_COLUMNS,
        physical_new_layer_confirmed=False,new_rope_identity_confirmed=False,mechanical_revolution_confirmed=False,
        actuator_command=None,video=index['video'])
    serialization._write(dest/'report.json',result)
    print('V22 complete; image observations only',result['summary']['complete'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','verify','run'));parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args()
    if args.stage=='freeze':freeze(args.base)
    elif args.stage=='verify':verify(args.base);print('V22 verified')
    else:run(args.base)


if __name__=='__main__':main()
