"""Post-hoc failure decomposition. Never used as detector input or thresholds.

The inspected node windows come from user-reported event times and serve only
to separate near-event availability from whole-video distance-censored support.
They must not become a new apparent blind timing validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
from pathlib import Path

import numpy as np

from .v16_video_measurement import BASE, ARMS, read, sha, write

NODE_TIMES = dict(left=[24.3,27.3,30.], right=[11.4,14.4,16.8])
INSPECT_TIMES = dict(left=[24.2,24.6,27.2,30.], right=[11.4,14.4,16.8])


def summary(rows):
    checks=('point_support','span_support','tangent_agreement')
    tested=[r for r in rows if 'checks' in r['features']]
    return dict(frames=len(rows), extraction_available=sum(r['extraction_available'] for r in rows),
        decoded_path_available=sum(r['decoded']['valid'] for r in rows),
        any_direct_support=sum(r['features'].get('supported_samples',0)>0 for r in rows),
        proxy_usable=sum(r['features']['usable'] for r in rows),
        with_evaluable_checks=len(tested), no_check_frames=len(rows)-len(tested),
        individual_failures_nonexclusive={key:sum(not r['features']['checks'][key] for r in tested) for key in checks},
        exclusive_failure_combinations=dict(Counter('+'.join(k for k in checks if not r['features']['checks'][k]) or 'PASS' for r in tested)),
        reasons=dict(Counter(r['features']['reason'] for r in rows)),
        interpretation='availability only; distant frames may legitimately lack near-boundary support')


def spatial_decomposition(meta, row):
    geometry=meta.get('geometry'); exclusion=meta.get('exclusion',{})
    result=dict(frame_index=row['frame_index'],time_s=row['time_s'],
        features={k:v for k,v in row['features'].items() if k!='comparisons'},
        decoded_reason=row['decoded']['reason'],decoded_valid=row['decoded']['valid'],
        free_seed=meta['free_seed'],exclusion=exclusion,rope_width_px=meta['rope'].get('width_px'),
        excluded_arc_profiles=meta['arc'].get('diagnostics',{}).get('excluded_profile_count'),
        arc_local_segments=len(meta['arc'].get('local_boundary_segments',[])))
    if not geometry:return result
    arrays=[np.asarray(s,float) for s in row['decoded'].get('observed_segments',[]) if len(s)]
    if not arrays:return result
    points=np.concatenate(arrays); origin=np.asarray(geometry['origin_xy']);d=np.asarray(geometry['d_xy'])
    points=points[(points-origin)@d>=0]
    result['downstream_predicted_points']=len(points)
    if not len(points):return result
    n=np.asarray(meta['rope']['normal']);rho=meta['rope']['rho_px'];width=geometry['width_px']
    signed=points@n-rho
    result['predicted_path_free_line_distance_quantiles_px']=np.percentile(abs(signed),[0,25,50,75,100]).tolist()
    if exclusion.get('valid'):
        half=exclusion['half_width_px'];inside=abs(signed)<=half
        result['predicted_points_in_arc_exclusion_band']=int(inside.sum())
        result['predicted_fraction_in_arc_exclusion_band']=float(inside.mean())
        result['exclusion_half_width_in_optical_widths']=half/width
        # This geometric identity diagnoses the observation policy, not hidden
        # image evidence: any boundary within half a rope stripe of its fitted
        # straight centerline falls inside a wider excluded corridor.
        result['band_removes_straight_line_half_width_contact_zone']=half>=width*.5
    supported=[p for c in row['features'].get('comparisons',[]) for p in c.get('samples',[])]
    if supported:
        pp=np.asarray([s['point'] for s in supported]);uu=(pp-origin)@d
        result['direct_supported_longitudinal_range_widths']=[float(uu.min()/width),float(uu.max()/width)]
        result['direct_supported_path_free_line_distance_quantiles_px']=np.percentile(abs(pp@n-rho),[0,25,50,75,100]).tolist()
    return result


def run(output,destination=None):
    output=Path(output)
    if destination is None:destination=output/'diagnostics'/('decomposition_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination=Path(destination)
    if destination.exists():raise FileExistsError('use new diagnostic directory')
    results={}; sources={}
    for session in NODE_TIMES:
        results[session]={}
        for arm in ARMS:
            path=output/'measure'/session/(arm+'.json');value=read(path);sources[str(path)]=sha(path);rows=value['records']
            node_rows=[r for r in rows if any(abs(r['time_s']-t)<=1.+1e-9 for t in NODE_TIMES[session])]
            inspected=[]
            for t in INSPECT_TIMES[session]:
                row=min(rows,key=lambda r:abs(r['time_s']-t));meta_path=output/'extract'/session/'frames'/f'f{row["frame_index"]:06d}.json'
                meta=read(meta_path);sources[str(meta_path)]=sha(meta_path);inspected.append(spatial_decomposition(meta,row))
            results[session][arm]=dict(all_video=summary(rows),user_node_windows_plus_minus_1s=summary(node_rows),
                each_node_window=[dict(reference_time_s=t,**summary([r for r in rows if abs(r['time_s']-t)<=1.+1e-9])) for t in NODE_TIMES[session]],
                inspected=inspected)
    write(destination/'summary.json',dict(posthoc_diagnostic=True,not_for_inference=True,
        user_time_windows_used_only_for_reporting=NODE_TIMES,inspection_times=INSPECT_TIMES,
        threshold_changes=False,source_snapshot=sources,script_sha256=sha(__file__),results=results))
    for session,arms in results.items():
        for arm,value in arms.items():
            print(session,arm,'all',value['all_video'],'nodes',value['user_node_windows_plus_minus_1s'])
            for r in value['inspected']:
                print('frame',r['time_s'],'checks',r['features'].get('checks'),'points',r['features'].get('supported_samples'),
                    'span',r['features'].get('contiguous_supported_span_px'),'angle',r['features'].get('median_tangent_angle_deg'),
                    'bandfraction',r.get('predicted_fraction_in_arc_exclusion_band'),'bandwidth',r.get('exclusion_half_width_in_optical_widths'))
    print(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE);parser.add_argument('--destination',type=Path)
    args=parser.parse_args();run(args.output,args.destination)
