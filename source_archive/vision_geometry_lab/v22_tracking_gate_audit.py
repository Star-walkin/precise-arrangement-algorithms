"""Fixed twelve-pair posthoc attribution, not a new tracking or event trial.

Reads only V22 inputs at predeclared whole-ten-second samples. Uses unchanged
V22 LK/mask thresholds, starts fresh on each pair, never proposes candidates.
Point displacement before successful bidirectional matching is NOT measured
physical displacement. LK patch error is not uniquely an illumination error.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np

from vision_geometry_lab import v22_flange_tracks as frozen

BASE = frozen.LAB/'outputs/v22_flange_tracks/gate_attribution'
START_INDICES = (0, 100, 200, 300, 400, 500)
STAGES = ('forward_status', 'backward_status', 'finite_coordinates',
          'forward_backward_error', 'lk_patch_error', 'inside_image',
          'inside_original_ROI', 'current_yellow_nonclipped_mask')


def yellow_mask(image):
    """Same globally eroded yellow mask as V22, before its per-ROI crop."""
    r = frozen.RULES
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= r['yellow_hue_min']) & (hsv[:, :, 0] <= r['yellow_hue_max']) &
            (hsv[:, :, 1] >= r['yellow_saturation_min']) &
            (hsv[:, :, 2] >= r['minimum_value']) & (hsv[:, :, 2] < r['maximum_value'])).astype(np.uint8)*255
    return cv2.erode(mask, np.ones((r['mask_erode_size'],)*2, np.uint8))


def quantiles(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return dict(count=len(a), p10=float(np.quantile(a,.1)), p50=float(np.median(a)),
                p90=float(np.quantile(a,.9)), maximum=float(np.max(a))) if len(a) else dict(count=0)


def breakdown(points, groups, current, status1, back, status2, errors, rois, color):
    """Independent failure counts and a declared-order waterfall over flange points."""
    p = np.asarray(points, float).reshape(-1,2)
    q = np.asarray(current, float).reshape(-1,2)
    b = np.asarray(back, float).reshape(-1,2)
    g = np.asarray(groups, int).reshape(-1)
    e = np.asarray(errors, float).reshape(-1)
    fwd = np.asarray(status1).reshape(-1).astype(bool)
    rev = np.asarray(status2).reshape(-1).astype(bool)
    n = len(p)
    if q.shape != p.shape or b.shape != p.shape or any(x.shape != (n,) for x in (g,e,fwd,rev)):
        raise ValueError('inconsistent correspondence shapes')
    if np.any((g < 0) | (g > 1)):
        raise ValueError('audit only the two flange groups')
    h,w = color.shape
    finite = np.isfinite(q).all(axis=1) & np.isfinite(b).all(axis=1)
    fb = np.linalg.norm(b-p,axis=1)
    displacement = np.linalg.norm(q-p,axis=1)
    pixel = np.rint(np.nan_to_num(q,nan=-1.,posinf=-1.,neginf=-1.)).astype(int)
    inside = (pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
    in_roi = np.zeros(n,bool); colored = np.zeros(n,bool)
    for i in np.flatnonzero(inside):
        x1,y1,x2,y2 = rois[g[i]]
        x,y = pixel[i]
        in_roi[i] = x1<=x<x2 and y1<=y<y2
        colored[i] = color[y,x] != 0
    masks = dict(forward_status=fwd, backward_status=rev, finite_coordinates=finite,
        forward_backward_error=np.isfinite(fb)&(fb<=frozen.RULES['fb_max_px']),
        lk_patch_error=np.isfinite(e)&(e<=frozen.RULES['lk_error_max']),
        inside_image=inside, inside_original_ROI=in_roi, current_yellow_nonclipped_mask=colored)
    alive=np.ones(n,bool); waterfall=[]
    for name in STAGES:
        previous=int(alive.sum()); alive &= masks[name]
        waterfall.append(dict(stage=name,input_points=previous,rejected_at_this_step=previous-int(alive.sum()),retained=int(alive.sum())))
    flow_supported = fwd & rev & finite & masks['forward_backward_error'] & masks['lk_patch_error']
    rows=[]
    for i in range(n):
        rows.append(dict(region=int(g[i]), previous_xy=p[i].tolist(), candidate_current_xy=q[i].tolist(),
            forward_backward_px=float(fb[i]), lk_patch_error=float(e[i]), candidate_displacement_px=float(displacement[i]),
            passes={name:bool(mask[i]) for name,mask in masks.items()},joint_kept=bool(alive[i])))
    result=dict(initial_point_count=n, retained_count=int(alive.sum()),
        individual_failed_nonexclusive={k:int((~v).sum()) for k,v in masks.items()},
        individual_passed_nonexclusive={k:int(v.sum()) for k,v in masks.items()},waterfall=waterfall,
        flow_supported_before_spatial_color=int(flow_supported.sum()),
        flow_supported_outside_ROI=int(np.sum(flow_supported & ~in_roi)),
        flow_supported_inside_ROI_color_failed=int(np.sum(flow_supported & in_roi & ~colored)),
        LK_candidate_displacement_px=quantiles(displacement[fwd & finite]),
        FB_error_px_for_finite_forward_status=quantiles(fb[fwd & finite]),
        LK_patch_error_for_forward_status=quantiles(e[fwd]),
        bidirectionally_supported_displacement_px=quantiles(displacement[flow_supported]),
        points=rows,
        causal_interpretation='No individual gate establishes true displacement, material identity or the physical cause of tracking failure.')
    return result, alive


def inspect_pair(previous, current, rois, time_s):
    h,w=previous.shape[:2]
    tracker=frozen.FlangeTracks(rois,(w,h))
    tracker.update(previous,time_s)
    image=cv2.resize(current,tracker.size,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    settings=dict(winSize=(frozen.RULES['lk_window'],)*2,maxLevel=frozen.RULES['lk_max_level'],
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    flow=frozen._flow_pair(tracker.gray,gray,tracker.points,settings)
    used=tracker.regions<2
    n=len(tracker.points)
    if flow is None:
        q=back=np.full((n,2),np.nan)
        s1=s2=np.zeros(n,np.uint8); error=np.full(n,np.inf)
    else:
        q,s1,error,back,s2=flow
        q=q.reshape(-1,2); back=back.reshape(-1,2)
        s1=s1.ravel(); s2=s2.ravel(); error=error.ravel()
    result, kept=breakdown(tracker.points[used],tracker.regions[used],q[used],s1[used],back[used],s2[used],error[used],tracker.rois,yellow_mask(image))
    # Match the frozen implementation's actual combined mask, independently.
    frozen_masks=tracker._masks(image)
    expected=np.zeros(used.sum(),bool)
    for i,(p,z,backpoint,status_a,status_b,err,g) in enumerate(zip(tracker.points[used],q[used],back[used],s1[used],s2[used],error[used],tracker.regions[used])):
        if not(status_a and status_b and np.isfinite(z).all() and np.isfinite(backpoint).all()): continue
        if np.linalg.norm(backpoint-p)>frozen.RULES['fb_max_px'] or not np.isfinite(err) or err>frozen.RULES['lk_error_max']: continue
        x,y=np.rint(z).astype(int)
        if 0<=x<tracker.size[0] and 0<=y<tracker.size[1]: expected[i]=frozen_masks[g][y,x]!=0
    if not np.array_equal(kept,expected):
        raise AssertionError('attribution gates differ from frozen combined implementation')
    result.update(combined_mask_matches_frozen_implementation=True, LK_output_available=flow is not None,
                  analysis_scale=tracker.scale, time_s=time_s,
                  per_ROI=[dict(region=k,initial=int(np.sum(tracker.regions[used]==k)),
                    retained=int(np.sum(kept & (tracker.regions[used]==k)))) for k in range(2)])
    return result


def run(output=BASE):
    output=Path(output)
    if output.exists(): raise FileExistsError('fresh audit output required')
    p=frozen.verify()
    cv2.setNumThreads(2)
    result=dict(scope='posthoc_fixed_12_pair_gate_attribution_not_new_predictions',
        source_code_sha256=frozen.sha(__file__), frozen_protocol_sha256=frozen.sha(frozen.BASE/'protocol.json'),
        rules_unchanged=frozen.RULES, start_indices=list(START_INDICES),
        creation_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selection='both videos; 0,10,20,30,40,50 seconds and immediate next 0.1-second frame; no result-based selection',
        baseline='fresh same frozen corner initialization on every selected first frame, not continuation from previous pairs',
        units='2048-wide analysis pixels; original pixels divide by analysis_scale',
        raw_frame_hash_verified=True, candidate_output=False, mechanical_phase=False,
        labels_read=False, threshold_changed=False, sides={})
    for side in ('left','right'):
        v=p['videos'][side]
        if frozen.sha(v['path']) != v['sha256']: raise ValueError('video hash mismatch')
        index=frozen.read(p['source_indices'][side]['path'])
        cap=cv2.VideoCapture(v['path']); rows=[]
        for first in START_INDICES:
            cap.set(cv2.CAP_PROP_POS_FRAMES,first)
            frames=[]; digests=[]
            for i in (first,first+1):
                ok,raw=cap.read()
                if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1!=i: raise ValueError('seek decoded wrong frame')
                digest=frozen.pixels_sha(raw)
                if digest!=index['records'][i]['frame_sha256']: raise ValueError('decoded frame hash mismatch')
                frames.append(cv2.flip(raw,1) if p['mirroring'][side] else raw);digests.append(digest)
            row=inspect_pair(*frames,p['rois'][side],first/v['fps'])
            row.update(frame_indices=[first,first+1],raw_frame_hashes=digests)
            rows.append(row)
        cap.release()
        total=sum(r['initial_point_count'] for r in rows)
        aggregate={name:sum(r['individual_failed_nonexclusive'][name] for r in rows) for name in STAGES}
        waterfall=[dict(stage=name,rejected_at_this_step=sum(r['waterfall'][j]['rejected_at_this_step'] for r in rows),retained=sum(r['waterfall'][j]['retained'] for r in rows)) for j,name in enumerate(STAGES)]
        result['sides'][side]=dict(pairs=rows,aggregate=dict(initial_points=total,individual_failed_nonexclusive=aggregate,
            sequential_waterfall=waterfall,flow_supported_before_spatial_color=sum(r['flow_supported_before_spatial_color'] for r in rows),
            flow_supported_outside_ROI=sum(r['flow_supported_outside_ROI'] for r in rows),
            flow_supported_inside_ROI_color_failed=sum(r['flow_supported_inside_ROI_color_failed'] for r in rows),
            retained=sum(r['retained_count'] for r in rows)))
    result['status']='complete'
    frozen.write(output/'audit.json',result)
    return {side:part['aggregate'] for side,part in result['sides'].items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=BASE)
    args=parser.parse_args()
    import json
    print(json.dumps(frozen.clean(run(args.output)),ensure_ascii=False,indent=2,allow_nan=False))
