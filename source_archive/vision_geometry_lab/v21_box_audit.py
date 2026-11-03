"""Read-only attachment/frame binding and old V18/V19 spatial coverage audit.

This script does not detect a bulge, change frozen algorithms, use event truth,
or confirm a layer. It examines only the requested frame and nearby binding
controls. The attachment can be photometrically edited, so exact and approximate
frame correspondence are explicitly distinguished.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vision_geometry_lab import v19_front_trial as v19


def image_read(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('image unavailable '+str(path))
    return image


def _inside(points, box):
    x1, y1, x2, y2 = box
    return (np.isfinite(points).all(axis=-1) & (points[..., 0] >= x1) & (points[..., 0] <= x2)
            & (points[..., 1] >= y1) & (points[..., 1] <= y2))


def _save_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('PNG encoding failed')
    with Path(path).open('xb') as f:
        f.write(encoded.tobytes())


def audit(attachment, output, side='left', frame_index=273, nearby=5):
    attachment, output = Path(attachment), Path(output)
    if output.exists():
        raise FileExistsError('new audit output directory required')
    p19 = v19.verify(v19.BASE)
    source_base = Path(p19['upstream_base'])
    index = v19._read(source_base/side/'index.json')
    pred = v19._read(v19.BASE/side/'predictions.json')
    if (pred['source_measurement_sha256'] != v19.sha(source_base/side/'measurement.npz')
            or pred['source_index_sha256'] != v19.sha(source_base/side/'index.json')):
        raise ValueError('V19 input binding invalid')
    with np.load(source_base/side/'measurement.npz', allow_pickle=False) as z:
        data = {key:z[key].copy() for key in ('rows_y','times_s','baseline_x','baseline_nx',
            'baseline_ny','transforms','pose_valid','normal_offset','normal_profiles')}
    pos = next(i for i,r in enumerate(index['records']) if r['frame_index'] == frame_index)
    annotated = image_read(attachment)
    if annotated.shape[:2] != (index['video']['height'], index['video']['width']):
        raise ValueError('attachment dimensions differ from source; no identity mapping assumed')
    # Isolate largest red annotation component; every difference statistic uses
    # the unmodified-looking nonred area, excluding an 8px annotation halo.
    red = ((annotated[:,:,2] > 200) & (annotated[:,:,1] < 70) & (annotated[:,:,0] < 70))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    if count < 2:
        raise ValueError('no red annotation component')
    label = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA])); red = labels == label
    ys, xs = np.nonzero(red)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    # At the rectangle centreline, distinguish the inner aperture from stroke.
    midx, midy = (box[0]+box[2])//2, (box[1]+box[3])//2
    rx = np.flatnonzero(red[midy]); ry = np.flatnonzero(red[:, midx])
    if not ((rx < midx).any() and (rx > midx).any() and (ry < midy).any() and (ry > midy).any()):
        raise ValueError('red component is not a complete rectangular outline')
    inner = [int(rx[rx < midx].max()+1), int(ry[ry < midy].max()+1),
             int(rx[rx > midx].min()-1), int(ry[ry > midy].min()-1)]
    keep = ~cv2.dilate(red.astype(np.uint8), np.ones((17,17),np.uint8)).astype(bool)
    first, last = max(0,frame_index-nearby), min(index['video']['frame_count']-1,frame_index+nearby)
    sources = {r['frame_index']:r for r in index['records']}
    cap = cv2.VideoCapture(index['video']['path']); cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    controls = []; requested = None
    try:
        for fid in range(first,last+1):
            ok, raw = cap.read()
            if not ok or v19.upstream.prior.pixels_sha(raw) != sources[fid]['frame_sha256']:
                raise ValueError('decoded source frame hash mismatch')
            delta = abs(annotated.astype(np.int16)-raw.astype(np.int16))
            values = delta[keep]
            controls.append(dict(frame_index=fid,time_s=sources[fid]['time_s'],
                source_frame_sha256=sources[fid]['frame_sha256'],
                nonred_channel_MAE=float(values.mean()), nonred_channel_max_error=int(values.max()),
                nonred_exact_pixel_fraction=float(np.mean(np.all(values==0,axis=1))),
                nonred_channel_p95_error=float(np.percentile(values,95))))
            if fid == frame_index:
                requested = raw
    finally:
        cap.release()
    ranked = sorted(controls,key=lambda r:r['nonred_channel_MAE'])
    if requested is None:
        raise ValueError('requested source missing')
    # A geometric consistency diagnostic, not a license to assert an exact frame
    # identity. ECC maps source template coordinates to annotated input pixels.
    scale = .25
    template = cv2.resize(cv2.cvtColor(requested,cv2.COLOR_BGR2GRAY),(0,0),fx=scale,fy=scale).astype(np.float32)/255.
    inp = cv2.resize(cv2.cvtColor(annotated,cv2.COLOR_BGR2GRAY),(0,0),fx=scale,fy=scale).astype(np.float32)/255.
    mask = cv2.resize(keep.astype(np.uint8)*255,(template.shape[1],template.shape[0]),interpolation=cv2.INTER_NEAREST)
    warp = np.eye(2,3,dtype=np.float32)
    try:
        corr, warp = cv2.findTransformECC(template,inp,warp,cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-7),mask,5)
        fullwarp = warp.astype(float); fullwarp[:,2] /= scale
        corners = np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]],float)
        inv = cv2.invertAffineTransform(fullwarp)
        mapped_corners = corners@inv[:,:2].T+inv[:,2]
        geometry = dict(available=True,ECC=float(corr),source_to_attachment_affine=fullwarp,
            attachment_box_corners_mapped_to_source=mapped_corners,
            maximum_box_corner_identity_mapping_difference_px=float(np.linalg.norm(mapped_corners-corners,axis=1).max()))
    except cv2.error as exc:
        geometry = dict(available=False,error=str(exc))
    jcount = len(data['rows_y']); rows = data['rows_y']; A = data['transforms'][pos]
    refs = np.column_stack([data['baseline_x'],rows])
    normals = np.column_stack([data['baseline_nx'],data['baseline_ny']])
    tangents = np.column_stack([normals[:,1],-normals[:,0]])
    supported = np.isfinite(refs).all(axis=1)&np.isfinite(normals).all(axis=1)
    stripref = (refs[:,None,None,:]+np.arange(-65,66)[None,None,:,None]*normals[:,None,None,:]
                +np.arange(-3,4)[None,:,None,None]*tangents[:,None,None,:])
    def into_raw(points):
        current = points@A[:,:2].T+A[:,2]
        if side == 'left':
            current[...,0] = annotated.shape[1]-1-current[...,0]
        return current
    strip = into_raw(stripref); centers = into_raw(refs)
    peaks = into_raw(refs+data['normal_offset'][pos,:,None]*normals)
    touches = _inside(strip,box).any(axis=(1,2)) & supported
    touches_inner = _inside(strip,inner).any(axis=(1,2)) & supported
    centers_inside = _inside(centers,box)&supported; peaks_inside = _inside(peaks,box)&supported
    finite_peak = np.isfinite(data['normal_offset'][pos])&supported
    finite_profile = np.isfinite(data['normal_profiles'][pos]).all(axis=1)&supported
    qrecord = pred['records'][pos]
    if qrecord['frame_index'] != frame_index or qrecord['time_s'] != index['records'][pos]['time_s']:
        raise ValueError('V19 requested row binding mismatch')
    quality = np.asarray(qrecord['quality_mask'],bool)
    states = qrecord['tracker']['row_states']
    per_rows = []
    for j in np.flatnonzero(touches):
        per_rows.append(dict(index=int(j),reference_y=float(rows[j]),baseline_center_raw_xy=centers[j],
            normal_peak_raw_xy=peaks[j],peak_inside_box=bool(peaks_inside[j]),
            center_inside_box=bool(centers_inside[j]),strip_touches_inner=bool(touches_inner[j]),
            V18_current_peak_finite=bool(finite_peak[j]),V18_current_profile_finite=bool(finite_profile[j]),
            V19_quality=bool(quality[j]),V19_quality_reason=qrecord['quality_reason'][j],
            V19_profile_residual=qrecord['profile_residual'][j],V19_tracker_row=states[j]))
    canonical_box = [annotated.shape[1]-1-box[2],box[1],annotated.shape[1]-1-box[0],box[3]] if side=='left' else box
    result = dict(schema='v21_attachment_box_coverage_audit_1',source_frame_index=frame_index,
        source_time_s=index['records'][pos]['time_s'],source_video=index['video'],
        source_frame_sha256=index['records'][pos]['frame_sha256'],attachment_path=str(attachment),
        attachment_sha256=v19.sha(attachment),attachment_shape=annotated.shape,
        source_index_sha256=v19.sha(source_base/side/'index.json'),
        source_measurement_sha256=v19.sha(source_base/side/'measurement.npz'),
        V19_predictions_sha256=v19.sha(v19.BASE/side/'predictions.json'),
        audit_source_sha256=v19.sha(Path(__file__)),
        red_box_outer_inclusive_raw_xyxy=box,red_box_inner_inclusive_raw_xyxy=inner,
        red_box_outer_canonical_identity_xyxy=canonical_box,red_component_pixels=int(red.sum()),
        excluded_red_halo_pixels=int((~keep).sum()),binding_controls=controls,
        closest_tested_frame=ranked[0],next_closest_tested_frame=ranked[1] if len(ranked)>1 else None,
        exact_unmodified_pixel_copy=bool(ranked[0]['nonred_exact_pixel_fraction']==1.),
        binding_scope='local_neighbor_comparison_and_affine_consistency_not_exact_pixel_identity',
        geometric_consistency=geometry,
        coverage_coordinate_assumption='identity source coordinates; consult affine difference diagnostic',
        pose_valid=bool(data['pose_valid'][pos]),reference_to_current_canonical_affine=A,
        coverage_counts=dict(total_original_grid_rows=jcount,baseline_supported_rows=int(supported.sum()),
            strips_touch_outer_box=int(touches.sum()),strips_touch_inner_box=int(touches_inner.sum()),
            baseline_centers_inside_outer_box=int(centers_inside.sum()),
            finite_current_peaks_inside_outer_box=int((peaks_inside&finite_peak).sum()),
            touched_rows_current_profile_finite=int((touches&finite_profile).sum()),
            touched_rows_current_peak_finite=int((touches&finite_peak).sum()),
            touched_rows_V19_quality_true=int((touches&quality).sum()),
            peak_inside_rows_V19_quality_true=int((peaks_inside&quality).sum())),
        touched_rows=per_rows,V19_new_candidates=qrecord['tracker']['new_candidates'],
        bulge_detected=None,new_layer_confirmed=False,physical_revolution_confirmed=False,
        conclusion_scope='sampling coverage and stored image-quality state only, not event validation')
    output.mkdir(parents=True)
    # Diagnostic overlay on the *verified raw source*, not on the annotation.
    overlay = requested.copy()
    cv2.rectangle(overlay,tuple(box[:2]),tuple(box[2:]),(0,0,255),4)
    for j in np.flatnonzero(touches):
        endpts = into_raw(np.stack([refs[j]-65*normals[j],refs[j]+65*normals[j]]))
        pts = np.rint(endpts).astype(int)
        cv2.line(overlay,tuple(pts[0]),tuple(pts[1]),(150,150,150),1,cv2.LINE_AA)
        if finite_peak[j]:
            color=(0,255,0) if quality[j] else (0,165,255)
            cv2.circle(overlay,tuple(np.rint(peaks[j]).astype(int)),4,color,-1,cv2.LINE_AA)
    cropbox=[max(0,box[0]-160),max(0,box[1]-120),min(overlay.shape[1],box[2]+160),min(overlay.shape[0],box[3]+120)]
    _save_png(output/'coverage_raw_crop.png',overlay[cropbox[1]:cropbox[3]+1,cropbox[0]:cropbox[2]+1])
    _save_png(output/'source_unmarked_crop.png',requested[cropbox[1]:cropbox[3]+1,cropbox[0]:cropbox[2]+1])
    result['diagnostic_crop_xyxy_inclusive']=cropbox
    result['overlay_legend']='red=attachment rectangle; gray=old normal search axis; green=V19 quality true selected peak; orange=quality false selected peak'
    v19._write(output/'box_audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attachment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--side',choices=('left','right'),default='left')
    p.add_argument('--frame-index',type=int,default=273);p.add_argument('--nearby',type=int,default=5)
    a=p.parse_args();r=audit(a.attachment,a.output,a.side,a.frame_index,a.nearby)
    print(v19.strict_json(dict(binding=r['closest_tested_frame'],geometry=r['geometric_consistency'],counts=r['coverage_counts'])))


if __name__=='__main__':main()
