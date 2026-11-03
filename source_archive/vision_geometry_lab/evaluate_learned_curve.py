"""Pre-registered local/full SAM2 experiment; no training on six review points.

Measurements are prompt-conditional support diagnostics, NOT mask/centerline
accuracy. The six rough dots are one frame, not six independent test cases.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image,ImageDraw
from .geometry import read_image,write_image,dump_json

ROOT=Path(__file__).resolve().parents[1]
LAB=Path(__file__).resolve().parent
OUTPUT=LAB/'outputs/learned_curve_v4/sam2'


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(value):
    payload=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def point_mask_distances(mask,points):
    points=np.asarray(points,float).reshape(-1,2)
    if not len(points): return []
    if not np.any(mask): return [None]*len(points)
    distance=cv2.distanceTransform((~mask.astype(bool)).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    result=[]
    for x,y in points:
        if not (0<=x<mask.shape[1] and 0<=y<mask.shape[0]): result.append(None)
        else: result.append(float(distance[min(mask.shape[0]-1,round(y)),min(mask.shape[1]-1,round(x))]))
    return result


def anchored_component(mask,anchor):
    _,components,_,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    x,y=np.rint(anchor).astype(int)
    if not (0<=x<mask.shape[1] and 0<=y<mask.shape[0]): return np.zeros_like(mask,bool)
    label=int(components[y,x])
    return components==label if label else np.zeros_like(mask,bool)


def crop_roi(shape,center,size):
    center=np.asarray(center,float);size=np.asarray(size,int)
    low=np.maximum(0,np.rint(center-size/2).astype(int))
    high=np.minimum([shape[1],shape[0]],low+size)
    return [*low.tolist(),*high.tolist()]


def intersect_rois(*rois):
    """Return the non-empty intersection of rectangular ROIs, else ``None``."""
    valid=[roi for roi in rois if roi is not None]
    if not valid:
        return None
    x0=max(int(roi[0]) for roi in valid);y0=max(int(roi[1]) for roi in valid)
    x1=min(int(roi[2]) for roi in valid);y1=min(int(roi[3]) for roi in valid)
    return [x0,y0,x1,y1] if x0<x1 and y0<y1 else None


def roi_contains(outer,inner):
    return bool(outer is not None and inner is not None and
                outer[0]<=inner[0] and outer[1]<=inner[1] and
                outer[2]>=inner[2] and outer[3]>=inner[3])


def foreground_fraction(mask,roi):
    if roi is None:
        return None
    x0,y0,x1,y1=map(int,roi)
    if x0>=x1 or y0>=y1:
        return None
    region=mask[y0:y1,x0:x1]
    return float(region.mean()) if region.size else None


def _point_membership(mask,point):
    x,y=map(float,point)
    if not (0<=x<mask.shape[1] and 0<=y<mask.shape[0]):
        return False
    ix=min(mask.shape[1]-1,int(round(x)));iy=min(mask.shape[0]-1,int(round(y)))
    return bool(mask[iy,ix])


def prompt_compliance(mask,component,points,labels):
    points=np.asarray(points,float).reshape(-1,2)
    labels=np.asarray(labels,int).reshape(-1)
    if len(points)!=len(labels):
        raise ValueError('Prompt labels must match points')
    inside=[_point_membership(mask,p) for p in points]
    positive=[inside[i] for i,label in enumerate(labels) if label==1]
    negative_outside=[not inside[i] for i,label in enumerate(labels) if label==0]
    positive_connected=[_point_membership(component,points[i]) for i,label in enumerate(labels) if label==1]
    return dict(
        prompt_inside_mask=inside,
        positive_prompt_inside=positive,
        negative_prompt_outside=negative_outside,
        positive_prompt_satisfaction_fraction=(float(np.mean(positive)) if positive else None),
        negative_prompt_satisfaction_fraction=(float(np.mean(negative_outside)) if negative_outside else None),
        prompt_constraints_satisfied=(bool(all(positive) and all(negative_outside)) if positive else None),
        positive_prompt_in_anchor_component=positive_connected,
        all_positive_prompts_anchor_connected=(bool(all(positive_connected)) if positive_connected else None))


def support_within_half_width(mask,points,width):
    points=np.asarray(points,float).reshape(-1,2)
    if not len(points):
        return None
    distances=point_mask_distances(mask,points)
    return float(sum(d is not None and d<=width*.5 for d in distances)/len(distances))


def _points_in_roi(points,roi):
    points=np.asarray(points,float).reshape(-1,2)
    if roi is None or not len(points):
        return points[:0]
    x0,y0,x1,y1=roi
    return points[(points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1)]


def _coarse_region_audit(mask,case_roi,regions):
    result={}
    for name,box in regions.items():
        overlap=intersect_rois(case_roi,box)
        box_area=max(0,int(box[2])-int(box[0]))*max(0,int(box[3])-int(box[1]))
        overlap_area=0 if overlap is None else (overlap[2]-overlap[0])*(overlap[3]-overlap[1])
        result[name]=dict(
            audit_region=list(map(int,box)),
            case_roi_overlap=overlap,
            case_roi_overlap_fraction=(float(overlap_area/box_area) if box_area else None),
            foreground_fraction_in_case_overlap=foreground_fraction(mask,overlap))
    return result


def mask_diagnostics(crop_mask,frame_shape,case,audit):
    """Compute the same diagnostics for a selected mask and every hypothesis."""
    height,width=map(int,frame_shape[:2]);x0,y0,x1,y1=map(int,case['roi'])
    crop_mask=np.asarray(crop_mask,bool)
    if crop_mask.shape!=(y1-y0,x1-x0):
        raise ValueError(f"Mask shape mismatch for {case['name']}: {crop_mask.shape}")
    mask=np.zeros((height,width),bool);mask[y0:y1,x0:x1]=crop_mask
    local_anchor=np.asarray(case['points'][0],float)-[x0,y0]
    crop_component=anchored_component(crop_mask,local_anchor)
    component=np.zeros_like(mask);component[y0:y1,x0:x1]=crop_component
    count,_,stats,_=cv2.connectedComponentsWithStats(crop_mask.astype(np.uint8))

    reference=audit.get('reference',{})
    reference_frame=reference.get('frame_index',246)
    same_reference_frame=case.get('frame_index')==reference_frame
    reference_points=np.asarray(reference.get('points_original_px',[]),float).reshape(-1,2)
    raw_distances=point_mask_distances(mask,reference_points) if same_reference_frame else []
    connected_distances=point_mask_distances(component,reference_points) if same_reference_frame else []

    case_visible_roi=intersect_rois(case['roi'],audit.get('common_roi'))
    fixed_roi=audit.get('fixed_sameframe_common_roi')
    fixed_comparable=bool(same_reference_frame and roi_contains(case['roi'],fixed_roi))
    fixed_metric_roi=fixed_roi if fixed_comparable else None

    if same_reference_frame and audit.get('free_segment') is not None:
        free_samples=np.linspace(*np.asarray(audit['free_segment'],float),40)
        case_samples=_points_in_roi(free_samples,case['roi'])
        fixed_samples=_points_in_roi(free_samples,fixed_metric_roi)
    else:
        case_samples=np.empty((0,2),float);fixed_samples=np.empty((0,2),float)
    width_px=float(audit.get('optical_width_px',0.))

    diagnostics=dict(
        all_reference_distances_px=raw_distances,
        connected_reference_distances_px=connected_distances,
        component_count=int(count-1),
        largest_component_pixels=int(max(stats[1:,cv2.CC_STAT_AREA],default=0)),
        anchor_component_pixels=int(crop_component.sum()),
        case_visible_audit_roi=case_visible_roi,
        case_visible_audit_roi_pixels=(None if case_visible_roi is None else
            int((case_visible_roi[2]-case_visible_roi[0])*(case_visible_roi[3]-case_visible_roi[1]))),
        foreground_fraction_in_case_visible_audit_roi=foreground_fraction(mask,case_visible_roi),
        fixed_sameframe_common_roi=fixed_metric_roi,
        fixed_sameframe_common_roi_pixels=(None if fixed_metric_roi is None else
            int((fixed_metric_roi[2]-fixed_metric_roi[0])*(fixed_metric_roi[3]-fixed_metric_roi[1]))),
        foreground_fraction_in_fixed_sameframe_common_roi=foreground_fraction(mask,fixed_metric_roi),
        coarse_audit_regions=_coarse_region_audit(mask,case['roi'],audit.get('audit_regions',{})),
        case_visible_free_span_samples=(int(len(case_samples)) if same_reference_frame else None),
        raw_free_span_support_within_half_width_case_roi=(
            support_within_half_width(mask,case_samples,width_px) if same_reference_frame else None),
        connected_free_span_support_within_half_width_case_roi=(
            support_within_half_width(component,case_samples,width_px) if same_reference_frame else None),
        fixed_sameframe_free_span_samples=(int(len(fixed_samples)) if fixed_comparable else None),
        raw_free_span_support_within_half_width_fixed_sameframe_common_roi=(
            support_within_half_width(mask,fixed_samples,width_px) if fixed_comparable else None),
        connected_free_span_support_within_half_width_fixed_sameframe_common_roi=(
            support_within_half_width(component,fixed_samples,width_px) if fixed_comparable else None))
    diagnostics.update(prompt_compliance(mask,component,case['points'],case['labels']))
    return mask,component,diagnostics


def prepare():
    reference=json.loads((LAB/'outputs/wound_curve_v3/manual_247/reference.json').read_text('utf-8'))
    baseline=json.loads((LAB/'outputs/wound_curve_v3/manual_247/automatic.json').read_text('utf-8'))
    video=Path(reference['clean_source'])
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,reference['frame_index'])
    ok,frame=cap.read()
    if not ok: raise RuntimeError('Cannot decode source')
    fingerprint=hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
    if fingerprint!=reference['original_frame_sha256']: raise ValueError('Clean frame fingerprint changed')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    clean=OUTPUT/'clean_ui0247.png';write_image(clean,frame)
    rope=baseline['rope']; segment=np.asarray(rope['observed_segment'],float)
    anchor=.1*segment[0]+.9*segment[1]
    width=float(rope['width_px']); direction=(segment[1]-segment[0])/np.linalg.norm(segment[1]-segment[0])
    normal=np.array([-direction[1],direction[0]])
    points=np.asarray(reference['points_original_px'],float)
    # These recipes are fixed before mask inference. The historical focus crop
    # predates the red dots; the local crop uses ONLY detected free-span geometry.
    rois={'full':[0,0,frame.shape[1],frame.shape[0]],
          'focus':[340,1922,2773,2846],
          'local':crop_roi(frame.shape,segment[1]+direction*560,[1536,1024])}
    specs=[('free_only',[anchor.tolist()],[]),
           ('manual_p1',[anchor.tolist(),points[0].tolist()],[0]),
           ('manual_p123',[anchor.tolist(),*points[:3].tolist()],[0,1,2]),
           ('manual_all6',[anchor.tolist(),*points.tolist()],list(range(6)))]
    cases=[]
    for kind,roi in rois.items():
        for prompt_kind,prompts,used in specs:
            cases.append(dict(name=f'{kind}_{prompt_kind}',image=str(clean),roi=roi,
                              points=prompts,labels=[1]*len(prompts),manual_input_indices=used,
                              frame_index=246,kind='main',roi_kind=kind))
    # Fixed perturbations are sensitivity probes, not new ground truth.
    for sign in (-1,1):
        shifted=(points+sign*.5*width*normal).tolist()
        cases.append(dict(name=f'local_all6_shift_{sign:+d}',image=str(clean),roi=rois['local'],
                          points=[anchor.tolist(),*shifted],labels=[1]*7,manual_input_indices=list(range(6)),
                          frame_index=246,kind='prompt_perturbation',roi_kind='local'))
    blank=OUTPUT/'blank_control.png';write_image(blank,np.full_like(frame,100))
    cases.append(dict(name='local_blank_all6',image=str(blank),roi=rois['local'],
                      points=[anchor.tolist(),*points.tolist()],labels=[1]*7,manual_input_indices=list(range(6)),
                      frame_index=None,kind='blank_control',roi_kind='local'))
    # Adjacent frame masks are shown for review, not scored against stale dots.
    for index in (245,247):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,other=cap.read()
        if not ok: raise RuntimeError('Cannot decode adjacent frame')
        path=OUTPUT/f'clean_ui{index+1:04d}.png';write_image(path,other)
        known=json.loads((LAB/f'outputs/wound_curve_v3/ui_{index+1:04d}_t{index/10:.3f}.json').read_text('utf-8'))
        seg=np.asarray(known['rope']['observed_segment'],float);point=.1*seg[0]+.9*seg[1]
        d=(seg[1]-seg[0])/np.linalg.norm(seg[1]-seg[0])
        roi=crop_roi(other.shape,seg[1]+d*560,[1536,1024])
        cases.append(dict(name=f'local_ui{index+1:04d}_free_only',image=str(path),roi=roi,
                          points=[point.tolist()],labels=[1],manual_input_indices=[],
                          frame_index=index,kind='adjacent_unlabelled',roi_kind='local'))
    cap.release()
    # Predeclared diagnostics far from intended route. These are NOT training
    # negatives and not truth masks: occupancy is only a warning of broad masks.
    audit=dict(reference=reference,free_anchor=anchor.tolist(),optical_width_px=width,
               free_segment=segment.tolist(),clean_sha256=fingerprint,
               common_roi=rois['focus'],rois=rois,
               fixed_sameframe_common_roi=intersect_rois(*rois.values()),
               audit_regions={'yellow_flange':[1650,1980,2150,2120],
                              'deep_wound_surface':[1650,2610,2200,2750]},
               limitation='rough route dots, not center/edge ground truth; one labelled image; no accuracy claim')
    protocol=dict(schema_version=1,output=str(OUTPUT),cases=cases,
                  mask_selection='highest SAM score, frozen before diagnostics',
                  roi_selection='fixed pre-red-dot focus crop and free-span-derived local crop; no heldout point selection',
                  branch_type='primary_pre_registered_cases',
                  protocol_status='primary cases frozen before mask inference',
                  primary_pre_registered=True,exploratory_after_primary=False,
                  post_hoc=False,blind=False,
                  training=False,hardware_access=False)
    dump_json(OUTPUT/'protocol.json',protocol);dump_json(OUTPUT/'audit_reference.json',audit)
    return protocol,audit


def validate_worker_bundle(protocol,audit):
    """Recompute every reuse-sensitive fingerprint before reading diagnostics."""
    output=Path(protocol['output']);protocol_path=output/'protocol.json'
    if not protocol_path.is_file():
        raise ValueError(f'Missing protocol file: {protocol_path}')
    saved_protocol=json.loads(protocol_path.read_text('utf-8'))
    if saved_protocol!=protocol:
        raise ValueError(f'Protocol object does not match saved protocol: {protocol_path}')
    protocol_sha256=_file_sha256(protocol_path)
    worker_path=output/'worker_results.json'
    if not worker_path.is_file():
        raise ValueError(f'Missing worker results: {worker_path}')
    worker=json.loads(worker_path.read_text('utf-8'))
    if worker.get('protocol_sha256')!=protocol_sha256:
        raise ValueError(f'protocol_sha256 mismatch: {protocol_path}')

    cases=protocol.get('cases')
    if not isinstance(cases,list):
        raise ValueError('Protocol cases must be a list')
    names=[case.get('name') for case in cases]
    if len(names)!=len(set(names)):
        raise ValueError('Protocol contains duplicate case names')
    worker_cases=worker.get('cases')
    if not isinstance(worker_cases,list) or len(worker_cases)!=len(cases):
        raise ValueError(f'worker_results case count mismatch: {worker_path}')
    worker_by_name={record.get('name'):record for record in worker_cases}
    if set(worker_by_name)!=set(names):
        raise ValueError(f'worker_results case names mismatch: {worker_path}')

    checkpoint_hashes=set()
    clean_checked=False
    reference_frame=audit.get('reference',{}).get('frame_index',246)
    for case in cases:
        name=case['name'];folder=output/name;inference_path=folder/'inference.json'
        if not inference_path.is_file():
            raise ValueError(f'Missing inference record for {name}: {inference_path}')
        record=json.loads(inference_path.read_text('utf-8'))
        if record!=worker_by_name[name]:
            raise ValueError(f'worker_results/inference mismatch for case {name}')
        if record.get('protocol_sha256')!=protocol_sha256:
            raise ValueError(f'protocol_sha256 mismatch for case {name}')
        expected_case_hash=_json_sha256(case)
        if record.get('case_sha256')!=expected_case_hash:
            raise ValueError(f'case_sha256 mismatch for case {name}')
        if any(record.get(key)!=value for key,value in case.items()):
            raise ValueError(f'Inference case fields mismatch for case {name}')

        image_path=Path(case['image'])
        if not image_path.is_file() or record.get('image_sha256')!=_file_sha256(image_path):
            raise ValueError(f'image_sha256 mismatch for case {name}: {image_path}')
        if not clean_checked and case.get('frame_index')==reference_frame and audit.get('clean_sha256'):
            frame=read_image(image_path)
            decoded_hash=hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
            if decoded_hash!=audit['clean_sha256']:
                raise ValueError(f'clean decoded-frame fingerprint mismatch for case {name}: {image_path}')
            clean_checked=True

        checkpoint_path=Path(record.get('checkpoint',''))
        if not checkpoint_path.is_file():
            raise ValueError(f'Missing checkpoint for case {name}: {checkpoint_path}')
        checkpoint_hash=_file_sha256(checkpoint_path)
        if record.get('checkpoint_sha256')!=checkpoint_hash:
            raise ValueError(f'checkpoint_sha256 mismatch for case {name}: {checkpoint_path}')
        checkpoint_hashes.add(checkpoint_hash)

        scores=np.asarray(record.get('scores',[]),float)
        if not len(scores) or not np.isfinite(scores).all():
            raise ValueError(f'Invalid model scores for case {name}')
        selected=record.get('selected_index')
        if selected!=int(np.argmax(scores)):
            raise ValueError(f'Selected mask is not argmax model score for case {name}')
        mask_hashes=record.get('mask_sha256')
        if not isinstance(mask_hashes,list) or len(mask_hashes)!=len(scores):
            raise ValueError(f'mask_sha256 count mismatch for case {name}')
        x0,y0,x1,y1=map(int,case['roi'])
        for index,expected_hash in enumerate(mask_hashes):
            mask_path=folder/f'mask_{index}.png'
            if not mask_path.is_file() or _file_sha256(mask_path)!=expected_hash:
                raise ValueError(f'mask_sha256 mismatch for case {name}: {mask_path}')
            mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape!=(y1-y0,x1-x0):
                raise ValueError(f'Mask dimensions mismatch for case {name}: {mask_path}')

    if len(checkpoint_hashes)!=1 or worker.get('checkpoint_sha256') not in checkpoint_hashes:
        raise ValueError(f'worker_results checkpoint_sha256 mismatch: {worker_path}')
    metadata_keys=('branch_type','protocol_status','primary_pre_registered',
                   'exploratory_after_primary','post_hoc','blind','parent_protocol_sha256',
                   'negative_semantics','training','hardware_access')
    for key in metadata_keys:
        if key in protocol and worker.get(key)!=protocol[key]:
            raise ValueError(f'worker_results missing or mismatched protocol metadata: {key}')
    return dict(protocol_sha256=protocol_sha256,
                checkpoint_sha256=next(iter(checkpoint_hashes)))


def evaluate(protocol,audit):
    output=Path(protocol['output'])
    integrity=validate_worker_bundle(protocol,audit)
    rows=[];tiles=[]
    ref=np.asarray(audit['reference']['points_original_px'],float)
    width=audit['optical_width_px']
    for case in protocol['cases']:
        folder=output/case['name']; record=json.loads((folder/'inference.json').read_text('utf-8'))
        frame=read_image(case['image'])
        hypotheses=[];selected_mask=None;selected_audit=None
        for index,score in enumerate(record['scores']):
            candidate=cv2.imread(str(folder/f'mask_{index}.png'),cv2.IMREAD_GRAYSCALE)>0
            full_candidate,_,candidate_audit=mask_diagnostics(candidate,frame.shape,case,audit)
            hypothesis=dict(index=index,model_score=score,selected=index==record['selected_index'],
                            **candidate_audit)
            hypotheses.append(hypothesis)
            if index==record['selected_index']:
                selected_mask=full_candidate;selected_audit=candidate_audit
        if selected_mask is None:
            raise ValueError(f"Selected mask index missing for case {case['name']}")
        mask=selected_mask
        heldout=[i for i in range(6) if i not in case['manual_input_indices']] if case['kind']=='main' else []
        all_dist=selected_audit['all_reference_distances_px']
        connected=selected_audit['connected_reference_distances_px']
        row=dict(name=case['name'],kind=case['kind'],selected_index=record['selected_index'],
                 scores=record['scores'],selection=record['selection'],
                 heldout_indices_one_based=[i+1 for i in heldout],
                 heldout_distances_px=[all_dist[i] for i in heldout],
                 heldout_connected_distances_px=[connected[i] for i in heldout],
                 identity_confirmed=False,accuracy_validated=False)
        row.update(selected_audit)
        # This nested copy makes it mechanically checkable that the selected
        # proposal received exactly the same audit as every other hypothesis.
        row['selected_mask_audit']=dict(selected_audit)
        # Same P4-P6 across all non-all-six main cases; these remain development
        # diagnostics (previously inspected), not a blind test.
        if case['kind']=='main' and not any(i in case['manual_input_indices'] for i in (3,4,5)):
            row['common_p456_within_half_width']=sum(connected[i] is not None and connected[i]<=width*.5 for i in (3,4,5))
        # Retain all hypotheses for audit. NEVER select using these diagnostics.
        row['all_hypotheses_audit_only']=hypotheses
        shown=frame.copy();shown[mask]=(.60*shown[mask]+.40*np.array([220,80,200])).astype(np.uint8)
        # Always mark INPUT in cyan; held-out approximate references in orange.
        for point,label in zip(case['points'],case['labels']):
            point=tuple(np.rint(point).astype(int))
            if label: cv2.circle(shown,point,8,(255,255,0),3,cv2.LINE_AA)
            else: cv2.drawMarker(shown,point,(0,0,255),cv2.MARKER_TILTED_CROSS,20,3)
        if case['frame_index']==246:
            for i,p in enumerate(ref):
                if i not in case['manual_input_indices']:
                    cv2.circle(shown,tuple(np.rint(p).astype(int)),10,(0,170,255),3,cv2.LINE_AA)
                    cv2.putText(shown,str(i+1),tuple(np.rint(p+[12,-12]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,170,255),2)
        cx0,cy0,cx1,cy1=audit['common_roi']
        focus=shown[cy0:cy1,cx0:cx1].copy()
        cv2.rectangle(focus,(0,0),(focus.shape[1],75),(16,16,16),-1)
        title=case['name']+' | CYAN=positive; RED=negative; ORANGE=reference | NOT VERIFIED'
        cv2.putText(focus,title,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
        cv2.putText(focus,'model score is NOT localization accuracy; no physical contact / actuator output',
                    (12,58),cv2.FONT_HERSHEY_SIMPLEX,.65,(230,230,230),1)
        write_image(folder/'focus.jpg',focus);dump_json(folder/'diagnostics.json',row)
        thumb=Image.fromarray(cv2.cvtColor(focus,cv2.COLOR_BGR2RGB));thumb.thumbnail((800,340))
        tile=Image.new('RGB',(820,390),'#111827');tile.paste(thumb,(10,35))
        ImageDraw.Draw(tile).text((10,10),case['name'],fill='white');tiles.append(tile)
        rows.append(row)
    for start in range(0,len(tiles),6):
        batch=tiles[start:start+6];sheet=Image.new('RGB',(1640,390*((len(batch)+1)//2)),'#111827')
        for i,tile in enumerate(batch): sheet.paste(tile,((i%2)*820,(i//2)*390))
        sheet.save(output/f'contact_sheet_{start//6+1}.jpg')
    metadata_keys=('branch_type','protocol_status','primary_pre_registered',
                   'exploratory_after_primary','post_hoc','blind','parent_protocol_sha256',
                   'negative_semantics','training','hardware_access')
    summary=dict(cases=len(rows),frames_with_rough_reference_points=1,rows=rows,
                 mask_selection=protocol['mask_selection'],integrity=integrity,
                 conclusion='support diagnostics only; no centreline/edge ground truth; no deployment acceptance',
                 hardware_access=False,models_trained=False)
    summary.update({key:protocol[key] for key in metadata_keys if key in protocol})
    json.dumps(summary,ensure_ascii=False,allow_nan=False)
    dump_json(output/'summary.json',summary)
    print(json.dumps({r['name']:dict(p456=r.get('common_p456_within_half_width'),
          distances=r['heldout_connected_distances_px'],
          fixed_area=r['foreground_fraction_in_fixed_sameframe_common_roi'],
          case_area=r['foreground_fraction_in_case_visible_audit_roi']) for r in rows},indent=2))


def build_negative_probe_protocol(original,audit,parent_protocol_sha256):
    points=np.asarray(audit['reference']['points_original_px'],float)
    directions=np.gradient(points,axis=0)
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    normals=np.column_stack([-directions[:,1],directions[:,0]])
    negatives=np.concatenate([points+1.5*audit['optical_width_px']*normals,
                              points-1.5*audit['optical_width_px']*normals]).tolist()
    cases=[]
    for roi_kind in ('focus','local'):
        base=next(c for c in original['cases'] if c['name']==roi_kind+'_manual_all6')
        cases.append(dict(base,name=roi_kind+'_all6_negative_probe',kind='exploratory_negative_probe',
                          source_case_name=base['name'],
                          negative_prompt_semantics='six route-dot normal offsets at +/-1.5 optical widths; heuristic only',
                          points=[*base['points'],*negatives],labels=[*base['labels'],*[0]*len(negatives)]))
    return dict(original,output=str(Path(original['output'])/'negative_probe'),cases=cases,
                branch_type='exploratory_negative_prompt_probe',
                protocol_status='exploratory; specified after observing primary failures',
                primary_pre_registered=False,exploratory_after_primary=True,
                post_hoc=True,blind=False,parent_protocol_sha256=parent_protocol_sha256,
                negative_semantics='heuristic offset samples, not independently labelled true negatives',
                training=False,hardware_access=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-inference',action='store_true')
    parser.add_argument('--negative-prompt-probe',action='store_true',
                        help='Separate exploratory branch after primary results; not a blind test')
    args=parser.parse_args();cv2.setNumThreads(2)
    if args.negative_prompt_probe:
        original=json.loads((OUTPUT/'protocol.json').read_text('utf-8'))
        audit=json.loads((OUTPUT/'audit_reference.json').read_text('utf-8'))
        protocol=build_negative_probe_protocol(original,audit,_file_sha256(OUTPUT/'protocol.json'))
        dump_json(Path(protocol['output'])/'protocol.json',protocol)
        request=Path(protocol['output'])/'protocol.json'
    elif args.reuse_inference:
        protocol=json.loads((OUTPUT/'protocol.json').read_text('utf-8'))
        audit=json.loads((OUTPUT/'audit_reference.json').read_text('utf-8'))
    else:
        protocol,audit=prepare()
        request=OUTPUT/'protocol.json'
    if not args.reuse_inference or args.negative_prompt_probe:
        subprocess.run([str(ROOT/'vision_training/.venv/Scripts/python.exe'),'-X','utf8',
                        str(LAB/'learned_curve_worker.py'),str(request)],check=True,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    evaluate(protocol,audit)


if __name__=='__main__': main()
