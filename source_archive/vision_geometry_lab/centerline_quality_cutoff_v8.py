"""Pure geometry for a human-confirmed last-trusted centre point.

No image/model is used to choose the cut. Only the preceding trusted segment
sets its tangent. The optional CLI reads labels and prints a proposed, raw-label
hash-bound V5 quality payload; it never writes a sidecar or changes a label.
Identical inputs produce identical output (no timestamps). Existing sidecars
must be explicitly reviewed/merged, never silently replaced by this helper.

The returned retained_polylines are diagnostics, not label edits. A downstream
rasterizer must clip BEFORE smoothing/resizing if it needs targets independent
of the inferred tail: merely masking a completed heatmap is not sufficient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EPS = 1e-8
LINE_NAMES = ('centerline', 'edge_flange', 'edge_surface', 'free_seed')
TARGET_LINE_NAMES = ('centerline', 'edge_flange', 'edge_surface')
RULE_TYPE = 'centerline_normal_cross_section'
RULE_VERSION = 1


def _dot(a, b):
    return a[0]*b[0]+a[1]*b[1]


def _sub(a, b):
    return [a[0]-b[0], a[1]-b[1]]


def _distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _shape(value):
    if not isinstance(value, (list, tuple)) or len(value)!=2 or any(type(v) is not int or v<2 for v in value):
        raise ValueError('image_shape_hw must contain two integers >= 2')
    return value[0], value[1]


def _line(value, shape, name):
    if not isinstance(value, (list, tuple)) or len(value)<2:
        raise ValueError(f'{name} requires at least two points')
    h,w=shape; result=[]
    for p in value:
        if not isinstance(p, (list, tuple)) or len(p)!=2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p):
            raise ValueError(f'{name} coordinates must be finite numbers')
        x,y=map(float,p)
        if not (0<=x<=w-1 and 0<=y<=h-1):
            raise ValueError(f'{name} outside closed pixel-centre rectangle')
        result.append([x,y])
    return result


def _deduplicate(points):
    result=[]
    for p in points:
        if not result or _distance(p,result[-1])>EPS: result.append(p)
    if len(result)>1 and _distance(result[0],result[-1])<=EPS: result.pop()
    return result


def _clip_rectangle(shape, origin, tangent):
    h,w=shape; rectangle=[[0.,0.],[float(w-1),0.],[float(w-1),float(h-1)],[0.,float(h-1)]]
    polygon=[]
    for a,b in zip(rectangle,rectangle[1:]+rectangle[:1]):
        da,db=_dot(_sub(a,origin),tangent),_dot(_sub(b,origin),tangent)
        inside_a,inside_b=da>=0.,db>=0.
        if inside_a: polygon.append(a)
        if inside_a != inside_b:
            fraction=da/(da-db)
            point=[a[i]+fraction*(b[i]-a[i]) for i in range(2)]
            polygon.append([min(w-1.,max(0.,point[0])),min(h-1.,max(0.,point[1]))])
    polygon=_deduplicate(polygon)
    area2=abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(polygon,polygon[1:]+polygon[:1])))
    if len(polygon)<3 or area2<=EPS:
        raise ValueError('Excluded half-plane has no positive image area')
    return polygon


def _edge_intersection(line, origin, tangent, name):
    distances=[_dot(_sub(p,origin),tangent) for p in line]
    candidates=[]
    for index,(a,b) in enumerate(zip(line,line[1:])):
        da,db=distances[index:index+2]
        if abs(da)<=EPS and abs(db)<=EPS:
            raise ValueError(f'{name} has a segment lying in the cutoff plane')
        if da*db<0. or abs(da)<=EPS or abs(db)<=EPS:
            fraction=da/(da-db)
            if not -EPS<=fraction<=1+EPS: continue
            fraction=min(1.,max(0.,fraction))
            point=[a[i]+fraction*(b[i]-a[i]) for i in range(2)]
            if not any(_distance(point,c['xy'])<=EPS for c in candidates):
                candidates.append(dict(xy=point, source_segment_index_0based=index,
                                       segment_fraction=fraction, extrapolated=False))
    if len(candidates)!=1:
        raise ValueError(f'{name} requires one existing-segment intersection; no extrapolation or multiple crossings')
    forward=distances[0]<-EPS and distances[-1]>=-EPS
    reverse=distances[-1]<-EPS and distances[0]>=-EPS
    if not (forward or reverse):
        raise ValueError(f'{name} must have one trusted prefix ending at or crossing the plane')
    oriented=line if forward else list(reversed(line))
    signed=distances if forward else list(reversed(distances))
    # Disallow touching and subsequently returning to the trusted half-plane.
    touched=False
    for value in signed:
        if value>=-EPS: touched=True
        elif touched: raise ValueError(f'{name} re-enters the trusted half-plane')
    hit=candidates[0]; retained=[]
    for point,value in zip(oriented,signed):
        if value<-EPS: retained.append(point[:])
        else: break
    if not retained or _distance(retained[-1],hit['xy'])>EPS: retained.append(hit['xy'][:])
    if len(retained)<2 or sum(_distance(a,b) for a,b in zip(retained,retained[1:]))<=EPS:
        raise ValueError(f'{name} has no nondegenerate trusted part')
    hit['input_order_reversed_for_retained_prefix']=bool(reverse)
    return hit,retained


def cutoff_geometry(polylines, image_shape_hw, last_trusted_center_index_1based):
    """Return a clipped exclusion half-plane and exact existing-edge cut points.

    The centre point index is explicitly one-based. Its previous trusted point
    and itself alone determine the tangent; later CENTER points only undergo
    safety checks, and never affect the plane, polygon or boundary intersections.
    Edge intersections interpolate the existing segment spanning the section;
    its endpoints may lie on either side, as authorized by the same-section
    cutoff. These intersections are derived geometry, not new measurements.
    All trusted centre/seed points must remain on the retained side. All later
    centre points must be strictly on the excluded side; looping tails require
    a separately confirmed region, not this half-plane shortcut.
    """
    shape=_shape(image_shape_hw)
    if not isinstance(polylines,dict) or set(polylines)!=set(LINE_NAMES):
        raise ValueError('Exactly four review polylines are required')
    lines={name:_line(polylines[name],shape,name) for name in LINE_NAMES}
    center=lines['centerline']; index=last_trusted_center_index_1based
    if type(index) is not int or not 2<=index<len(center):
        raise ValueError('Last trusted one-based index must be >= 2 and precede an inferred tail')
    origin=center[index-1][:]; previous=center[index-2]; tangent=_sub(origin,previous)
    norm=math.hypot(*tangent)
    if norm<=EPS: raise ValueError('Last trusted segment is degenerate')
    tangent=[v/norm for v in tangent]; normal=[-tangent[1],tangent[0]]
    signed=lambda p:_dot(_sub(p,origin),tangent)
    if any(signed(p)>EPS for p in center[:index]):
        raise ValueError('Trusted center prefix crosses into the excluded half-plane')
    if any(signed(p)<=EPS for p in center[index:]):
        raise ValueError('Inferred center tail is not strictly beyond the cutoff plane')
    if any(signed(p)>EPS for p in lines['free_seed']):
        raise ValueError('Free seed crosses into the excluded half-plane')
    intersections={}; retained={'centerline':[p[:] for p in center[:index]],'free_seed':[p[:] for p in lines['free_seed']]}
    for name in ('edge_flange','edge_surface'):
        intersections[name],retained[name]=_edge_intersection(lines[name],origin,tangent,name)
    offsets=[_dot(_sub(intersections[name]['xy'],origin),normal) for name in ('edge_flange','edge_surface')]
    if offsets[0]*offsets[1]>=-EPS*EPS:
        raise ValueError('Existing edge intersections do not bracket the trusted center point')
    polygon=_clip_rectangle(shape,origin,tangent)
    return dict(last_trusted_center_index_1based=index,origin_xy=origin,
                tangent_from_center_indices_1based=[index-1,index],unit_tangent_xy=tangent,
                unit_cross_section_direction_xy=normal,
                excluded_halfplane='dot(xy - origin_xy, unit_tangent_xy) >= 0',
                boundary_policy=f'C{index} is the geometric cut; quality/metrics exclude the boundary (zero arc length)',
                polygon_xy=polygon,image_rectangle_xyxy=[0,0,shape[1]-1,shape[0]-1],
                edge_intersections=intersections,retained_polylines=retained,
                inferred_center_points_used_for_tangent=False,
                labels_modified=False,
                raster_warning='Clip supervision geometry before blur/resize; a post-raster exclusion alone can leak inferred targets into trusted support.')


def build_quality_cutoff_payload(sample, annotation_bytes, last_trusted_center_index_1based=7):
    """Build, but never save, a deterministic V5-compatible quality proposal.

    Caller must already have obtained human confirmation of this index and the
    same cross-section cutoff for both boundaries. Hash binds the ORIGINAL raw
    label bytes, not a canonical/re-serialized JSON representation.
    """
    if not isinstance(annotation_bytes,bytes): raise TypeError('annotation_bytes must be original raw bytes')
    annotation=json.loads(annotation_bytes.decode('utf-8-sig'))
    image_hash=sample.get('image_sha256')
    if not isinstance(image_hash,str) or len(image_hash)!=64 or any(c not in '0123456789abcdef' for c in image_hash):
        raise ValueError('sample image_sha256 must be lowercase SHA256')
    if annotation.get('sample_id')!=sample.get('sample_id') or annotation.get('image_sha256')!=image_hash:
        raise ValueError('Annotation sample/image binding mismatch')
    if annotation.get('schema_version')!=1 or annotation.get('coordinate_system')!='full_image_xy' or annotation.get('reviewed') is not True:
        raise ValueError('Expected reviewed full_image_xy schema-1 annotation')
    if annotation.get('status') not in ('clear','partial'):
        raise ValueError('Cutoff requires a clear/partial annotation')
    geometry=cutoff_geometry(annotation['polylines'],sample['image_shape_hw'],last_trusted_center_index_1based)
    index=last_trusted_center_index_1based
    region=dict(region_id=f'human_confirmed_after_center_c{index}',kind='context_inferred',
                exclude_from_precision=True,polygon_xy=geometry['polygon_xy'],
                confirmation=f'Human confirmed center through C{index}; both edges use the same normal cross-section.',
                cutoff_origin_xy=geometry['origin_xy'],cutoff_unit_tangent_xy=geometry['unit_tangent_xy'],
                tangent_source_indices_1based=geometry['tangent_from_center_indices_1based'])
    payload=dict(schema_version=1,sample_id=sample['sample_id'],image_sha256=image_hash,
                 annotation_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
                 coordinate_system='full_image_xy',regions=[region],
                 trusted_supervision_rule=dict(type=RULE_TYPE,version=RULE_VERSION,
                     last_trusted_center_index_1based=index,region_id=region['region_id']),
                 trusted_supervision_polylines={name:geometry['retained_polylines'][name] for name in TARGET_LINE_NAMES})
    validate_trusted_supervision(sample,annotation,payload)
    return dict(quality_payload=payload,diagnostic=geometry,
                write_policy='Proposal only. If an existing sidecar is byte/semantically identical, leave it unchanged; otherwise explicitly review/merge, never overwrite other regions.')


def _check_coordinates(actual, expected, name):
    if not isinstance(actual,list) or len(actual)!=len(expected):
        raise ValueError(f'{name} does not match recomputed cutoff geometry')
    for value,target in zip(actual,expected):
        if isinstance(target,list):
            _check_coordinates(value,target,name)
        elif (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
              or not math.isclose(value,target,rel_tol=0.,abs_tol=1e-7)):
            raise ValueError(f'{name} does not match recomputed cutoff geometry')


def validate_trusted_supervision(sample, annotation, quality_payload):
    """Validate optional derived targets against one supported geometric rule.

    Returns only the three independently re-computed target lines, or ``None``
    when BOTH optional rule and polylines fields are absent (legacy behavior).
    A rule without its trusted lines is rejected instead of silently restoring
    legacy full-line rasterization. The quality loader
    must verify the original annotation-byte SHA before calling this function;
    it cannot be reconstructed from an already parsed annotation dictionary.
    These lines are for supervision only, never RGB/seed or decoder input.
    """
    if not isinstance(quality_payload,dict): raise ValueError('Quality payload must be an object')
    if 'trusted_supervision_polylines' not in quality_payload:
        if 'trusted_supervision_rule' in quality_payload:
            raise ValueError('trusted_supervision_rule requires trusted_supervision_polylines')
        return None
    for name in ('sample_id','image_sha256'):
        if sample.get(name)!=annotation.get(name) or sample.get(name)!=quality_payload.get(name):
            raise ValueError(f'Trusted supervision {name} binding mismatch')
    if quality_payload.get('schema_version')!=1 or quality_payload.get('coordinate_system')!='full_image_xy':
        raise ValueError('Trusted supervision requires schema-1 full_image_xy quality')
    if annotation.get('schema_version')!=1 or annotation.get('coordinate_system')!='full_image_xy' or annotation.get('reviewed') is not True or annotation.get('status') not in ('clear','partial'):
        raise ValueError('Trusted supervision requires a reviewed clear/partial annotation')
    rule=quality_payload.get('trusted_supervision_rule')
    required={'type','version','last_trusted_center_index_1based','region_id'}
    if (not isinstance(rule,dict) or set(rule)!=required or rule.get('type')!=RULE_TYPE
            or type(rule.get('version')) is not int or rule['version']!=RULE_VERSION):
        raise ValueError('Unsupported trusted supervision type/version/rule fields')
    index=rule['last_trusted_center_index_1based']
    if rule['region_id']!=f'human_confirmed_after_center_c{index}':
        raise ValueError('Trusted supervision region_id must match the confirmed center index')
    geometry=cutoff_geometry(annotation['polylines'],sample['image_shape_hw'],index)
    regions=quality_payload.get('regions')
    if not isinstance(regions,list): raise ValueError('Trusted supervision requires regions')
    matching=[r for r in regions if isinstance(r,dict) and r.get('region_id')==rule['region_id']]
    if len(matching)!=1: raise ValueError('Trusted supervision must refer to exactly one cutoff region')
    region=matching[0]
    if region.get('kind')!='context_inferred' or region.get('exclude_from_precision') is not True:
        raise ValueError('Trusted cutoff region must exclude context-inferred precision')
    for key,expected in (('polygon_xy',geometry['polygon_xy']),
                         ('cutoff_origin_xy',geometry['origin_xy']),
                         ('cutoff_unit_tangent_xy',geometry['unit_tangent_xy']),
                         ('tangent_source_indices_1based',geometry['tangent_from_center_indices_1based'])):
        _check_coordinates(region.get(key),expected,key)
    actual=quality_payload['trusted_supervision_polylines']
    if not isinstance(actual,dict) or set(actual)!=set(TARGET_LINE_NAMES):
        raise ValueError('Trusted supervision must contain exactly three target polylines, never free_seed')
    for name in TARGET_LINE_NAMES:
        _check_coordinates(actual[name],geometry['retained_polylines'][name],f'trusted_supervision_polylines.{name}')
    return {name:[p[:] for p in geometry['retained_polylines'][name]] for name in TARGET_LINE_NAMES}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--sample-id',required=True,action='append')
    parser.add_argument('--last-trusted-center-index',type=int,default=7)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text('utf-8-sig'))
    results=[]
    for identifier in args.sample_id:
        selected=[s for s in manifest['samples'] if s['sample_id']==identifier]
        if len(selected)!=1: raise ValueError('sample-id must select exactly one manifest row')
        sample=selected[0]
        results.append(build_quality_cutoff_payload(sample,Path(sample['annotation_path']).read_bytes(),args.last_trusted_center_index))
    print(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
