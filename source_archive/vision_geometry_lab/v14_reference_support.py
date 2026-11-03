"""Full trusted downstream edge support audit, independent of prediction.

The denominator is the complete quality-clipped trusted annotation after the
seed plane, including portions outside the input strip or without usable target
columns. No model probability or predicted geometry is accepted by this API.
Ground truth is used for this explicit coverage audit only.
"""
import hashlib
import json
import math

import numpy as np

from .v11_path_contract import _checked_geometry, image_to_strip


PROTOCOL = dict(name='v14_full_trusted_boundary_reference_support',version=1,
    reference='quality supervision override for ALL target polylines; otherwise original trusted annotation',
    seed='unchanged original free_seed defines downstream half-plane',
    clipping='unchanged V6 clip_scoring_segments, original ignore plus quality exclusions, no bridges',
    denominator='ALL resulting trusted downstream native-image arc length, not successful target columns',
    quadrature='each finite segment split into equal-length cells <=2 native px; weighted cell midpoints',
    input_support='mapped strip coordinate in pixel-center bounds; bilinear IMAGE-valid mask equals 1',
    target_support='input-supported and semantic edge mask AND assigned valid low/high head at nearest integer strip column; never seek a valid column',
    unavailable='empty trusted reference returns null fractions, never success',
    prediction_input=False,physical_accuracy_validated=False)
MAX_NATIVE_SAMPLE_STEP=2.
SEMANTIC_EDGES=('edge_flange','edge_surface')


def protocol_sha256():
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _binary(value,shape,name):
    array=np.asarray(value)
    if array.shape!=shape or (array.dtype!=np.bool_ and not np.isin(array,[0,1]).all()):
        raise ValueError(name+' must be a binary array with shape '+str(shape))
    return array.astype(bool,copy=False)


def _input_supported(strip_xy,image_valid):
    """Bilinear footprint must contain only real pixels, not nearest-mask lookup."""
    points=np.asarray(strip_xy,float)
    height,width=image_valid.shape
    x,y=points[:,0],points[:,1]
    inside=(x>=0)&(x<=width-1)&(y>=0)&(y<=height-1)
    answer=np.zeros(len(points),bool)
    if not inside.any():
        return answer
    u,v=x[inside],y[inside]
    x0,y0=np.floor(u).astype(int),np.floor(v).astype(int)
    x1,y1=np.minimum(x0+1,width-1),np.minimum(y0+1,height-1)
    fx,fy=u-x0,v-y0
    mask=image_valid.astype(float,copy=False)
    value=(1-fx)*(1-fy)*mask[y0,x0]+fx*(1-fy)*mask[y0,x1]+(1-fx)*fy*mask[y1,x0]+fx*fy*mask[y1,x1]
    answer[inside]=value>=1.-1e-8
    return answer


def _edge_support(pieces,geometry,image_valid,target_columns):
    total,supported,scorable=0.,0.,0.
    sample_count,input_count,target_count=0,0,0
    outside_count,padding_count,unknown_count=0,0,0
    piece_lengths=[]
    height,width=image_valid.shape
    for piece in pieces:
        length=0.
        for a,b in zip(np.asarray(piece,float)[:-1],np.asarray(piece,float)[1:]):
            size=float(np.linalg.norm(b-a))
            if size<=1e-12:
                continue
            cells=max(1,int(math.ceil(size/MAX_NATIVE_SAMPLE_STEP)))
            weight=size/cells
            points=a+((np.arange(cells)+.5)/cells)[:,None]*(b-a)
            mapped=image_to_strip(points,geometry)
            available=_input_supported(mapped,image_valid)
            inside=(mapped[:,0]>=0)&(mapped[:,0]<=width-1)&(mapped[:,1]>=0)&(mapped[:,1]<=height-1)
            # Use nearest GRID column, not nearest TARGET-valid column.
            columns=np.clip(np.floor(mapped[:,0]+.5),0,width-1).astype(int)
            valid_target=available&target_columns[columns]
            total+=size
            length+=size
            supported+=float(available.sum())*weight
            scorable+=float(valid_target.sum())*weight
            sample_count+=cells
            input_count+=int(available.sum())
            target_count+=int(valid_target.sum())
            outside_count+=int((~inside).sum())
            padding_count+=int((inside&~available).sum())
            unknown_count+=int((available&~valid_target).sum())
        if length>0:
            piece_lengths.append(length)
    return dict(status='scored_reference_support' if total>0 else 'no_trusted_downstream_reference',
        trusted_downstream_length_px=total,input_supported_length_px=supported,scorable_target_length_px=scorable,
        input_support_fraction=supported/total if total else None,
        scorable_fraction_of_trusted=scorable/total if total else None,
        scorable_fraction_of_input=scorable/supported if supported else None,
        trusted_piece_count=len(piece_lengths),trusted_piece_lengths_px=piece_lengths,
        midpoint_samples=sample_count,input_supported_midpoints=input_count,scorable_midpoints=target_count,
        outside_strip_midpoints=outside_count,unsupported_bilinear_footprint_midpoints=padding_count,
        input_supported_but_target_unknown_midpoints=unknown_count,
        target_valid_columns=int(target_columns.sum()),sample_step_max_native_px=MAX_NATIVE_SAMPLE_STEP,
        denominator_restricted_to_target_success=False,missing_reference_is_success=False)


def boundary_reference_support(label,quality,geometry,image_valid,targets):
    """Audit original trusted reference scope before/after V11 input/target masks.

    Returns `edges.edge_flange` and `edges.edge_surface`. Each reports complete
    native-image downstream length, input-supported length, scorable length and
    their fractions. C07/C08 use validated quality overrides (C7 cutoff).
    `targets` supplies only semantic edge column-valid masks and diagnostics.
    It is never used to crop the reference denominator or to decode predictions.
    The caller supplies already hash-validated label/quality dictionaries.
    """
    from .centerline_path_metrics_v6 import clip_scoring_segments
    _checked_geometry(geometry)
    shape=tuple(geometry['shape_hw'])
    valid=_binary(image_valid,shape,'image_valid')
    source=quality.get('supervision_polylines',label['polylines'])
    if label.get('status') not in ('clear','partial'):
        source={}
    seed=label['polylines']['free_seed']
    excluded=label.get('ignore_polygons',[])+quality.get('precision_exclude_polygons',[])
    edges={}
    source_heads=np.asarray(targets['edge_source'])
    head_valid=_binary(targets['edge_valid'],(2,shape[1]),'edge_valid')
    if source_heads.shape!=(2,shape[1]) or not np.isin(source_heads,[0,1,2]).all():
        raise ValueError('edge_source must contain canonical head assignment codes 0,1,2')
    for code,name in enumerate(SEMANTIC_EDGES,1):
        columns=_binary(targets[name+'_valid'],(shape[1],),name+'_valid')
        assigned=((source_heads==code)&head_valid).any(axis=0)
        line=source.get(name,[])
        pieces=clip_scoring_segments([line] if len(line)>=2 else [],free_seed=seed,
            downstream=True,ignore_polygons=excluded)
        edges[name]=_edge_support(pieces,geometry,valid,columns&assigned)
        semantic=_edge_support(pieces,geometry,valid,columns)
        edges[name]['semantic_scorable_length_px']=semantic['scorable_target_length_px']
        edges[name]['semantic_scorable_fraction_of_trusted']=semantic['scorable_fraction_of_trusted']
        edges[name]['semantic_valid_but_unassigned_columns']=int((columns&~assigned).sum())
    total=sum(e['trusted_downstream_length_px'] for e in edges.values())
    supported=sum(e['input_supported_length_px'] for e in edges.values())
    scorable=sum(e['scorable_target_length_px'] for e in edges.values())
    return dict(protocol=PROTOCOL.copy(),protocol_sha256=protocol_sha256(),edges=edges,
        combined=dict(trusted_downstream_length_px=total,input_supported_length_px=supported,
            scorable_target_length_px=scorable,input_support_fraction=supported/total if total else None,
            scorable_fraction_of_trusted=scorable/total if total else None),
        reference_source='hash_bound_quality_polylines' if 'supervision_polylines' in quality else 'original_partial_annotation',
        target_builder_diagnostics=targets.get('diagnostics',{}),
        quality_and_gt_used_for_support_audit_only=True,prediction_input=False,
        physical_accuracy_validated=False,unknown_reference_not_background=True)
