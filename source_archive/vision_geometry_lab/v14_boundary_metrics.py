"""NumPy-only boundary scoring, separate from GT-independent image-valid decode.

No field data is accessed at import or in tests. Predictions are never sorted,
repaired, recentered, or decoded inside a GT/quality mask. Ground truth affects
only which columns can be scored. A missing trusted prediction is not a hit;
an unknown annotation is not background. No physical/layer assertion is made.
"""
import hashlib
import json

import numpy as np

from .v11_path_contract import STEP_WIDTHS, _checked_geometry


PROTOCOL = dict(name='v14_independent_boundary_scoring',version=1,
    decode='argmax separately for raw low/high heads over IMAGE-valid rows only',
    prediction_missing='no image-valid rows or no positive probability mass on image-valid rows',
    tie='first smallest valid row; no confidence/GT threshold',
    gt_use='scoring and trusted-column masks only; never decoding or prediction repair',
    edge_tolerance_native_px=5.,
    pixel_error='absolute strip-row error times unchanged V11 native_px_per_row',
    width='signed raw (pred_high-pred_low)*native_px_per_row; never sorted or absolute-valued',
    relative_width_error='abs(pred_width-true_width)/true_width on trusted positive reference widths',
    center_bracketing='raw pred_low < trusted human center < raw pred_high; strict, not repaired',
    crossing='raw pred_low > raw pred_high; coincident rows reported separately',
    denominators='edge hit and center bracketing include missing trusted predictions as failures; '
        'error quantiles, width errors and crossing rate conditional on available predictions, with missing counts',
    averaging='equal trusted strip columns, not independent frames or arc-length path scoring',
    unknown='not scored as background or success',
    physical_accuracy_validated=False,rope_identity_confirmed=False)
SIDE_NAMES = ('low_normal','high_normal')


def protocol_sha256():
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _bool_mask(value,shape,name):
    mask=np.asarray(value)
    if mask.shape!=shape:
        raise ValueError(name+' has unexpected shape')
    if mask.dtype!=np.bool_ and not np.isin(mask,[0,1]).all():
        raise ValueError(name+' must be boolean or binary')
    return mask.astype(bool,copy=False)


def decode_boundaries(probabilities,image_valid):
    """Decode two heads without accepting, reading, or constructing any GT mask.

    Returns float row indices and independent validity masks [2,W]. Invalid
    coordinates are zero-filled only behind validity=false. Probability maps
    need not sum to one, but must be finite/nonnegative. All-zero mass is
    explicitly missing, not an arbitrary row-zero prediction.
    """
    probability=np.asarray(probabilities)
    if (probability.ndim!=3 or probability.shape[0]!=2 or
            min(probability.shape[1:])<1 or not np.isfinite(probability).all() or
            np.any(probability<0)):
        raise ValueError('Expected finite nonnegative probabilities [2,H,W]')
    valid=_bool_mask(image_valid,probability.shape[1:],'image_valid')
    masked=np.where(valid[None],probability,-np.inf)
    rows=np.argmax(masked,axis=1).astype(np.float64)
    positive=np.max(np.where(valid[None],probability,0.),axis=1)>0
    available=valid.any(axis=0)[None]&positive
    rows[~available]=0.
    return dict(edge_rows=rows,edge_valid=available,
                targets_used=False,edges_sorted_or_repaired=False)


def _checked_targets(targets,height,width):
    edge_valid=_bool_mask(targets['edge_valid'],(2,width),'target.edge_valid')
    center_valid=_bool_mask(targets['center_valid'],(width,),'target.center_valid')
    width_valid=_bool_mask(targets['width_valid'],(width,),'target.width_valid')
    result=dict(edge_valid=edge_valid,center_valid=center_valid,width_valid=width_valid)
    for key,shape,mask in (('edge_rows',(2,width),edge_valid),
            ('center_rows',(width,),center_valid),('local_width_px',(width,),width_valid)):
        value=np.asarray(targets[key],float)
        if value.shape!=shape or not np.isfinite(value[mask]).all():
            raise ValueError('Invalid trusted target coordinates: '+key)
        if key!='local_width_px' and (np.any(value[mask]<0) or np.any(value[mask]>height-1)):
            raise ValueError('Trusted target lies outside image-valid strip coordinate bounds')
        result[key]=value
    if np.any(width_valid & ~(edge_valid.all(axis=0)&center_valid)):
        raise ValueError('Trusted width requires trusted center and both edges')
    if np.any(result['local_width_px'][width_valid]<=0):
        raise ValueError('Trusted local width must be positive')
    if np.any(width_valid & ~((result['edge_rows'][0]<result['center_rows']) &
                              (result['center_rows']<result['edge_rows'][1]))):
        raise ValueError('Trusted width must bracket the trusted center')
    return result


def _summary(values):
    values=np.asarray(values,float)
    return dict(count=int(len(values)),mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p95=float(np.percentile(values,95)) if len(values) else None)


def _edge_summary(trusted_count,errors,statuses=None):
    errors=list(map(float,errors))
    good=sum(e<=PROTOCOL['edge_tolerance_native_px']+1e-9 for e in errors)
    stats=_summary(errors)
    output=dict(trusted_columns=int(trusted_count),predicted_trusted_columns=len(errors),
        missing_prediction_columns=int(trusted_count-len(errors)),hit_5px_count=int(good),
        hit_5px_fraction=good/trusted_count if trusted_count else None,
        error_median_px=stats['median'],error_p95_px=stats['p95'],error_mean_px=stats['mean'],
        absolute_errors_px=errors,
        error_stats_are_conditional_on_available_prediction=True)
    if statuses is not None:
        output['column_status']=statuses
    return output


def score_boundaries(probabilities,image_valid,targets,geometry):
    """Decode before reading GT, then score V13 trusted columns independently.

    `targets` is a V13 array dictionary. Its row_valid / edge_row_valid and
    Gaussian distributions are intentionally not read for prediction decoding.
    Unknown target coordinates may be NaN: only valid=true coordinates matter.
    The returned raw decoded coordinates can be checked for GT invariance.
    """
    decoded=decode_boundaries(probabilities,image_valid)
    height,width=np.asarray(probabilities).shape[1:]
    _checked_geometry(geometry)
    if geometry['shape_hw']!=[height,width]:
        raise ValueError('Geometry and probability map shapes differ')
    native_step=float(geometry['native_px_per_row'])
    if (not np.isfinite(native_step) or native_step<=0 or
            not np.isclose(native_step,float(geometry['width_px'])*STEP_WIDTHS,rtol=1e-10,atol=1e-10)):
        raise ValueError('Invalid original-image pixel scale')
    target=_checked_targets(targets,height,width)
    mask=target['width_valid']
    derived=(target['edge_rows'][1,mask]-target['edge_rows'][0,mask])*native_step
    if not np.allclose(derived,target['local_width_px'][mask],rtol=1e-5,atol=1e-4):
        raise ValueError('Trusted width differs from trusted raw edge difference')
    predictions,available=decoded['edge_rows'],decoded['edge_valid']
    edges={}
    for side,name in enumerate(SIDE_NAMES):
        trusted=target['edge_valid'][side]
        score=trusted&available[side]
        errors=np.abs(predictions[side,score]-target['edge_rows'][side,score])*native_step
        status=np.full(width,'unknown',dtype=object)
        status[trusted]='missing_prediction'
        status[score]='scored'
        edges[name]=_edge_summary(int(trusted.sum()),errors,status.tolist())
    both=target['edge_valid'].all(axis=0)
    pair=both&available.all(axis=0)
    raw_difference=predictions[1]-predictions[0]
    crossing=int(np.sum(pair&(raw_difference<0)))
    coincident=int(np.sum(pair&(raw_difference==0)))
    center_scope=both&target['center_valid']
    center_predictions=center_scope&available.all(axis=0)
    bracketed=center_predictions&(predictions[0]<target['center_rows'])&(target['center_rows']<predictions[1])
    joint=dict(trusted_both_columns=int(both.sum()),paired_prediction_columns=int(pair.sum()),
        missing_pair_columns=int((both&~available.all(axis=0)).sum()),crossing_count=crossing,
        crossing_fraction_among_available=crossing/int(pair.sum()) if pair.any() else None,
        coincident_count=coincident,
        coincident_fraction_among_available=coincident/int(pair.sum()) if pair.any() else None,
        center_trusted_both_columns=int(center_scope.sum()),center_paired_prediction_columns=int(center_predictions.sum()),
        center_missing_pair_columns=int((center_scope&~available.all(axis=0)).sum()),
        strict_center_bracket_count=int(bracketed.sum()),
        strict_center_bracket_fraction=int(bracketed.sum())/int(center_scope.sum()) if center_scope.any() else None,
        reference_center='trusted HUMAN center, not a predicted center',predictions_sorted_or_repaired=False)
    width_scope=target['width_valid']
    width_predicted=width_scope&available.all(axis=0)
    signed_widths=raw_difference[width_predicted]*native_step
    true_widths=target['local_width_px'][width_predicted]
    absolute=np.abs(signed_widths-true_widths)
    relative=absolute/true_widths
    abs_stats,relative_stats=_summary(absolute),_summary(relative)
    width_metrics=dict(trusted_columns=int(width_scope.sum()),predicted_trusted_columns=int(width_predicted.sum()),
        missing_prediction_columns=int((width_scope&~available.all(axis=0)).sum()),
        nonpositive_raw_prediction_count=int(np.sum(signed_widths<=0)),
        mae_px=abs_stats['mean'],absolute_error_median_px=abs_stats['median'],absolute_error_p95_px=abs_stats['p95'],
        mean_absolute_relative_error=relative_stats['mean'],relative_error_median=relative_stats['median'],
        relative_error_p95=relative_stats['p95'],absolute_errors_px=absolute.tolist(),relative_errors=relative.tolist(),
        raw_signed_predicted_widths_px=signed_widths.tolist(),reference_widths_px=true_widths.tolist(),
        error_stats_are_conditional_on_available_prediction=True,width_is_physical_diameter=False)
    raw_rows=[[float(value) if valid else None for value,valid in zip(p,mask)]
              for p,mask in zip(predictions,available)]
    return dict(protocol=PROTOCOL.copy(),protocol_sha256=protocol_sha256(),
        decoded=dict(edge_rows=raw_rows,edge_valid=available.tolist(),targets_used=False,
                     edges_sorted_or_repaired=False),
        edges=edges,joint=joint,width=width_metrics,
        native_px_per_row=native_step,targets_used_for_scoring_only=True,
        unknown_columns_not_background=True,rope_identity_confirmed=False,physical_accuracy_validated=False)


def aggregate_boundary_scores(scores):
    """Pool column observations, never average per-image medians/percentiles."""
    scores=list(scores)
    if any(s.get('protocol_sha256')!=protocol_sha256() for s in scores):
        raise ValueError('Cannot combine different boundary scoring protocols')
    edges={name:_edge_summary(sum(s['edges'][name]['trusted_columns'] for s in scores),
        [e for s in scores for e in s['edges'][name]['absolute_errors_px']]) for name in SIDE_NAMES}
    count_keys=('trusted_both_columns','paired_prediction_columns','missing_pair_columns','crossing_count',
        'coincident_count','center_trusted_both_columns','center_paired_prediction_columns',
        'center_missing_pair_columns','strict_center_bracket_count')
    joint={k:sum(s['joint'][k] for s in scores) for k in count_keys}
    pairs=joint['paired_prediction_columns']
    centers=joint['center_trusted_both_columns']
    joint.update(crossing_fraction_among_available=joint['crossing_count']/pairs if pairs else None,
        coincident_fraction_among_available=joint['coincident_count']/pairs if pairs else None,
        strict_center_bracket_fraction=joint['strict_center_bracket_count']/centers if centers else None,
        reference_center='trusted HUMAN center, not a predicted center',predictions_sorted_or_repaired=False)
    absolute=[e for s in scores for e in s['width']['absolute_errors_px']]
    relative=[e for s in scores for e in s['width']['relative_errors']]
    a,r=_summary(absolute),_summary(relative)
    width={k:sum(s['width'][k] for s in scores) for k in ('trusted_columns','predicted_trusted_columns',
        'missing_prediction_columns','nonpositive_raw_prediction_count')}
    width.update(mae_px=a['mean'],absolute_error_median_px=a['median'],absolute_error_p95_px=a['p95'],
        mean_absolute_relative_error=r['mean'],relative_error_median=r['median'],relative_error_p95=r['p95'],
        absolute_errors_px=absolute,relative_errors=relative,
        error_stats_are_conditional_on_available_prediction=True,width_is_physical_diameter=False)
    return dict(samples=len(scores),protocol=PROTOCOL.copy(),protocol_sha256=protocol_sha256(),
        edges=edges,joint=joint,width=width,
        aggregation='equal trusted columns, NOT independent images or arc-length centerline metrics',
        physical_accuracy_validated=False,rope_identity_confirmed=False)
